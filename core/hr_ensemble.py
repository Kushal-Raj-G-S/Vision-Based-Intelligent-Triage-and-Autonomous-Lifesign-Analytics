"""
Heart-rate ensemble: peak selector + spectro-temporal tracker (numpy / scipy only — no torch, sklearn or joblib).

Why this exists. The classical pipeline (CHROM) and the old ML correction shrink toward the population mean. The two
models here instead CHOOSE a real spectral peak, and they fail on different sessions, so their mean is better than
either: out-of-fold on 125 sessions / 75 people (scripts/hr_lab.py, scripts/hr_deep4.py), MAE ~3.8-3.9 BPM vs 11.4
classical and 4.5 old correction. That figure is video-dataset evidence, NOT a live-webcam validation, and the
"selector + tracker" pair was one of ~22 combinations tried (the pre-specified mean of four models was 3.94) —
see data/evolution/22-hr-ensemble-integration.md before quoting a number.

  selector : HistGradientBoosting classifier scoring up to ~6 spectral-peak candidates per cycle (29 features) ->
             per-cycle argmax -> IQR-filtered median.  Trees exported to models/hr_selector.npz.
  tracker  : conv net over the last 11 CHROM+POS spectra (1 s apart) -> posterior over 116 frequency bins -> argmax
             -> IQR-filtered median.  Weights exported to models/hr_tracker.npz (mean posterior of several seeds).
  answer   : mean of the two session medians (or whichever exists while the other is still warming up).

Both need a CHROM engine (the app's own, with capture_spectra=True) AND a POS engine on the same frames; the POS
engine, the feature building and the two models run on a worker thread so the video loop never waits for them.
Everything here is additive: with the model files missing, or VITAL_HR_ENSEMBLE=0, `load_models()` returns None and the
app behaves exactly as before.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import find_peaks
from scipy.special import erf

log = logging.getLogger("vital.hr_ensemble")

GRID_HZ = np.arange(0.70, 3.0001, 0.02)               # 116 bins, matches AdvancedRPPG._wide_spectrum
GRID_BPM = GRID_HZ * 60.0
NB = len(GRID_HZ)
K_PEAKS = 3
SEL_STRIDE = 5                                         # selector decision every 5 cycles (as evaluated)
TRK_STRIDE = 15                                        # tracker decision every 15 cycles (evaluated at 5; see doc 22)
TRK_START = 150                                        # tracker needs ~5 s of cycles first (as evaluated)
TRK_STEPS, TRK_SPACING = 11, 30                        # 11 spectra, 30 cycles (~1 s) apart
ROLL = 30
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def iqr_median(x) -> float:
    """Same aggregation as AdvancedRPPG.get_final_summary / scripts: median after a 1.5xIQR outlier filter."""
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0:
        return float("nan")
    if len(x) >= 4:
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        keep = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
        if len(keep):
            x = keep
    return float(np.median(x))


# ── selector: exported gradient-boosted trees ───────────────────────────────────────────────────────────────────────
class TreeEnsemble:
    def __init__(self, z):
        self.feat = z["feature_idx"].astype(np.int64)
        self.thr = z["threshold"].astype(np.float64)
        self.left = z["left"].astype(np.int64)
        self.right = z["right"].astype(np.int64)
        self.value = z["value"].astype(np.float64)
        self.is_leaf = z["is_leaf"].astype(bool)
        self.miss_left = z["missing_left"].astype(bool)
        self.roots = z["roots"].astype(np.int64)
        self.baseline = float(z["baseline"])

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, np.float64)
        n = len(X)
        rows = np.arange(n)
        raw = np.full(n, self.baseline)
        for root in self.roots:
            node = np.full(n, root, np.int64)
            for _ in range(64):
                leaf = self.is_leaf[node]
                if leaf.all():
                    break
                x = X[rows, self.feat[node]]
                go_left = np.where(np.isnan(x), self.miss_left[node], x <= self.thr[node])
                node = np.where(leaf, node, np.where(go_left, self.left[node], self.right[node]))
            raw += self.value[node]
        return 1.0 / (1.0 + np.exp(-raw))


# ── tracker: numpy forward pass of scripts/hr_deep4.py::SpecTracker ─────────────────────────────────────────────────
def _gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def _conv1d(x, w, b, pad, dil=1):
    """x (N, Cin, L), w (Cout, Cin, K) -> (N, Cout, L); same-length output, torch Conv1d semantics."""
    n, _, length = x.shape
    k = w.shape[2]
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    cols = np.stack([xp[:, :, i * dil:i * dil + length] for i in range(k)], axis=-1)      # (N, Cin, L, K)
    out = np.tensordot(cols, w, axes=([1, 3], [1, 2]))                                    # (N, L, Cout)
    return out.transpose(0, 2, 1) + b[None, :, None]


class TrackerNet:
    def __init__(self, z, prefix):
        g = lambda k: z[f"{prefix}_{k}"].astype(np.float32)
        self.p = {k: g(k) for k in ("enc_0_weight", "enc_0_bias", "enc_2_weight", "enc_2_bias", "tmp_0_weight", "tmp_0_bias",
                                    "tmp_2_weight", "tmp_2_bias", "head_0_weight", "head_0_bias", "head_3_weight",
                                    "head_3_bias", "head_5_weight", "head_5_bias", "head_7_weight", "head_7_bias", "bias")}

    def logits(self, x):
        """x (B, L, 2, NB) float32 (each spectrum already divided by its own max) -> (B, NB)."""
        p = self.p
        b, t, _, nb = x.shape
        h = _gelu(_conv1d(x.reshape(b * t, 2, nb), p["enc_0_weight"], p["enc_0_bias"], 3))
        h = _gelu(_conv1d(h, p["enc_2_weight"], p["enc_2_bias"], 2))                      # (B*L, ch, NB)
        ch = h.shape[1]
        h = h.reshape(b, t, ch, nb).transpose(0, 3, 2, 1).reshape(b * nb, ch, t)
        h = _gelu(_conv1d(h, p["tmp_0_weight"], p["tmp_0_bias"], 1))
        h = _gelu(_conv1d(h, p["tmp_2_weight"], p["tmp_2_bias"], 1))
        h = np.concatenate([h[:, :, -1], h.mean(2), h.max(2)], 1)                         # (B*NB, 3ch)
        h = h.reshape(b, nb, -1).transpose(0, 2, 1)                                       # (B, 3ch, NB)
        g = np.broadcast_to(h.mean(2, keepdims=True), h.shape)
        h = np.concatenate([h, g], 1)
        h = _gelu(_conv1d(h, p["head_0_weight"], p["head_0_bias"], 2))
        h = _gelu(_conv1d(h, p["head_3_weight"], p["head_3_bias"], 4, 2))
        h = _gelu(_conv1d(h, p["head_5_weight"], p["head_5_bias"], 8, 4))
        return _conv1d(h, p["head_7_weight"], p["head_7_bias"], 0)[:, 0, :] + p["bias"][None, :]


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Models:
    def __init__(self, selector: TreeEnsemble, trackers: list):
        self.selector, self.trackers = selector, trackers

    def tracker_posterior(self, x) -> np.ndarray:
        return sum(_softmax(t.logits(x)) for t in self.trackers) / len(self.trackers)


def load_models(model_dir: Optional[Path] = None) -> Optional[Models]:
    """Return the loaded models, or None (disabled / files missing / unreadable) — callers then fall back."""
    if os.environ.get("VITAL_HR_ENSEMBLE", "1").strip().lower() in ("0", "false", "off", "no"):
        log.info("HR ensemble disabled by VITAL_HR_ENSEMBLE")
        return None
    d = Path(model_dir) if model_dir else MODEL_DIR
    try:
        sel = TreeEnsemble(np.load(d / "hr_selector.npz", allow_pickle=False))
        zt = np.load(d / "hr_tracker.npz", allow_pickle=False)
        nets = [TrackerNet(zt, f"net{j}") for j in range(int(zt["n_nets"]))]
        log.info("HR ensemble loaded: %d trees, %d tracker nets", len(sel.roots), len(nets))
        return Models(sel, nets)
    except Exception as exc:                                  # noqa: BLE001 — any failure must degrade to the old path
        log.warning("HR ensemble unavailable (%s: %s) — using the classical/hybrid path", type(exc).__name__, exc)
        return None


# ── candidate features (exact streaming copy of scripts/hr_peaks.py::candidate_table, one cycle at a time) ───────────
def _peaks(spec):
    sm = np.convolve(spec, np.array([0.25, 0.5, 0.25]), mode="same")
    idx, _ = find_peaks(sm, prominence=sm.max() * 0.05)
    if len(idx) == 0:
        idx = np.array([int(np.argmax(sm))])
    order = np.argsort(-sm[idx])[:K_PEAKS]
    return idx[order], sm


def _bin(bpm):
    return int(np.clip(round((bpm / 60.0 - GRID_HZ[0]) / 0.02), 0, NB - 1))


def candidate_rows(t, h):
    """Rows (and their frequencies in BPM) for cycle t. `h` holds per-cycle lists: bpm, pos_bpm, sqi, snr, stab, harm,
    motion, pos_sqi, chrom_spec, pos_spec."""
    cs, cp = h["chrom_spec"][t], h["pos_spec"][t]
    ic, smc = _peaks(cs)
    ip, smp = _peaks(cp)
    cands = {}
    for rank, i in enumerate(ic):
        cands.setdefault(int(i), {})["rc"] = rank
    for rank, i in enumerate(ip):
        cands.setdefault(int(i), {})["rp"] = rank
    prod_t, pos_t = h["bpm"][t], float(h["pos_bpm"][t])
    cands.setdefault(_bin(prod_t), {})["prod"] = True
    cands.setdefault(_bin(pos_t), {})["posprod"] = True
    merged = []
    for k in sorted(cands):
        if merged and k - merged[-1][0] <= 1:
            old_k, flags = merged[-1]
            flags.update(cands[k])
            if smc[k] + smp[k] > smc[old_k] + smp[old_k]:
                merged[-1][0] = k
        else:
            merged.append([k, dict(cands[k])])
    lo = max(0, t - ROLL)
    win = np.asarray(h["bpm"][lo:t + 1], np.float64)
    rmean, rstd = float(np.mean(win)), float(np.std(win))
    rows, bpms = [], []
    for b, flag in merged:
        f = GRID_BPM[b]
        b2 = _bin(f * 2.0) if f * 2.0 <= 180 else None
        bh = _bin(f / 2.0) if f / 2.0 >= 42 else None
        pc, pp = float(smc[b]), float(smp[b])
        rows.append([
            f, pc, pp, pc + pp,
            flag.get("rc", 9), flag.get("rp", 9), float("prod" in flag), float("posprod" in flag),
            float(smc[b2]) if b2 is not None else 0.0, float(smc[bh]) if bh is not None else 0.0,
            float(smp[b2]) if b2 is not None else 0.0, float(smp[bh]) if bh is not None else 0.0,
            abs(f - prod_t), abs(f - pos_t), abs(prod_t - pos_t),
            f - rmean, rstd, rmean,
            float(h["sqi"][t]), float(h["snr"][t]), float(h["stab"][t]), float(h["harm"][t]),
            float(h["motion"][t]), float(h["pos_sqi"][t]),
            0.0,                                                  # mouth_open: Haar tracker has no landmarks (was NaN->0 in training)
            float(smc.max()), float(smp.max()),
            pc / (smc.max() + 1e-9), pp / (smp.max() + 1e-9),
        ])
        bpms.append(f)
    return np.array(rows, np.float64), np.array(bpms, np.float64)


def seq_indices(k, n):
    return np.clip(k - (TRK_STEPS - 1) * TRK_SPACING + TRK_SPACING * np.arange(TRK_STEPS), 0, n - 1)


class HREnsembleSession:
    """One measurement session. Call feed() from the video loop for every frame; read estimate() any time."""

    def __init__(self, models: Models, fps: int, window_size: int, threaded: bool = True, drop_when_full: bool = True):
        """drop_when_full=True for a live camera (never stall the video; a dropped frame is logged); False for offline
        file analysis, which runs faster than real time and must apply back-pressure instead of losing frames."""
        from core.rppg import AdvancedRPPG                       # local import: core.rppg must not import this module
        self.models = models
        self._drop_when_full = drop_when_full
        self.pos = AdvancedRPPG(fps=fps, window_size=window_size, algorithm="POS")
        self.pos.capture_spectra = True
        self.h = {k: [] for k in ("bpm", "pos_bpm", "sqi", "snr", "stab", "harm", "motion", "pos_sqi", "chrom_spec", "pos_spec")}
        self.sel_hist, self.trk_hist = [], []
        self.lock = threading.Lock()
        self.error: Optional[str] = None
        self._q: "queue.Queue" = queue.Queue(maxsize=600)
        self._threaded, self._stop = threaded, False
        self._t = None
        if threaded:
            self._t = threading.Thread(target=self._run, name="hr-ensemble", daemon=True)
            self._t.start()

    # -- producer side (video loop) -----------------------------------------------------------------------------------
    def feed(self, rgb, motion_delta: float, chrom_result: dict) -> None:
        """rgb = the ROI colour tuple actually given to the main engine this frame (None = no face: skipped, exactly as in
        the research replay). chrom_result = the main engine's process_ppg_signal() dict for this frame."""
        if rgb is None or self.error:
            return
        item = (tuple(float(v) for v in rgb), float(motion_delta), _light(chrom_result))
        if self._threaded:
            if not self._drop_when_full:
                self._q.put(item)
                return
            try:
                self._q.put_nowait(item)
            except queue.Full:                                    # worker fell behind: drop rather than stall the video
                log.warning("HR ensemble queue full — frame dropped")
        else:
            self._step(item)

    def flush(self, timeout: float = 5.0) -> None:
        """Wait (up to `timeout`) until the worker has processed everything queued so far; the worker keeps running."""
        if not self._threaded:
            return
        end = time.time() + timeout
        while self._q.unfinished_tasks and time.time() < end and self._t is not None and self._t.is_alive():
            time.sleep(0.01)

    def finalize(self, timeout: float = 20.0) -> None:
        """TERMINAL: let the worker finish everything queued, then stop it (end of a video / session)."""
        if self._threaded and self._t is not None:
            self._q.put(None)
            self._t.join(timeout)

    def close(self) -> None:
        if self._threaded and self._t is not None and self._t.is_alive():
            self._stop = True
            try:
                self._q.put_nowait(None)
            except queue.Full:
                pass

    # -- worker side --------------------------------------------------------------------------------------------------
    def _run(self):
        while not self._stop:
            item = self._q.get()
            try:
                if item is None:
                    break
                self._step(item)
            finally:
                self._q.task_done()

    def _step(self, item):
        if self.error:
            return
        try:
            rgb, motion, cr = item
            self.pos.add_frame(rgb, motion_delta=motion)
            pr = self.pos.process_ppg_signal()
            if not (cr["ready"] and cr["bpm"] > 0 and pr["ready"] and pr["bpm"] > 0) or "spectrum_wide" not in cr or "spectrum_wide" not in pr:
                return
            h = self.h
            h["bpm"].append(cr["bpm"]); h["pos_bpm"].append(pr["bpm"]); h["sqi"].append(cr["sqi"])
            h["snr"].append(cr["snr_db"]); h["stab"].append(cr["stab"]); h["harm"].append(cr["harm"])
            h["motion"].append(motion); h["pos_sqi"].append(pr.get("sqi", 0.0))
            h["chrom_spec"].append(np.asarray(cr["spectrum_wide"], np.float32))
            h["pos_spec"].append(np.asarray(pr["spectrum_wide"], np.float32))
            t = len(h["bpm"]) - 1
            if t % SEL_STRIDE == 0:
                rows, bpms = candidate_rows(t, h)
                p = self.models.selector.predict_proba(rows)
                with self.lock:
                    self.sel_hist.append(float(bpms[int(np.argmax(p))]))
            if t >= TRK_START and (t - TRK_START) % TRK_STRIDE == 0:
                idx = seq_indices(t, t + 1)
                x = np.empty((1, TRK_STEPS, 2, NB), np.float32)
                x[0, :, 0] = np.stack([h["chrom_spec"][i] for i in idx])
                x[0, :, 1] = np.stack([h["pos_spec"][i] for i in idx])
                x /= (x.max(3, keepdims=True) + 1e-9)
                post = self.models.tracker_posterior(x)
                with self.lock:
                    self.trk_hist.append(float(GRID_BPM[int(post[0].argmax())]))
        except Exception as exc:                                  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("HR ensemble worker failed — disabling for this session")

    # -- consumer side ------------------------------------------------------------------------------------------------
    def estimate(self) -> dict:
        with self.lock:
            sel, trk = list(self.sel_hist), list(self.trk_hist)
        s = iqr_median(sel) if len(sel) >= 3 else float("nan")
        k = iqr_median(trk) if len(trk) >= 3 else float("nan")
        parts = [v for v in (s, k) if np.isfinite(v)]
        bpm = float(np.mean(parts)) if parts else None
        return {"bpm": bpm, "selector": None if not np.isfinite(s) else s, "tracker": None if not np.isfinite(k) else k,
                "n_selector": len(sel), "n_tracker": len(trk), "error": self.error}


def _light(res: dict) -> dict:
    """Keep only what the ensemble needs from a process_ppg_signal() result (don't hold the big arrays)."""
    return {"ready": bool(res.get("ready")), "bpm": float(res.get("bpm", 0) or 0), "sqi": float(res.get("sqi", 0.0) or 0.0),
            "snr_db": float(res.get("snr_db", 0.0) or 0.0), "stab": float(res.get("stability_score", 0.0) or 0.0),
            "harm": float(res.get("harmonic_ratio", 0.0) or 0.0), "spectrum_wide": res.get("spectrum_wide")}
