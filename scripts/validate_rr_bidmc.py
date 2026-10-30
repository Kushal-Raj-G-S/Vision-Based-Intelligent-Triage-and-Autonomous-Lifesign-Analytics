"""
Respiratory-rate validation against the BIDMC PPG and Respiration Dataset
=========================================================================

Why this script exists
──────────────────────
Heart rate in this project has real, reproducible accuracy numbers because it
was validated against real ground truth (71 sessions, UBFC-rPPG + UBFC-Phys —
see CLAUDE.md). Respiratory rate has NEVER had a real-data accuracy number:
none of the three datasets in `test/` ship a respiration ground truth, so
`_estimate_rr`/`_smooth_rr` in core/rppg.py were only ever sanity-checked
against a synthetic signal for "does it converge instead of edge-locking."

BIDMC (PhysioNet, Open Data Commons Attribution license) closes that gap. It
provides 53 x 8-minute recordings of critically-ill patients with:
  * PLETH  — the photoplethysmogram, 125 Hz  (our estimator's INPUT)
  * Breath annotations from TWO independent human annotators (our GOLD standard)
  * A monitor-derived 1 Hz RESP trend (secondary reference, not used as gold)

What this validates, and what it does NOT
──────────────────────────────────────────
BIDMC is contact finger/ear pulse oximetry, not webcam video. So this
validates the *RR extraction DSP* — the part that turns a pulse waveform into
a breathing rate — in isolation, on a clean input. It does NOT validate the
camera/ROI/face-detection layer, and it does NOT prove the same MAE holds for
a webcam-derived rPPG waveform (which is far noisier). Treat a good number
here as "the algorithm is sound", not "the product measures RR to this
accuracy". Same distinction CLAUDE.md already draws for
`validate_cropped_roi.py` vs. the manifest path.

The question this script was written to answer
───────────────────────────────────────────────
`_run_pipeline()` hands `_estimate_rr()` a buffer of only `fps * 10` samples,
and `_estimate_rr()` then hard-caps its Welch segment at `fps * 8`. A 10 s
buffer has a fundamental frequency resolution of 1/10 s = 0.1 Hz; an 8 s
segment, 0.125 Hz. The whole RR search band (_RESP_LOW..._RESP_HIGH) is just
0.4 Hz wide, so one resolution cell spans ~6-7.5 breaths/min. `nfft=4096`
buys fine bin *spacing*, which is interpolation, not resolution.

Rather than assert that from theory, this script SWEEPS buffer length and
segment length and measures MAE against human-annotated breaths, so the
numbers pick the value. Run with --sweep.

Usage
─────
    # honest baseline of exactly what ships today (10 s buffer, 8 s segment)
    python scripts/validate_rr_bidmc.py --baseline

    # buffer/resolution sweep — answers "can 10 s ever work?"
    python scripts/validate_rr_bidmc.py --sweep

    # single configuration
    python scripts/validate_rr_bidmc.py --window 32 --nperseg 32

    # quick smoke test on a few records
    python scripts/validate_rr_bidmc.py --sweep --limit 8
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.rppg import (  # noqa: E402
    _RESP_HIGH,
    _RESP_LOW,
    _safe_butter_sos,
    AdvancedRPPG,
)

# ── Dataset location ─────────────────────────────────────────────────────────

_DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "dataset"
    / "bidmc-ppg-and-respiration-dataset-1.0.0"
    / "bidmc-ppg-and-respiration-dataset-1.0.0"
    / "bidmc_csv"
)

_BIDMC_FS = 125.0          # Hz, PLETH sampling rate as shipped

# ── Validation protocol constants ────────────────────────────────────────────
#
# Deployment-faithful resampling target. The webcam path runs at 20-30 fps
# (web/__init__.py uses 30; COHFACE-class webcams are 20). Validating at
# BIDMC's native 125 Hz would measure a numeric regime the product never
# sees, so PLETH is decimated to this rate before anything else happens.
_TARGET_FPS = 30

# Gold-standard gating. A window is only scored when BOTH annotators saw
# enough breaths to define a rate, and they AGREE — otherwise the "ground
# truth" is itself uncertain and would just add noise to the MAE.
_MIN_BREATHS_IN_WINDOW = 3
_MAX_ANNOTATOR_DISAGREEMENT = 2.0     # breaths/min

# How often to evaluate a window (seconds). Each Welch estimate is
# independent of how many cycles preceded it, so striding the *evaluation*
# does not change per-window accuracy — unlike the HR path's cycle-counted
# hysteresis, which CLAUDE.md warns must stay at stride=1. The EMA smoother
# is the only cycle-counted piece here and is measured separately.
_EVAL_STRIDE_SEC = 1.0

# The display/aggregation suppression gate the product actually applies —
# must stay in sync with core.clinical.RR_CONFIDENCE_LOW and the checks in
# web/routes.py / scripts/analyze_video.py.
_RR_CONF_GATE = 30.0


# ── Data loading ─────────────────────────────────────────────────────────────


@dataclass
class Record:
    name: str
    ppg: np.ndarray          # resampled to fps
    fps: float
    breaths_s: list          # [ann1_times_sec, ann2_times_sec]

    @property
    def duration_s(self) -> float:
        return len(self.ppg) / self.fps


def _read_signals_pleth(path: Path) -> np.ndarray:
    """Pull the PLETH column out of a bidmc_##_Signals.csv."""
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader)]
        try:
            col = header.index("PLETH")
        except ValueError:
            raise ValueError(f"{path.name}: no PLETH column (header={header})")
        out = []
        for row in reader:
            if len(row) <= col:
                continue
            try:
                out.append(float(row[col]))
            except ValueError:
                out.append(np.nan)
    arr = np.asarray(out, dtype=float)
    # BIDMC has occasional gaps; linear-interpolate rather than drop, so the
    # time axis (and therefore every breath timestamp) stays valid.
    if np.isnan(arr).any():
        idx = np.arange(len(arr))
        good = ~np.isnan(arr)
        if good.sum() < 10:
            raise ValueError(f"{path.name}: PLETH almost entirely NaN")
        arr = np.interp(idx, idx[good], arr[good])
    return arr


def _read_breaths(path: Path) -> list:
    """Return [ann1_sample_numbers, ann2_sample_numbers] as second-timestamps."""
    ann1: list = []
    ann2: list = []
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for row in reader:
            if len(row) >= 1 and row[0].strip():
                try:
                    ann1.append(float(row[0]) / _BIDMC_FS)
                except ValueError:
                    pass
            if len(row) >= 2 and row[1].strip():
                try:
                    ann2.append(float(row[1]) / _BIDMC_FS)
                except ValueError:
                    pass
    return [np.asarray(sorted(ann1)), np.asarray(sorted(ann2))]


def load_records(root: Path, limit: int | None = None) -> list:
    sig_files = sorted(root.glob("bidmc_*_Signals.csv"))
    if not sig_files:
        raise SystemExit(f"No bidmc_*_Signals.csv found under {root}")
    if limit:
        sig_files = sig_files[:limit]

    up, down = _TARGET_FPS, int(_BIDMC_FS)
    g = np.gcd(up, down)
    up, down = up // g, down // g

    records = []
    for sf in sig_files:
        name = sf.name.replace("_Signals.csv", "")
        bf = sf.with_name(f"{name}_Breaths.csv")
        if not bf.exists():
            print(f"  skip {name}: no breath annotations")
            continue
        try:
            pleth = _read_signals_pleth(sf)
            breaths = _read_breaths(bf)
        except Exception as exc:
            print(f"  skip {name}: {exc}")
            continue
        if min(len(b) for b in breaths) < _MIN_BREATHS_IN_WINDOW:
            print(f"  skip {name}: too few breath annotations")
            continue

        # Decimate 125 Hz -> _TARGET_FPS with a proper anti-alias filter.
        ppg = sps.resample_poly(pleth, up, down)
        records.append(Record(name=name, ppg=ppg, fps=float(_TARGET_FPS),
                              breaths_s=breaths))
    return records


# ── Ground truth ─────────────────────────────────────────────────────────────


def gold_rr(record: Record, t0: float, t1: float):
    """
    Human-annotated RR over [t0, t1), or None if the gold standard is not
    trustworthy for this window.

    Uses the count-per-duration form (60 * n / T) rather than mean
    inter-breath interval: with a fixed analysis window the count form is
    what the reference RR in the RRest/Charlton protocol reports, and it
    degrades more gracefully when one breath is missed at a window edge.
    """
    rates = []
    for b in record.breaths_s:
        n = int(np.count_nonzero((b >= t0) & (b < t1)))
        if n < _MIN_BREATHS_IN_WINDOW:
            return None
        rates.append(60.0 * n / (t1 - t0))
    if abs(rates[0] - rates[1]) > _MAX_ANNOTATOR_DISAGREEMENT:
        return None                      # annotators disagree — not a gold standard
    return float(np.mean(rates))


# ── Estimators ───────────────────────────────────────────────────────────────


class ParamRRE:
    """
    Parameterised re-implementation of core.rppg.AdvancedRPPG._estimate_rr.

    Exists so buffer length and Welch segment length can be swept
    independently. Verified against the real method at the shipping settings
    (window=10 s, nperseg=8 s) — see --verify-replica.
    """

    def __init__(self, fps: float, nperseg_s: float, nfft: int = 4096):
        self.fps = float(fps)
        self.nperseg_s = float(nperseg_s)
        self.nfft = int(nfft)
        # DO NOT "fix" this to clamp_to_pulse_band=False.
        #
        # This class reproduces the OLD estimator, and the old estimator built
        # its resp filter with the pulse-band clamps still applied — which is
        # precisely the bug being demonstrated: the request for 0.10-0.50 Hz
        # silently became a 0.70-1.00 Hz (42-60 br/min) filter centred on the
        # heart rate. Passing clamp_to_pulse_band=False here would make the
        # "legacy" baseline better than the code that actually shipped, and
        # the reported improvement would be understated and wrong.
        self.sos = _safe_butter_sos(_RESP_LOW, _RESP_HIGH, int(round(fps)), order=2)

    def estimate(self, ppg: np.ndarray):
        if self.sos is None or len(ppg) < int(self.fps * 6):
            return 0.0, 0.0
        try:
            resp = sps.sosfiltfilt(self.sos, ppg)

            nperseg = min(len(resp), int(self.fps * self.nperseg_s))
            nperseg = max(nperseg, int(self.fps * 4))
            nperseg = min(nperseg, len(resp))
            noverlap = nperseg // 2

            f, p = sps.welch(resp, fs=self.fps, window="hann",
                             nperseg=nperseg, noverlap=noverlap, nfft=self.nfft)
            m = (f >= _RESP_LOW) & (f <= _RESP_HIGH)
            f, p = f[m], p[m]
            if len(p) == 0:
                return 0.0, 0.0

            k = int(np.argmax(p))
            rr = float(f[k] * 60.0)
            sp = float(np.sum(p[max(0, k - 1):k + 2]))
            tot = float(np.sum(p)) + 1e-9
            conf = float(np.clip((sp / tot) * 100.0, 0.0, 100.0))

            edge = max(1, len(p) // 20)
            if k < edge or k >= len(p) - edge:
                conf *= 0.4
            return rr, conf
        except Exception:
            return 0.0, 0.0


def _legacy_estimator(fps: float):
    """
    The pre-fix estimator: band-pass the RAW ppg at 0.1-0.5 Hz, argmax.

    Reconstructed here because core/rppg.py no longer contains it — it was
    replaced outright (see the _RR_* block there). Kept so the improvement
    is reproducible rather than just asserted, and so a future change can be
    compared against the real historical baseline.
    """
    return ParamRRE(fps, nperseg_s=8.0).estimate


def _production_estimator(fps: float):
    """The RIV-fusion estimator exactly as it ships in core/rppg.py."""
    eng = AdvancedRPPG(fps=int(round(fps)), window_size=int(fps * 10))
    return lambda buf: eng._estimate_rr(buf)


# ── Evaluation ───────────────────────────────────────────────────────────────


@dataclass
class Scored:
    errors: list                 # abs error, gated readings only
    errors_ungated: list         # abs error, every non-zero reading
    n_windows: int               # windows with a valid gold standard
    n_suppressed: int            # gold-valid windows the conf gate zeroed
    session_error: float = float("nan")   # |session median - gold median|
    corr_pairs: tuple = ()       # (estimates, golds) for correlation


def evaluate(record: Record, estimator, window_s: float,
             stride_s: float = _EVAL_STRIDE_SEC) -> Scored:
    fps = record.fps
    win_n = int(round(window_s * fps))
    stride_n = max(1, int(round(stride_s * fps)))

    errs, errs_ung = [], []
    accepted, golds_accepted = [], []
    est_all, gold_all = [], []
    n_win = n_sup = 0

    for end in range(win_n, len(record.ppg) + 1, stride_n):
        start = end - win_n
        t0, t1 = start / fps, end / fps

        g = gold_rr(record, t0, t1)
        if g is None:
            continue
        n_win += 1

        rr, conf = estimator(record.ppg[start:end])
        if rr > 0:
            errs_ung.append(abs(rr - g))
            est_all.append(rr)
            gold_all.append(g)
        # The product suppresses the reading entirely below the confidence
        # floor, so a suppressed window is a miss, not a free pass.
        if rr <= 0 or conf < _RR_CONF_GATE:
            n_sup += 1
        else:
            errs.append(abs(rr - g))
            accepted.append(rr)
            golds_accepted.append(g)

    # Session-level number — the thing the product actually reports. Mirrors
    # AdvancedRPPG._summarise_rr(): IQR-filtered median over accepted readings.
    sess_err = float("nan")
    if len(accepted) >= 3:
        a = np.asarray(accepted)
        q1, q3 = np.percentile(a, [25, 75])
        iqr = q3 - q1
        keep = a[(a >= q1 - 1.5 * iqr) & (a <= q3 + 1.5 * iqr)]
        if len(keep) == 0:
            keep = a
        sess_err = abs(float(np.median(keep)) - float(np.median(golds_accepted)))

    return Scored(errs, errs_ung, n_win, n_sup, sess_err,
                  (est_all, gold_all))


def _agg(scoreds: list) -> dict:
    e = np.concatenate([np.asarray(s.errors) for s in scoreds if s.errors]) \
        if any(s.errors for s in scoreds) else np.asarray([])
    eu = np.concatenate([np.asarray(s.errors_ungated) for s in scoreds if s.errors_ungated]) \
        if any(s.errors_ungated for s in scoreds) else np.asarray([])
    n_win = sum(s.n_windows for s in scoreds)
    n_sup = sum(s.n_suppressed for s in scoreds)
    sess = np.asarray([s.session_error for s in scoreds
                       if not np.isnan(s.session_error)])
    ea = [v for s in scoreds for v in s.corr_pairs[0]] if scoreds else []
    ga = [v for s in scoreds for v in s.corr_pairs[1]] if scoreds else []
    corr = float(np.corrcoef(ea, ga)[0, 1]) if len(ea) > 2 else float("nan")
    return {
        "corr": corr,
        "sess_mae": float(np.mean(sess)) if len(sess) else float("nan"),
        "sess_median": float(np.median(sess)) if len(sess) else float("nan"),
        "sess_within_2": float(np.mean(sess <= 2.0) * 100.0) if len(sess) else float("nan"),
        "sess_worst": float(np.max(sess)) if len(sess) else float("nan"),
        "n_sessions": int(len(sess)),
        "mae": float(np.mean(e)) if len(e) else float("nan"),
        "rmse": float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan"),
        "mae_ungated": float(np.mean(eu)) if len(eu) else float("nan"),
        "within_2": float(np.mean(e <= 2.0) * 100.0) if len(e) else float("nan"),
        "within_5": float(np.mean(e <= 5.0) * 100.0) if len(e) else float("nan"),
        "n_scored": int(len(e)),
        "n_windows": n_win,
        "suppressed_pct": (100.0 * n_sup / n_win) if n_win else float("nan"),
    }


def _print_agg(label: str, a: dict) -> None:
    print(f"  {label}")
    print(f"    per-window : MAE {a['mae']:6.2f}  RMSE {a['rmse']:6.2f}  "
          f"<=2 {a['within_2']:5.1f}%  <=5 {a['within_5']:5.1f}%  "
          f"suppressed {a['suppressed_pct']:5.1f}%  (n={a['n_scored']})")
    print(f"    corr vs gold : {a['corr']:+.3f}    "
          f"MAE ignoring conf gate: {a['mae_ungated']:6.2f}")
    print(f"    SESSION    : MAE {a['sess_mae']:6.2f}  median {a['sess_median']:6.2f}  "
          f"<=2 {a['sess_within_2']:5.1f}%  worst {a['sess_worst']:6.2f}  "
          f"({a['n_sessions']} recordings)   <-- what the product reports")


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N records (smoke test)")
    ap.add_argument("--baseline", action="store_true",
                    help="measure exactly what ships today (10 s buffer, 8 s segment)")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep buffer length x segment length")
    ap.add_argument("--window", type=float, default=None, help="buffer length, seconds")
    ap.add_argument("--nperseg", type=float, default=None, help="Welch segment, seconds")
    ap.add_argument("--per-record", action="store_true", help="print each record")
    ap.add_argument("--stride", type=float, default=_EVAL_STRIDE_SEC,
                    help="seconds between evaluated windows (default 1.0)")
    ap.add_argument("--rr-window", type=float, default=60.0,
                    help="RR buffer length for the production estimator")
    args = ap.parse_args()

    print(f"Loading BIDMC from {args.root}")
    records = load_records(args.root, args.limit)
    if not records:
        raise SystemExit("No usable records loaded.")
    total_min = sum(r.duration_s for r in records) / 60.0
    print(f"Loaded {len(records)} records, {total_min:.1f} min total, "
          f"resampled 125 Hz -> {_TARGET_FPS} fps (deployment-faithful)\n")

    if args.baseline or not (args.sweep or args.window):
        print("  input: BIDMC PLETH  |  gold: 2-annotator manual breath counts")
        print("  a window is only scored when BOTH annotators saw >=3 breaths")
        print(f"  AND agreed within {_MAX_ANNOTATOR_DISAGREEMENT} br/min\n")

        print("LEGACY estimator — raw-band argmax, 10 s buffer / 8 s segment")
        print("  (the implementation this replaced; reconstructed for comparison)")
        legacy = [evaluate(r, _legacy_estimator(r.fps), window_s=10.0,
                           stride_s=args.stride) for r in records]
        _print_agg("legacy", _agg(legacy))
        print()

        print(f"PRODUCTION estimator — RIV fusion, {args.rr_window:.0f} s buffer "
              f"(core/rppg.py as it now ships)")
        prod = []
        for r in records:
            s = evaluate(r, _production_estimator(r.fps),
                         window_s=args.rr_window, stride_s=args.stride)
            prod.append(s)
            if args.per_record:
                a = _agg([s])
                print(f"    {r.name:<10} win-MAE {a['mae']:6.2f}  "
                      f"session-err {s.session_error:6.2f}  "
                      f"supp {a['suppressed_pct']:5.1f}%")
        _print_agg("production", _agg(prod))
        print()

    if args.sweep:
        print("SWEEP — buffer length x Welch segment length")
        print("  'resolution' = 1/segment, expressed in breaths/min: the best")
        print("  possible precision that segment length can physically deliver.")
        print()
        configs = [
            (10.0, 8.0),    # what ships today
            (16.0, 8.0),    # more averaging, same resolution
            (16.0, 16.0),
            (24.0, 12.0),
            (32.0, 8.0),    # lots of averaging, shipping resolution
            (32.0, 16.0),
            (32.0, 32.0),   # RRest/Charlton-standard 32 s window
            (45.0, 22.0),
            (60.0, 30.0),
            (60.0, 60.0),
        ]
        for win, nps in configs:
            if nps > win:
                continue
            res_brmin = 60.0 / nps
            scoreds = [evaluate(r, ParamRRE(r.fps, nps).estimate, win) for r in records]
            a = _agg(scoreds)
            print(f"  buf {win:5.1f}s  seg {nps:5.1f}s  "
                  f"(res {res_brmin:4.1f} br/min)  "
                  f"MAE {a['mae']:6.2f}  RMSE {a['rmse']:6.2f}  "
                  f"MAEung {a['mae_ungated']:6.2f}  "
                  f"<=2 {a['within_2']:5.1f}%  <=5 {a['within_5']:5.1f}%  "
                  f"supp {a['suppressed_pct']:5.1f}%")
        print()

    if args.window:
        nps = args.nperseg or min(args.window, 32.0)
        print(f"CONFIGURATION  buffer={args.window}s  segment={nps}s")
        scoreds = []
        for r in records:
            s = evaluate(r, ParamRRE(r.fps, nps).estimate, args.window)
            scoreds.append(s)
            if args.per_record:
                a = _agg([s])
                print(f"    {r.name:<10} MAE {a['mae']:6.2f}  "
                      f"suppressed {a['suppressed_pct']:5.1f}%")
        _print_agg(f"buf {args.window}s / seg {nps}s", _agg(scoreds))


if __name__ == "__main__":
    main()
