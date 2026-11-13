"""
Proof-of-concept report: HOW a heart-rate number came out of one recording, stage by stage, with pictures.

Runs as a SEPARATE low-priority process (core/poc_jobs.py starts it), so it can never slow down or break the app.

    python -m core.poc_report --video clip.avi --out poc/patients/Name/<run> --name "Name" --source upload [--truth 98.3]
    python -m core.poc_report --trace poc/patients/Name/<run>/live_trace.npz --out poc/patients/Name/<run> --name "Name" --source live

Output folder (one per test):
    01_frames_and_rgb/          frames, red/green/blue channel frames, average R,G,B per frame, which colour contributes more and WHY
    02_pulse_signals_chrom_pos/ CHROM and POS pulse waveforms (raw projection -> after band-pass, heartbeats marked)
    03_welch_spectrum/          Welch frequency spectrum of the band-passed pulse
    04_ensemble/                what the selector and the tracker see and choose; the final answer
    summary.txt, result.json    the numbers
Heart rate only (respiration is not part of this report).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LIVE_FS = 30.0
CH = ["R", "G", "B"]
COL = {"R": "#d62728", "G": "#2ca02c", "B": "#1f77b4"}
# the three skin patches as fractions of the detected face box (x, y, w, h) - identical to core/camera.py
PATCHES = {"forehead": (0.25, 0.05, 0.50, 0.20), "left cheek": (0.15, 0.45, 0.30, 0.20), "right cheek": (0.55, 0.45, 0.30, 0.20)}
PATCH_COL = {"forehead": (255, 80, 0), "left cheek": (0, 220, 220), "right cheek": (0, 220, 220)}      # BGR as the app draws them


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _bandpass(x, fps, lo=0.7, hi=3.0):
    from scipy import signal as sp
    sos = sp.butter(3, [lo, hi], btype="band", fs=fps, output="sos")
    return sp.sosfiltfilt(sos, x)


def _pulse_snr_db(x, fps):
    """Strongest peak in 42-180 BPM vs the median of the rest of that band (dB), on the band-passed signal."""
    from scipy import signal as sp
    f, p = sp.welch(_bandpass(x, fps), fs=fps, window="hann", nperseg=len(x), nfft=4096)
    m = (f >= 0.7) & (f <= 3.0)
    f, p = f[m], p[m]
    k = int(np.argmax(p))
    rest = p[np.abs(f - f[k]) > 0.15]
    return float(10 * np.log10(p[k] / (np.median(rest) + 1e-18)))


# ───────────────────────────── extraction ─────────────────────────────────────────────────────────────────────────────
def extract_video(video: str, d1: Path) -> dict:
    """Decode a video with the app's own Camera (face + ROI) and keep one CLEAN frame per second."""
    from core.camera import Camera
    cam = Camera(source=video)
    if cam.dummy_mode:
        raise RuntimeError(f"cannot open video: {video}")
    cam.record_roi_parts = True
    fps = cam.get_source_fps(30.0)
    (d1 / "frames").mkdir(parents=True, exist_ok=True)
    t, rgb, motion, parts, frames = [], [], [], [], []
    i = 0
    while i < 100000:
        fb, roi, _moving, mot = cam.get_frame(annotate=False)          # clean frame; ROI numbers identical to annotated mode
        if fb is None:
            break
        if roi is not None:
            p = (cam.last_roi_parts or {}).get("prod")
            t.append(i / fps); rgb.append(roi); motion.append(float(mot))
            parts.append([p[k][2 - c] for k in range(3) for c in range(3)] if p else [np.nan] * 9)    # (B,G,R) -> R,G,B per patch
            if i % int(round(fps)) == 0 and cam.last_face_box:
                fn = d1 / "frames" / f"frame_{i:05d}_t{i / fps:06.1f}s.jpg"
                fn.write_bytes(fb)
                frames.append({"t": i / fps, "file": fn, "box": cam.last_face_box[:4]})
        i += 1
    return {"t": np.array(t), "rgb": np.array(rgb, float), "motion": np.array(motion), "parts": np.array(parts, float),
            "frames": frames, "fps": fps, "mode": "upload", "n_frames_total": i}


def load_trace(npz: str, d1: Path) -> dict:
    """A live session recorded by core/poc_jobs.LiveTraceRecorder."""
    z = np.load(npz, allow_pickle=True)
    (d1 / "frames").mkdir(parents=True, exist_ok=True)
    frames = []
    src_dir = Path(npz).parent / "_live_frames"
    for tt, name, box in zip(z["frame_t"], z["frame_files"], z["frame_boxes"]):
        src = src_dir / str(name)
        if src.exists():
            dst = d1 / "frames" / src.name
            os.replace(src, dst)
            frames.append({"t": float(tt), "file": dst, "box": tuple(float(v) for v in box)})
    try:
        src_dir.rmdir()                                  # empty now: the frames were moved into 01_frames_and_rgb/frames
    except OSError:
        pass
    t = np.array(z["t"], float)
    return {"t": t - t[0], "rgb": np.array(z["rgb"], float), "motion": np.array(z["motion"], float), "parts": np.array(z["parts"], float),
            "frames": frames, "fps": float(len(t) / max(1e-9, t[-1] - t[0])), "mode": "live", "n_frames_total": int(len(t))}


def _uniform(tr: dict):
    """The stream the engines see. Upload: as is. Live: resampled onto a uniform 30 Hz grid exactly like the app does."""
    if tr["mode"] != "live":
        return tr["t"], tr["rgb"], tr["motion"], tr["parts"], float(tr["fps"])
    t = tr["t"]
    g = np.arange(0.0, t[-1], 1.0 / LIVE_FS)
    rgb = np.stack([np.interp(g, t, tr["rgb"][:, k]) for k in range(3)], 1)
    mot = np.interp(g, t, tr["motion"])
    parts = np.stack([np.interp(g, t, tr["parts"][:, k]) for k in range(9)], 1)
    return g, rgb, mot, parts, LIVE_FS


# ───────────────────────────── stage 1 ───────────────────────────────────────────────────────────────────────────────
def _draw_patches(img, box):
    import cv2
    sx, sy, sw, sh = (int(round(v)) for v in box)
    out = img.copy()
    cv2.rectangle(out, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
    for name, (ox, oy, ow, oh) in PATCHES.items():
        x1, y1 = sx + int(sw * ox), sy + int(sh * oy)
        cv2.rectangle(out, (x1, y1), (x1 + int(sw * ow), y1 + int(sh * oh)), PATCH_COL[name], 2)
    return out


def _cs_weights(rgb, fps):
    """The colour weights CHROM and POS actually apply (averaged over 10 s windows), and each colour's share of the result."""
    n = len(rgb)
    win = int(fps * 10)
    wc, wp, sc, sp_ = [], [], [], []
    for s in range(0, max(1, n - win + 1), max(1, int(fps * 5))):
        w = rgb[s:s + win]
        if len(w) < win // 2:
            continue
        nrm = (w - w.mean(0)) / (w.mean(0) + 1e-9)
        R, G, B = nrm[:, 0], nrm[:, 1], nrm[:, 2]
        X, Y = 3 * R - 2 * G, 1.5 * R + G - 1.5 * B
        a = (X.std() + 1e-9) / (Y.std() + 1e-9)
        wc.append([3 - 1.5 * a, -2 - a, 1.5 * a])                       # S = X - a*Y written per colour
        Xs, Ys = G - B, -2 * R + G + B
        b = (Xs.std() + 1e-9) / (Ys.std() + 1e-9)
        wp.append([-2 * b, 1 + b, -1 + b])                              # S = Xs + b*Ys written per colour
        sc.append(nrm.std(0)); sp_.append(nrm.std(0))
    wc, wp, sd = np.mean(wc, 0), np.mean(wp, 0), np.mean(sc, 0)
    share = lambda w: 100 * np.abs(w * sd) / np.sum(np.abs(w * sd))
    return wc, wp, share(wc), share(wp)


def stage1(tr, U, d1: Path, notes: dict) -> dict:
    import cv2
    plt = _plt()
    t, rgb, motion, parts, fps = U
    # contact sheet with the 3 patches boxed
    frames = tr["frames"]
    if frames:
        pick = [frames[int(k)] for k in np.linspace(0, len(frames) - 1, min(12, len(frames)))]
        fig, axs = plt.subplots(3, 4, figsize=(14, 8))
        for ax in axs.ravel():
            ax.axis("off")
        for ax, fr in zip(axs.ravel(), pick):
            img = cv2.imread(str(fr["file"]))
            ax.imshow(cv2.cvtColor(_draw_patches(img, fr["box"]), cv2.COLOR_BGR2RGB)); ax.set_title(f"t = {fr['t']:.0f} s", fontsize=9)
        fig.suptitle(f"Stage 1a - video converted to frames ({tr['fps']:.1f} frames/s measured, {tr['n_frames_total']} frames, face found in {len(tr['t'])}); "
                     "boxes = the 3 skin patches that are averaged (forehead, left cheek, right cheek)", fontsize=10)
        fig.tight_layout(); fig.savefig(d1 / "contact_sheet_with_patches.png", dpi=105); plt.close(fig)
        # red / green / blue channel frames
        (d1 / "channel_frames").mkdir(exist_ok=True)
        pick6 = [frames[int(k)] for k in np.linspace(0, len(frames) - 1, min(6, len(frames)))]
        fig, axs = plt.subplots(4, len(pick6), figsize=(3.3 * len(pick6), 10.5), squeeze=False)
        for j, fr in enumerate(pick6):
            img = cv2.imread(str(fr["file"]))
            variants = [("original", img)]
            for c, idx in (("R", 2), ("G", 1), ("B", 0)):
                v = np.zeros_like(img); v[:, :, idx] = img[:, :, idx]
                cv2.imwrite(str(d1 / "channel_frames" / f"t{fr['t']:06.1f}s_{c}.jpg"), v)
                variants.append((c + " channel only", v))
            for r_, (name, v) in enumerate(variants):
                axs[r_][j].imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB)); axs[r_][j].axis("off")
                axs[r_][j].set_title(f"{name}  t={fr['t']:.0f}s" if r_ == 0 or j == 0 else "", fontsize=8)
        fig.suptitle("Stage 1b - the same frames seen through only the red, only the green and only the blue channel", fontsize=11)
        fig.tight_layout(); fig.savefig(d1 / "channel_frames_sheet.png", dpi=100); plt.close(fig)

    # csv + trace plot
    with open(d1 / "rgb_per_frame.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "R", "G", "B", "motion", "forehead_R", "forehead_G", "forehead_B", "leftcheek_R", "leftcheek_G", "leftcheek_B",
                    "rightcheek_R", "rightcheek_G", "rightcheek_B"])
        w.writerows(np.round(np.column_stack([t, rgb, motion, parts]), 4).tolist())
    ac = np.stack([_bandpass(rgb[:, k] - rgb[:, k].mean(), fps) for k in range(3)], 1)
    fig, axs = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for k, c in enumerate(CH):
        axs[0].plot(t, rgb[:, k], color=COL[c], label=f"{c} (mean {rgb[:, k].mean():.1f})")
        axs[1].plot(t, 100 * ac[:, k] / rgb[:, k].mean(), color=COL[c], lw=1, label=c)
    axs[0].set_ylabel("average colour value (0-255)"); axs[0].legend(); axs[0].set_title("Stage 1c - average R, G, B of the face patches, one value per frame")
    axs[1].set_ylabel("pulsating part, % of its own level"); axs[1].set_xlabel("time (s)"); axs[1].legend(loc="upper right")
    axs[1].set_title("same signals after removing the slow level (band-pass 0.7-3 Hz = 42-180 BPM): this tiny wobble is the pulse")
    fig.tight_layout(); fig.savefig(d1 / "rgb_trace.png", dpi=105); plt.close(fig)

    # which colour contributes more
    stats = {}
    for k, c in enumerate(CH):
        snr = _pulse_snr_db(rgb[:, k] - rgb[:, k].mean(), fps)
        stats[c] = dict(dc=float(rgb[:, k].mean()), ac=float(ac[:, k].std()), acdc=float(100 * ac[:, k].std() / rgb[:, k].mean()), snr=snr)
    patch_stats = {}
    for pi, name in enumerate(PATCHES):
        patch_stats[name] = {}
        for k, c in enumerate(CH):
            x = parts[:, pi * 3 + k]
            if np.all(np.isfinite(x)):
                patch_stats[name][c] = dict(acdc=float(100 * _bandpass(x - x.mean(), fps).std() / x.mean()), snr=_pulse_snr_db(x - x.mean(), fps))
    wc, wp, shc, shp = _cs_weights(rgb, fps)
    cc = np.corrcoef(ac.T)[np.triu_indices(3, 1)]

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    def bars(ax, vals, title, ylabel, fmt):
        b = ax.bar(CH, vals, color=[COL[c] for c in CH]); ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel)
        for r_, v in zip(b, vals):
            ax.annotate(fmt.format(v), (r_.get_x() + r_.get_width() / 2, r_.get_height()), ha="center", va="bottom", fontsize=10)
    bars(axs[0][0], [stats[c]["acdc"] for c in CH], "(a) size of the pulse relative to the colour level", "AC/DC, %", "{:.3f}")
    bars(axs[0][1], [stats[c]["snr"] for c in CH], "(b) how clearly the pulse stands out of the noise", "pulse signal-to-noise, dB", "{:.1f}")
    ax = axs[1][0]; wdt = 0.25
    for pi, name in enumerate(PATCHES):
        vals = [patch_stats[name].get(c, {}).get("snr", np.nan) for c in CH]
        xs = np.arange(3) + (pi - 1) * wdt
        bb = ax.bar(xs, vals, wdt, label=name)
        for r_, v in zip(bb, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.1f}", (r_.get_x() + r_.get_width() / 2, r_.get_height()), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(CH); ax.legend(fontsize=8, loc="lower center", ncol=3); ax.set_ylim(0, ax.get_ylim()[1] * 1.12); ax.set_title("(c) same signal-to-noise, split by skin patch", fontsize=10); ax.set_ylabel("dB")
    ax = axs[1][1]
    for off, (nm, w, sh) in enumerate((("CHROM", wc, shc), ("POS", wp, shp))):
        xs = np.arange(3) + (off - 0.5) * 0.35
        bb = ax.bar(xs, w, 0.35, label=nm, color=["#2ca02c", "#9467bd"][off])
        for r_, v, s_ in zip(bb, w, sh):
            ax.annotate(f"{v:+.2f}\n({s_:.0f}%)", (r_.get_x() + r_.get_width() / 2, v), ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels(CH); ax.legend(fontsize=8, loc="lower right")
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_ * 1.2, hi_ * 1.25)
    ax.set_title("(d) weight each colour gets inside CHROM and POS (and its % share of the result)", fontsize=10)
    fig.suptitle("Stage 1d - which colour contributes more to the pulse, and by how much", fontsize=12)
    fig.tight_layout(); fig.savefig(d1 / "channel_contribution.png", dpi=105); plt.close(fig)

    best_snr = max(stats, key=lambda c: stats[c]["snr"]); best_ac = max(stats, key=lambda c: stats[c]["acdc"])
    spread = max(s["snr"] for s in stats.values()) - min(s["snr"] for s in stats.values())
    lines = ["STAGE 1 - which colour contributes more to the pulse, and why", "",
             f"recording: {notes['source']} | {tr['fps']:.1f} frames/s | {tr['n_frames_total']} frames | face found in {len(tr['t'])} | engine rate {fps:.0f} Hz", "",
             "MEASURED ON THIS RECORDING (all three skin patches pooled; forehead 50%, each cheek 25%)",
             f"{'colour':7s} {'level (DC)':>11s} {'pulse size (AC)':>16s} {'AC/DC %':>9s} {'pulse SNR dB':>13s}"]
    for c in CH:
        lines.append(f"{c:7s} {stats[c]['dc']:11.2f} {stats[c]['ac']:16.4f} {stats[c]['acdc']:9.3f} {stats[c]['snr']:13.1f}")
    lines += ["", f"=> Clearest pulse (highest signal-to-noise): {best_snr}.   Largest relative pulse (AC/DC): {best_ac}.",
              "", "Per skin patch (pulse SNR in dB / AC/DC %):"]
    for name in PATCHES:
        lines.append(f"  {name:12s} " + "   ".join(f"{c}: {patch_stats[name][c]['snr']:.1f} dB / {patch_stats[name][c]['acdc']:.3f}%" for c in CH if c in patch_stats[name]))
    lines += ["", "HOW CHROM AND POS USE THE COLOURS (average weights over this recording; % = share of the result)",
              f"  CHROM : R {wc[0]:+.2f} ({shc[0]:.0f}%)   G {wc[1]:+.2f} ({shc[1]:.0f}%)   B {wc[2]:+.2f} ({shc[2]:.0f}%)",
              f"  POS   : R {wp[0]:+.2f} ({shp[0]:.0f}%)   G {wp[1]:+.2f} ({shp[1]:.0f}%)   B {wp[2]:+.2f} ({shp[2]:.0f}%)",
              "  A negative weight does not mean 'weak colour': the formulas SUBTRACT colours from each other so that brightness changes shared",
              "  by all three cancel (lighting, small movements) and only the part that differs between colours - the pulse - is kept.", "",
              "WHY A COLOUR CAN CONTRIBUTE MORE (physics - the textbook expectation)",
              "  1. Blood absorbs light unevenly. Oxygenated haemoglobin absorbs strongly around 540-580 nm (green/yellow) and weakly in red (>620 nm).",
              "     Every heartbeat briefly raises the blood volume under the skin, so green light is absorbed a bit more at each beat: the green",
              "     channel dips and rises with the pulse more than red does.",
              "  2. Depth. Blue (~450 nm) barely penetrates skin and is absorbed by melanin in the surface layer, so it sees the surface, not the blood.",
              "     Red (~650 nm) reaches deepest but blood absorbs it little, so each beat changes it only slightly.",
              "  3. Sensor. Most webcams use a Bayer sensor with twice as many green pixels as red or blue, so green is also the least noisy channel.",
              "  Expected order: green > red > blue.", "", "WHAT THIS RECORDING SHOWS"]
    if spread < 1.0:
        lines.append(f"  The three colours pulse almost equally (signal-to-noise spread only {spread:.1f} dB) - no clear winner, so the textbook order is neither confirmed nor refuted.")
    elif best_snr == "G":
        lines.append(f"  Green stands out most ({stats['G']['snr']:.1f} dB vs R {stats['R']['snr']:.1f}, B {stats['B']['snr']:.1f}) - consistent with the textbook expectation.")
    else:
        lines.append(f"  {best_snr} stands out most ({stats[best_snr]['snr']:.1f} dB), not green ({stats['G']['snr']:.1f} dB): this recording does NOT follow the textbook order.")
        lines.append("  Possible reasons (not tested here): webcam auto white-balance / auto exposure moving the channels, or heavy lighting change.")
    lines += [f"  The pulsating parts of the three colours are {100 * cc.min():.0f}-{100 * cc.max():.0f}% correlated: most of what each colour wobbles by is shared by",
              "  all three (the pulse itself plus any brightness change from movement or lighting - this table cannot separate the two), which is exactly",
              "  why CHROM/POS combine the colours instead of reading a single one."]
    (d1 / "channel_contribution.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"stats": stats, "patch_stats": patch_stats, "chrom_weights": wc.tolist(), "pos_weights": wp.tolist(), "channel_correlation": [float(cc.min()), float(cc.max())]}


# ───────────────────────────── stages 2-4 ────────────────────────────────────────────────────────────────────────────
def stages234(tr, U, dirs, notes: dict, truth) -> dict:
    from scipy import signal as sp
    from core.rppg import AdvancedRPPG, _project_chrom, _project_pos
    from core import hr_ensemble as E
    plt = _plt()
    d2, d3, d4 = dirs
    t, rgb, motion, parts, fps = U
    n_valid = len(t)
    win = max(300, int(fps * 10))
    eng = {"CHROM": AdvancedRPPG(fps=int(round(fps)), window_size=win), "POS": AdvancedRPPG(fps=int(round(fps)), window_size=win, algorithm="POS")}
    for e in eng.values():
        e.capture_spectra = True
    models = E.load_models()
    ses = E.HREnsembleSession(models, int(round(fps)), win, threaded=False) if models else None
    snaps_k = sorted({int(0.55 * n_valid), int(0.78 * n_valid), n_valid - 1})
    snap = {c: {} for c in eng}
    cyc_time = []
    for k in range(n_valid):
        res = {}
        for c, e in eng.items():
            e.add_frame(tuple(rgb[k]), motion_delta=float(motion[k]))
            res[c] = e.process_ppg_signal()
            if k in snaps_k and res[c]["ready"]:
                snap[c][k] = dict(ppg=np.array(res[c]["ppg_signal"], float), spec=np.array(res[c]["spectrum_wide"], float), bpm=res[c]["bpm"])
        if ses is not None:
            before = len(ses.h["bpm"])
            ses.feed(tuple(rgb[k]), float(motion[k]), res["CHROM"])
            if len(ses.h["bpm"]) > before:
                cyc_time.append(t[k])
    final = {c: e.get_final_summary().get("final_bpm") for c, e in eng.items()}
    pulse_windows, spectrum_windows = {}, {}

    ks = [k for k in snaps_k if all(k in snap[c] for c in eng)]
    for c in eng:                                                                      # ---- stage 2
        if not ks:
            break
        fig, axs = plt.subplots(len(ks), 1, figsize=(13, 3.2 * len(ks)))
        axs = np.atleast_1d(axs)
        for ax, k in zip(axs, ks):
            s = snap[c][k]; n = len(s["ppg"])
            r_, g_, b_ = (rgb[k - n + 1:k + 1, j] for j in range(3))
            raw = _project_chrom(r_, g_, b_) if c == "CHROM" else _project_pos(r_, g_, b_, fps)
            tt = np.arange(n) / fps
            ax.plot(tt, (raw - np.mean(raw)) / (np.std(raw) + 1e-12), color="#bbbbbb", lw=1, label=f"{c} raw projection")
            y = (s["ppg"] - np.mean(s["ppg"])) / (np.std(s["ppg"]) + 1e-12)
            ax.plot(tt, y, color="#2ca02c", lw=2, label="after band-pass (what the engine uses)")
            pk, _ = sp.find_peaks(y, distance=int(0.4 * fps), prominence=0.3)
            ax.plot(tt[pk], y[pk], "rv", label=f"{len(pk)} heartbeats")
            hb = 60.0 / np.mean(np.diff(pk) / fps) if len(pk) > 2 else float("nan")
            pulse_windows.setdefault(k, {"t": float(t[k])})[c] = {"beats_bpm": None if not np.isfinite(hb) else float(hb), "engine_bpm": float(s["bpm"])}
            ax.set_title(f"window ending at t = {t[k]:.1f} s | beats counted: {hb:.0f} BPM | engine reading: {s['bpm']:.0f} BPM", fontsize=10)
            ax.set_xlabel("seconds within the 10 s window"); ax.legend(loc="upper right", fontsize=8)
        fig.suptitle(f"Stage 2 - {c} pulse-oriented signal (each peak = one heartbeat)"); fig.tight_layout()
        fig.savefig(d2 / f"pulse_{c.lower()}.png", dpi=105); plt.close(fig)
    if ks:
        k = ks[-1]
        m = min(len(snap["CHROM"][k]["ppg"]), len(snap["POS"][k]["ppg"]))
        fig, ax = plt.subplots(figsize=(13, 4))
        for c, colr in (("CHROM", "#2ca02c"), ("POS", "#9467bd")):
            y = snap[c][k]["ppg"][-m:]; ax.plot(np.arange(m) / fps, (y - y.mean()) / (y.std() + 1e-12), color=colr, lw=2, label=c)
        ax.set_title(f"Stage 2 - CHROM vs POS on the same last 10 s (t = {t[k]:.1f} s): two different formulas, same heartbeat rhythm")
        ax.set_xlabel("seconds"); ax.legend(); fig.tight_layout(); fig.savefig(d2 / "chrom_vs_pos.png", dpi=105); plt.close(fig)
        np.savetxt(d2 / "pulse_last10s.csv", np.column_stack([np.arange(m) / fps, snap["CHROM"][k]["ppg"][-m:], snap["POS"][k]["ppg"][-m:]]),
                   delimiter=",", header="time_s,chrom_pulse,pos_pulse", comments="")

    peaks_txt = ["STAGE 3 - Welch frequency spectrum of the band-passed pulse (116 bins, 42-180 BPM)", ""]   # ---- stage 3
    if ks:
        fig, axs = plt.subplots(len(ks), 1, figsize=(13, 3.2 * len(ks)))
        axs = np.atleast_1d(axs)
        for ax, k in zip(axs, ks):
            for c, colr in (("CHROM", "#2ca02c"), ("POS", "#9467bd")):
                sp_ = snap[c][k]["spec"]; ax.plot(E.GRID_BPM, sp_ / sp_.max(), color=colr, lw=2, label=c)
                pk, _ = sp.find_peaks(np.convolve(sp_, [0.25, 0.5, 0.25], "same"), prominence=sp_.max() * 0.05)
                top = pk[np.argsort(-sp_[pk])][:3]
                spectrum_windows.setdefault(k, {"t": float(t[k])})[c] = [[float(E.GRID_BPM[j]), float(sp_[j] / sp_.max())] for j in top]
                peaks_txt.append(f"window ending t={t[k]:.1f}s  {c:5s} top peaks (BPM, relative power): " + ", ".join(f"{E.GRID_BPM[j]:.0f} ({sp_[j] / sp_.max():.2f})" for j in top))
            if truth:
                ax.axvline(truth, color="red", ls="--", label=f"reference {truth:.1f} BPM")
            ax.set_title(f"window ending at t = {t[k]:.1f} s"); ax.set_xlabel("beats per minute (= frequency in Hz x 60)"); ax.set_ylabel("relative power"); ax.legend(loc="upper right", fontsize=8)
        fig.suptitle("Stage 3 - Welch spectrum: the tallest peak is the heart rate"); fig.tight_layout(); fig.savefig(d3 / "welch_spectrum.png", dpi=105); plt.close(fig)
    peaks_txt += ["", "Reading it: the strongest peak is the pulse rate. A smaller peak at about half or double that rate is a harmonic - the classic cause of",
                  "wrong readings, and the reason stage 4 chooses between peaks instead of taking the tallest."]
    (d3 / "peaks.txt").write_text("\n".join(peaks_txt) + "\n", encoding="utf-8")

    sel = trk = ens = None                                                              # ---- stage 4
    if ses is not None and len(ses.h["bpm"]) > E.TRK_START + 10:
        n_cyc = len(ses.h["bpm"])
        tt_ = sorted({max(0, (int(f * n_cyc) // 5) * 5) for f in (0.5, 0.75, 0.97)})
        fig, axs = plt.subplots(1, len(tt_), figsize=(5.5 * len(tt_), 4), sharey=True)
        axs = np.atleast_1d(axs)
        for ax, tc in zip(axs, tt_):
            rows_, bp = E.candidate_rows(tc, ses.h)
            p = models.selector.predict_proba(rows_)
            ax.bar(np.round(bp).astype(int).astype(str), p, color=["#2ca02c" if j == int(np.argmax(p)) else "#bbbbbb" for j in range(len(p))])
            ax.set_title(f"cycle {tc} (t = {cyc_time[min(tc, len(cyc_time) - 1)]:.0f} s)\nchosen: {bp[int(np.argmax(p))]:.0f} BPM", fontsize=10); ax.set_xlabel("candidate peak (BPM)")
        axs[0].set_ylabel("probability this peak is the true HR")
        fig.suptitle("Stage 4a - peak selector: candidate peaks from CHROM + POS, scored by 200 small decision trees (29 features each)")
        fig.tight_layout(); fig.savefig(d4 / "selector_candidates.png", dpi=105); plt.close(fig)
        dec, post = [], []
        for tc in range(E.TRK_START, n_cyc, E.TRK_STRIDE):
            idx = E.seq_indices(tc, tc + 1)
            x = np.empty((1, E.TRK_STEPS, 2, E.NB), np.float32)
            x[0, :, 0] = np.stack([ses.h["chrom_spec"][j] for j in idx]); x[0, :, 1] = np.stack([ses.h["pos_spec"][j] for j in idx])
            x /= (x.max(3, keepdims=True) + 1e-9)
            post.append(models.tracker_posterior(x)[0]); dec.append(cyc_time[min(tc, len(cyc_time) - 1)])
        post = np.array(post)
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.imshow(post.T, aspect="auto", origin="lower", cmap="magma", extent=[dec[0], dec[-1], E.GRID_BPM[0], E.GRID_BPM[-1]])
        ax.plot(dec, E.GRID_BPM[post.argmax(1)], "c.", ms=4, label="tracker's pick")
        if truth:
            ax.axhline(truth, color="red", ls="--", label=f"reference {truth:.1f}")
        ax.set_xlabel("time (s)"); ax.set_ylabel("BPM"); ax.legend(loc="upper right")
        ax.set_title("Stage 4b - spectro-temporal tracker: probability of each heart rate over time (bright = confident); it favours peaks that persist")
        fig.tight_layout(); fig.savefig(d4 / "tracker_posterior.png", dpi=105); plt.close(fig)
        sel, trk = E.iqr_median(ses.sel_hist), E.iqr_median(ses.trk_hist)
        ens = float(np.mean([sel, trk]))
        fig, ax = plt.subplots(figsize=(13, 4.5))
        ts = [cyc_time[min(j * 5, len(cyc_time) - 1)] for j in range(len(ses.sel_hist))]
        tk = [cyc_time[min(E.TRK_START + j * E.TRK_STRIDE, len(cyc_time) - 1)] for j in range(len(ses.trk_hist))]
        ax.plot(ts, ses.sel_hist, ".", color="#1f77b4", ms=3, label=f"selector picks (median {sel:.1f})")
        ax.plot(tk, ses.trk_hist, ".", color="#ff7f0e", ms=4, label=f"tracker picks (median {trk:.1f})")
        ax.axhline(ens, color="#2ca02c", lw=2, label=f"ENSEMBLE = mean of the two = {ens:.1f}")
        if truth:
            ax.axhline(truth, color="red", ls="--", label=f"reference {truth:.1f}")
        ax.set_xlabel("time (s)"); ax.set_ylabel("BPM"); ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Stage 4c - every decision of both models; the final answer is the median of each, then their mean")
        fig.tight_layout(); fig.savefig(d4 / "decisions_timeline.png", dpi=105); plt.close(fig)

    f1 = lambda v: "n/a" if v is None else f"{v:.1f} BPM"
    lines = ["STAGE 4 - final heart rate and where it came from", "", f"recording: {notes['source']}  |  patient: {notes['name']}", "",
             f"classical CHROM only (no models)     : {f1(final['CHROM'])}", f"classical POS only (no models)       : {f1(final['POS'])}"]
    if ens is not None:
        lines += [f"peak selector (median of its picks)   : {sel:.1f} BPM", f"spectro-temporal tracker (median)     : {trk:.1f} BPM",
                  f"ENSEMBLE = mean(selector, tracker)    : {ens:.1f} BPM   <-- the number the app shows"]
    else:
        lines.append("ensemble: not available (too short a recording, or the model files were not found)")
    if truth:
        lines.append(f"reference (finger oximeter)           : {truth:.1f} BPM")
        for nm, v in (("classical CHROM", final["CHROM"]), ("ensemble", ens)):
            if v is not None:
                lines.append(f"error of {nm:16s}: {abs(v - truth):.1f} BPM")
    lines += ["", "HONEST NOTES",
              "- The ensemble models were trained on recorded sessions from public datasets (UBFC-rPPG, UBFC-Phys). If this recording is one of them its",
              "  result is optimistic; for a new person or camera the realistic error is the out-of-fold figure, about 3.9 BPM (125 recorded sessions, 75 people).",
              "- Without a reference device there is no 'error' to report for this recording - only the numbers above and how well the stages agree.",
              "- Heart rate only. Evidence is from recorded video; live-webcam accuracy is not validated (see data/evolution/23-*.md)."]
    (d4 / "ensemble_result.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"classical_chrom": final["CHROM"], "classical_pos": final["POS"], "selector": sel, "tracker": trk, "ensemble": ens,
            "pulse_windows": [pulse_windows[k] for k in sorted(pulse_windows)], "spectrum_windows": [spectrum_windows[k] for k in sorted(spectrum_windows)],
            "n_selector_decisions": len(ses.sel_hist) if ses is not None else 0, "n_tracker_decisions": len(ses.trk_hist) if ses is not None else 0}


# ───────────────────────────── driver ────────────────────────────────────────────────────────────────────────────────
def _page(out: Path) -> None:
    try:
        from core import poc_site
        poc_site.build_run_page(out)
    except Exception as exc:                                           # noqa: BLE001
        print("report page failed:", exc)


def build(out: Path, name: str, source: str, video: str | None, trace: str | None, truth: float | None) -> dict:
    t0 = time.time()
    d1, d2, d3, d4 = (out / n for n in ("01_frames_and_rgb", "02_pulse_signals_chrom_pos", "03_welch_spectrum", "04_ensemble"))
    for d in (d1, d2, d3, d4):
        d.mkdir(parents=True, exist_ok=True)
    tr = extract_video(video, d1) if video else load_trace(trace, d1)
    notes = {"name": name, "source": source + (f" ({Path(video).name})" if video else " camera")}
    result = {"patient": name, "source": source, "reference_bpm": truth, "frames_total": tr["n_frames_total"], "frames_with_face": int(len(tr["t"])), "fps_measured": round(tr["fps"], 2)}
    if len(tr["t"]) < 300:
        msg = f"TOO SHORT: only {len(tr['t'])} frames with a face ({len(tr['t']) / max(tr['fps'], 1):.0f} s). At least ~10 s of face video is needed for one reading."
        (out / "summary.txt").write_text(msg + "\n", encoding="utf-8")
        result["status"] = "too_short"
        (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        _page(out)
        return result
    U = _uniform(tr)
    result.update(stage1(tr, U, d1, notes))
    result.update(stages234(tr, U, (d2, d3, d4), notes, truth))
    result.update({"status": "done", "seconds_to_build": round(time.time() - t0, 1), "duration_s": round(float(U[0][-1]), 1)})
    (out / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    sm = ["PROOF-OF-CONCEPT SUMMARY", f"patient: {name} | source: {notes['source']} | duration {result['duration_s']:.0f} s | camera/video rate {tr['fps']:.1f} frames/s", ""]
    for k in ("classical_chrom", "classical_pos", "selector", "tracker", "ensemble"):
        if result.get(k) is not None:
            sm.append(f"{k:16s}: {result[k]:.1f} BPM")
    if truth:
        sm.append(f"{'reference':16s}: {truth:.1f} BPM")
    sm += ["", "Folders: 01 frames + colours, 02 CHROM/POS pulse, 03 Welch spectrum, 04 ensemble. Heart rate only."]
    (out / "summary.txt").write_text("\n".join(sm) + "\n", encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--trace"); ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="Unnamed"); ap.add_argument("--source", default="upload")
    ap.add_argument("--truth", type=float, default=None)
    a = ap.parse_args()
    if not (a.video or a.trace):
        sys.exit("give --video or --trace")
    res = build(Path(a.out), a.name, a.source, a.video, a.trace, a.truth)
    print(json.dumps({k: res.get(k) for k in ("status", "ensemble", "classical_chrom", "seconds_to_build")}, default=float))


if __name__ == "__main__":
    main()
