#!/usr/bin/env python3
"""
Proof of concept: show, stage by stage, HOW the heart-rate number comes out of a video.

    ./venv/Scripts/python.exe poc/run_poc.py "C:/Users/M.Manoj/Downloads/test.avi" --truth 98.3

Uses the app's own classes (core.camera.Camera, core.rppg.AdvancedRPPG, core.hr_ensemble) - nothing is re-implemented, so
the pictures are the real pipeline.  Outputs, one folder per stage (next to this file):

  01_frames_and_rgb/          video -> frames -> average R,G,B per frame, and which colour channel / skin patch carries the pulse
  02_pulse_signals_chrom_pos/ the CHROM and POS pulse waveforms (raw projection, then after the band-pass filter)
  03_welch_spectrum/          Welch frequency spectrum of the band-passed pulse (where the heart rate is read from)
  04_ensemble/                what the two ensemble models see and choose, and the final answer
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
from core.camera import Camera                                              # noqa: E402
from core.rppg import AdvancedRPPG, _project_chrom, _project_pos            # noqa: E402
from core import hr_ensemble as E                                           # noqa: E402

OUT = Path(__file__).resolve().parent
D1, D2, D3, D4 = (OUT / n for n in ("01_frames_and_rgb", "02_pulse_signals_chrom_pos", "03_welch_spectrum", "04_ensemble"))
CH = ["R", "G", "B"]
COL = {"R": "#d62728", "G": "#2ca02c", "B": "#1f77b4"}


def bandpass(x, fps, lo=0.7, hi=3.0):
    sos = sp.butter(3, [lo, hi], btype="band", fs=fps, output="sos")
    return sp.sosfiltfilt(sos, x)


def pulse_snr_db(x, fps):
    """Strongest peak in 42-180 BPM vs the median of the rest of that band (dB), on the band-passed channel."""
    f, p = sp.welch(bandpass(x, fps), fs=fps, window="hann", nperseg=len(x), nfft=4096)
    m = (f >= 0.7) & (f <= 3.0)
    f, p = f[m], p[m]
    k = int(np.argmax(p))
    rest = p[np.abs(f - f[k]) > 0.15]
    return 10 * np.log10(p[k] / (np.median(rest) + 1e-18)), float(f[k] * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--truth", type=float, default=None, help="reference heart rate (BPM) from the oximeter, if known")
    a = ap.parse_args()
    for d in (D1, D2, D3, D4):
        (d / "frames").mkdir(parents=True, exist_ok=True) if d is D1 else d.mkdir(parents=True, exist_ok=True)

    # ============================== STAGE 1: video -> frames -> average R,G,B ==============================
    cam = Camera(source=a.video)
    if cam.dummy_mode:
        sys.exit(f"cannot open video: {a.video}")
    cam.record_roi_parts = True
    fps = cam.get_source_fps(30.0)
    rows, saved, i = [], [], 0
    while i < 100000:
        fb, roi, moving, motion = cam.get_frame(annotate=True)
        if fb is None:
            break
        parts = (cam.last_roi_parts or {}).get("prod") if roi is not None else None
        r, g, b = (roi if roi is not None else (np.nan,) * 3)
        row = [i, i / fps, int(roi is not None), r, g, b, float(motion)]
        for k in range(3):                                   # forehead, left cheek, right cheek: parts are (B, G, R)
            row += [parts[k][2], parts[k][1], parts[k][0]] if parts else [np.nan] * 3
        rows.append(row)
        if roi is not None and i % int(round(fps)) == 0:
            fn = D1 / "frames" / f"frame_{i:04d}_t{i / fps:05.1f}s.jpg"
            fn.write_bytes(fb)
            saved.append(fn)
        i += 1
    A = np.array(rows, float)
    valid = A[:, 2] == 1
    t, rgb, motion_arr = A[valid, 1], A[valid, 3:6], A[valid, 6]
    n_valid = int(valid.sum())
    with open(D1 / "rgb_per_frame.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "face_found", "R", "G", "B", "motion", "forehead_R", "forehead_G", "forehead_B",
                    "leftcheek_R", "leftcheek_G", "leftcheek_B", "rightcheek_R", "rightcheek_G", "rightcheek_B"])
        w.writerows(np.round(A, 4).tolist())

    # contact sheet (12 frames, evenly spaced)
    pick = [saved[int(k)] for k in np.linspace(0, len(saved) - 1, 12)]
    fig, axs = plt.subplots(3, 4, figsize=(14, 8))
    for ax, fn in zip(axs.ravel(), pick):
        ax.imshow(plt.imread(fn)); ax.set_title(fn.stem.split("_t")[1].replace("s", " s"), fontsize=9); ax.axis("off")
    fig.suptitle(f"Stage 1a - video converted to frames ({fps:.2f} fps, {len(rows)} frames, face found in {n_valid}); "
                 f"boxes = the 3 skin patches averaged (forehead, left cheek, right cheek)")
    fig.tight_layout(); fig.savefig(D1 / "contact_sheet.png", dpi=110); plt.close(fig)

    # RGB traces: raw mean and pulsating part
    ac = np.stack([bandpass(rgb[:, k] - rgb[:, k].mean(), fps) for k in range(3)], 1)
    fig, axs = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for k, c in enumerate(CH):
        axs[0].plot(t, rgb[:, k], color=COL[c], label=f"{c} (mean {rgb[:, k].mean():.1f})")
        axs[1].plot(t, 100 * ac[:, k] / rgb[:, k].mean(), color=COL[c], lw=1, label=c)
    axs[0].set_ylabel("average colour value (0-255)"); axs[0].legend(); axs[0].set_title("Stage 1b - average R, G, B of the face patches, one value per frame")
    axs[1].set_ylabel("pulsating part, % of its own level"); axs[1].set_xlabel("time (s)"); axs[1].legend(loc="upper right")
    axs[1].set_title("same signals after removing the slow level (band-pass 0.7-3 Hz = 42-180 BPM): this tiny wobble IS the pulse")
    fig.tight_layout(); fig.savefig(D1 / "rgb_trace.png", dpi=110); plt.close(fig)

    # which channel / patch contributes more
    lines = ["STAGE 1 - which colour carries the pulse?", "",
             f"video: {a.video}", f"frame rate {fps:.2f} fps | {len(rows)} frames | face found in {n_valid} ({100 * n_valid / len(rows):.0f}%)", "",
             "Per colour channel (all three skin patches pooled; forehead 50%, each cheek 25%):",
             f"{'channel':8s} {'level (DC)':>11s} {'pulse size (AC)':>16s} {'AC/DC %':>9s} {'pulse SNR dB':>13s}"]
    stats = {}
    for k, c in enumerate(CH):
        snr, _ = pulse_snr_db(rgb[:, k] - rgb[:, k].mean(), fps)
        stats[c] = (rgb[:, k].mean(), ac[:, k].std(), 100 * ac[:, k].std() / rgb[:, k].mean(), snr)
        lines.append(f"{c:8s} {stats[c][0]:11.2f} {stats[c][1]:16.4f} {stats[c][2]:9.3f} {stats[c][3]:13.1f}")
    best = max(stats, key=lambda c: stats[c][3])
    cc = np.corrcoef(ac.T)[np.triu_indices(3, 1)]
    lo_, hi_ = round(100 * cc.min()), round(100 * cc.max())
    corr_txt = f"about {hi_}%" if lo_ == hi_ else f"{lo_}-{hi_}%"
    spread = max(s[3] for s in stats.values()) - min(s[3] for s in stats.values())
    lines += ["", f"=> Best signal-to-noise: {best}.  Largest relative pulse size (AC/DC): {max(stats, key=lambda c: stats[c][2])}.",
              f"   In THIS clip the three channels pulse almost equally (SNR spread only {spread:.1f} dB) and their pulsating parts are",
              f"   {corr_txt} correlated: nearly all of the wobble is COMMON to R, G and B (the pulse itself plus any brightness change",
              "   from motion or lighting; this table cannot separate the two). CHROM and POS (stage 2) combine the channels to cancel",
              "   the lighting/motion part and keep the pulse. (Literature: green usually carries the strongest pulse because haemoglobin absorbs green",
              "   light; this clip shows no clear winner, so no channel is singled out here.)", "",
              "Per skin patch (green channel). AC/DC includes noise, so compare it with the SNR column:",
              f"{'patch':12s} {'AC/DC %':>9s} {'pulse SNR dB':>13s}"]
    for name, off in (("forehead", 7), ("left cheek", 10), ("right cheek", 13)):
        gch = A[valid, off + 1]
        if np.all(np.isfinite(gch)):
            acg = bandpass(gch - gch.mean(), fps)
            lines.append(f"{name:12s} {100 * acg.std() / gch.mean():9.3f} {pulse_snr_db(gch - gch.mean(), fps)[0]:13.1f}")
    (D1 / "channel_contribution.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ====================== run the REAL engines (CHROM and POS) + the ensemble on the same frames ======================
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
            e.add_frame(tuple(rgb[k]), motion_delta=float(motion_arr[k]))
            res[c] = e.process_ppg_signal()
            if k in snaps_k and res[c]["ready"]:
                snap[c][k] = dict(ppg=np.array(res[c]["ppg_signal"], float), spec=np.array(res[c]["spectrum_wide"], float), bpm=res[c]["bpm"])
        if ses is not None:
            before = len(ses.h["bpm"])
            ses.feed(tuple(rgb[k]), float(motion_arr[k]), res["CHROM"])
            if len(ses.h["bpm"]) > before:
                cyc_time.append(t[k])
    final = {c: e.get_final_summary().get("final_bpm") for c, e in eng.items()}

    # ============================== STAGE 2: pulse waveforms ==============================
    ks = [k for k in snaps_k if all(k in snap[c] for c in eng)]
    for c in eng:
        fig, axs = plt.subplots(len(ks), 1, figsize=(13, 3.2 * len(ks)))
        axs = np.atleast_1d(axs)
        for ax, k in zip(axs, ks):
            s = snap[c][k]
            n = len(s["ppg"])
            r_, g_, b_ = (rgb[k - n + 1:k + 1, j] for j in range(3))
            raw = _project_chrom(r_, g_, b_) if c == "CHROM" else _project_pos(r_, g_, b_, fps)
            tt = np.arange(n) / fps
            ax.plot(tt, (raw - np.mean(raw)) / (np.std(raw) + 1e-12), color="#bbbbbb", lw=1, label=f"{c} raw projection")
            y = (s["ppg"] - np.mean(s["ppg"])) / (np.std(s["ppg"]) + 1e-12)
            ax.plot(tt, y, color="#2ca02c", lw=2, label="after band-pass (what the engine uses)")
            pk, _ = sp.find_peaks(y, distance=int(0.4 * fps), prominence=0.3)
            ax.plot(tt[pk], y[pk], "rv", label=f"{len(pk)} heartbeats")
            hb = 60.0 / np.mean(np.diff(pk) / fps) if len(pk) > 2 else float("nan")
            ax.set_title(f"window ending at t = {t[k]:.1f} s | beats counted: {hb:.0f} BPM | engine reading: {s['bpm']:.0f} BPM", fontsize=10)
            ax.set_xlabel("seconds within the 10 s window"); ax.legend(loc="upper right", fontsize=8)
        fig.suptitle(f"Stage 2 - {c} pulse-oriented signal (each peak = one heartbeat)"); fig.tight_layout()
        fig.savefig(D2 / f"pulse_{c.lower()}.png", dpi=110); plt.close(fig)
    if ks:
        k = ks[-1]
        fig, ax = plt.subplots(figsize=(13, 4))
        for c, colr in (("CHROM", "#2ca02c"), ("POS", "#9467bd")):
            y = snap[c][k]["ppg"]; ax.plot(np.arange(len(y)) / fps, (y - y.mean()) / y.std(), color=colr, lw=2, label=c)
        ax.set_title(f"Stage 2 - CHROM vs POS on the same last 10 s (t = {t[k]:.1f} s): two different formulas, same heartbeat rhythm")
        ax.set_xlabel("seconds"); ax.legend(); fig.tight_layout(); fig.savefig(D2 / "chrom_vs_pos.png", dpi=110); plt.close(fig)
        np.savetxt(D2 / "pulse_last10s.csv", np.column_stack([np.arange(len(snap["CHROM"][k]["ppg"])) / fps, snap["CHROM"][k]["ppg"],
                   snap["POS"][k]["ppg"][-len(snap["CHROM"][k]["ppg"]):] if len(snap["POS"][k]["ppg"]) >= len(snap["CHROM"][k]["ppg"]) else np.resize(snap["POS"][k]["ppg"], len(snap["CHROM"][k]["ppg"]))]),
                   delimiter=",", header="time_s,chrom_pulse,pos_pulse", comments="")

    # ============================== STAGE 3: Welch spectrum ==============================
    peaks_txt = ["STAGE 3 - Welch frequency spectrum of the band-passed pulse (116 bins, 42-180 BPM)", ""]
    fig, axs = plt.subplots(len(ks), 1, figsize=(13, 3.2 * len(ks)))
    axs = np.atleast_1d(axs)
    for ax, k in zip(axs, ks):
        for c, colr in (("CHROM", "#2ca02c"), ("POS", "#9467bd")):
            sp_ = snap[c][k]["spec"]; ax.plot(E.GRID_BPM, sp_ / sp_.max(), color=colr, lw=2, label=c)
            pk, _ = sp.find_peaks(np.convolve(sp_, [0.25, 0.5, 0.25], "same"), prominence=sp_.max() * 0.05)
            top = pk[np.argsort(-sp_[pk])][:3]
            peaks_txt.append(f"window ending t={t[k]:.1f}s  {c:5s} top peaks (BPM, relative power): " +
                             ", ".join(f"{E.GRID_BPM[j]:.0f} ({sp_[j] / sp_.max():.2f})" for j in top))
        if a.truth:
            ax.axvline(a.truth, color="red", ls="--", label=f"reference {a.truth:.1f} BPM")
        ax.set_title(f"window ending at t = {t[k]:.1f} s"); ax.set_xlabel("beats per minute (= frequency in Hz x 60)"); ax.set_ylabel("relative power")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Stage 3 - Welch spectrum: the tallest peak is the heart rate"); fig.tight_layout()
    fig.savefig(D3 / "welch_spectrum.png", dpi=110); plt.close(fig)
    peaks_txt += ["", "Reading it: the strongest peak is the pulse rate. A smaller peak at about half or double that rate is a harmonic - the",
                  "classic cause of wrong readings, and the reason stage 4 chooses between peaks instead of just taking the tallest."]
    (D3 / "peaks.txt").write_text("\n".join(peaks_txt) + "\n", encoding="utf-8")

    # ============================== STAGE 4: ensemble ==============================
    sel, trk, ens = None, None, None
    if ses is not None and len(ses.h["bpm"]) > 160:
        n_cyc = len(ses.h["bpm"])
        # (a) selector: candidate peaks and the probability it gives each
        tt_ = sorted({max(0, (int(f * n_cyc) // 5) * 5) for f in (0.5, 0.75, 0.97)})
        fig, axs = plt.subplots(1, len(tt_), figsize=(5.5 * len(tt_), 4), sharey=True)
        axs = np.atleast_1d(axs)
        for ax, tc in zip(axs, tt_):
            rows_, bp = E.candidate_rows(tc, ses.h)
            p = models.selector.predict_proba(rows_)
            ax.bar(np.round(bp).astype(int).astype(str), p, color=["#2ca02c" if j == int(np.argmax(p)) else "#bbbbbb" for j in range(len(p))])
            ax.set_title(f"cycle {tc} (t = {cyc_time[min(tc, len(cyc_time) - 1)]:.0f} s)\nchosen: {bp[int(np.argmax(p))]:.0f} BPM", fontsize=10)
            ax.set_xlabel("candidate peak (BPM)")
        axs[0].set_ylabel("selector's probability this peak is the true HR")
        fig.suptitle("Stage 4a - peak selector: candidate peaks from CHROM + POS spectra, scored by 200 small decision trees (29 features each)")
        fig.tight_layout(); fig.savefig(D4 / "selector_candidates.png", dpi=110); plt.close(fig)
        # (b) tracker posterior over time
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
        if a.truth:
            ax.axhline(a.truth, color="red", ls="--", label=f"reference {a.truth:.1f}")
        ax.set_xlabel("time (s)"); ax.set_ylabel("BPM"); ax.legend(loc="upper right")
        ax.set_title("Stage 4b - spectro-temporal tracker: probability of each heart rate over time (bright = confident); it favours peaks that PERSIST")
        fig.tight_layout(); fig.savefig(D4 / "tracker_posterior.png", dpi=110); plt.close(fig)
        # (c) timeline of both models' decisions
        sel, trk = E.iqr_median(ses.sel_hist), E.iqr_median(ses.trk_hist)
        ens = float(np.mean([sel, trk]))
        fig, ax = plt.subplots(figsize=(13, 4.5))
        ts = [cyc_time[min(j * 5, len(cyc_time) - 1)] for j in range(len(ses.sel_hist))]
        tk = [cyc_time[min(E.TRK_START + j * E.TRK_STRIDE, len(cyc_time) - 1)] for j in range(len(ses.trk_hist))]
        ax.plot(ts, ses.sel_hist, ".", color="#1f77b4", ms=3, label=f"selector picks (median {sel:.1f})")
        ax.plot(tk, ses.trk_hist, ".", color="#ff7f0e", ms=4, label=f"tracker picks (median {trk:.1f})")
        ax.axhline(ens, color="#2ca02c", lw=2, label=f"ENSEMBLE = mean of the two = {ens:.1f}")
        if a.truth:
            ax.axhline(a.truth, color="red", ls="--", label=f"reference {a.truth:.1f}")
        ax.set_xlabel("time (s)"); ax.set_ylabel("BPM"); ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Stage 4c - every decision of both models; the final answer is the median of each, then their mean")
        fig.tight_layout(); fig.savefig(D4 / "decisions_timeline.png", dpi=110); plt.close(fig)

    res_lines = ["STAGE 4 - final heart rate and where it came from", "", f"video: {a.video}", ""]
    res_lines.append(f"classical CHROM only (no models)     : {final['CHROM']:.1f} BPM" if final["CHROM"] is not None else "classical CHROM only: n/a")
    res_lines.append(f"classical POS only (no models)       : {final['POS']:.1f} BPM" if final["POS"] is not None else "classical POS only: n/a")
    if ens is not None:
        res_lines += [f"peak selector (median of its picks)   : {sel:.1f} BPM", f"spectro-temporal tracker (median)     : {trk:.1f} BPM",
                      f"ENSEMBLE = mean(selector, tracker)    : {ens:.1f} BPM   <-- the number the app shows"]
    if a.truth:
        res_lines.append(f"reference (finger oximeter)           : {a.truth:.1f} BPM")
        for name, v in (("classical CHROM", final["CHROM"]), ("ensemble", ens)):
            if v is not None:
                res_lines.append(f"error of {name:16s}: {abs(v - a.truth):.1f} BPM")
    res_lines += ["", "HONEST NOTES",
                  "- The ensemble models were trained on ALL 125 recorded sessions, which includes this video's session, so this single",
                  "  result is optimistic ('in-sample'). The unbiased accuracy is the out-of-fold figure: about 3.9 BPM mean error on 125",
                  "  recorded sessions / 75 people (classical 11.35, old hybrid 6.69 on the same sessions).",
                  "- That is evidence on recorded video, not on a live webcam. See data/evolution/22-*.md and 23-*.md."]
    (D4 / "ensemble_result.txt").write_text("\n".join(res_lines) + "\n", encoding="utf-8")
    print("\n".join(res_lines))
    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
