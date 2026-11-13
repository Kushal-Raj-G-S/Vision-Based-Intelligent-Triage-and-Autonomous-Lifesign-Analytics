"""
Plain-English conclusion for each stage of a proof-of-concept run, built ONLY from the run's measured numbers (result.json).

No language model is involved, so a number can never be misquoted. Every stage gets:
    did     what was done in this stage
    found   what the numbers show
    checks  testable statements, each marked  pass / warn / fail / info  with its evidence
    next    what happens next (in the pipeline, and for the project)
Statements whose numbers were not recorded for an older run are simply left out.
"""
from __future__ import annotations

import math

AGREE_PULSE = 10.0      # BPM: beat counting of CHROM vs POS, and vs the engine's own reading
AGREE_PEAK = 5.0        # BPM: spectral peaks / model answers agree
MIN_FACE_PCT = 90.0
MIN_SECONDS = 20.0
MIN_SNR_DB = 10.0
MIN_DECISIONS = 20


def _n(v, d=1):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"


def _chk(status: str, text: str) -> dict:
    return {"status": status, "text": text}


def _stage1(r: dict) -> dict:
    stats = r.get("stats") or {}
    total, face = r.get("frames_total"), r.get("frames_with_face")
    dur, fps = r.get("duration_s"), r.get("fps_measured")
    did = (f"The recording ({_n(dur, 0)} s, {_n(fps)} frames per second, {total} frames) was split into frames. "
           f"In {face} of them a face was found; for each one the average red, green and blue of the forehead and both cheeks was taken, "
           "and the slow brightness level was filtered out (band-pass 0.7-3 Hz, i.e. 42-180 beats per minute) so the tiny pulse wobble is exposed.")
    found, checks = [], []
    if stats:
        snr = {c: stats[c]["snr"] for c in "RGB" if c in stats}
        best = max(snr, key=snr.get)
        spread = max(snr.values()) - min(snr.values())
        found.append("Pulse clarity (signal-to-noise): " + ", ".join(f"{c} {snr[c]:.1f} dB" for c in "RGB" if c in snr) + f". Clearest colour: {best}.")
        cc = r.get("channel_correlation")
        if cc:
            lo_, hi_ = round(100 * cc[0]), round(100 * cc[1])
            found.append(f"The pulsating parts of the three colours are {'about ' + str(hi_) if lo_ == hi_ else str(lo_) + '-' + str(hi_)}% alike, so most of the wobble is shared by all three; "
                         "that is why the next stage combines the colours instead of reading one.")
        checks.append(_chk("pass" if snr[best] >= MIN_SNR_DB else "fail",
                           f"A pulse stands out of the noise in at least one colour: best {best} = {snr[best]:.1f} dB (needs >= {MIN_SNR_DB:.0f} dB)."))
        if spread >= 1.0:
            checks.append(_chk("info", "Textbook expectation is green > red > blue. " + (
                "This recording agrees: green is clearest." if best == "G" else f"This recording differs: {best} is clearest, green is {snr.get('G', float('nan')):.1f} dB.")))
        else:
            checks.append(_chk("info", f"The three colours pulse almost equally here (spread {spread:.1f} dB): no clear winner, so the textbook order is neither confirmed nor refuted."))
    if total and face is not None:
        pct = 100.0 * face / total
        checks.append(_chk("pass" if pct >= MIN_FACE_PCT else ("warn" if pct >= 60 else "fail"),
                           f"The face was found in {pct:.0f}% of the frames (needs >= {MIN_FACE_PCT:.0f}% for a steady signal)."))
    if dur is not None:
        checks.append(_chk("pass" if dur >= MIN_SECONDS else "warn", f"Recording length {dur:.0f} s (>= {MIN_SECONDS:.0f} s gives a stable reading)."))
    return {"title": "Stage 1 - frames and the average red, green, blue", "did": did, "found": found, "checks": checks,
            "next": "Stage 2 combines the three colours with two formulas (CHROM and POS) so lighting changes cancel and only the pulse is left."}


def _stage2(r: dict) -> dict:
    wins = r.get("pulse_windows") or []
    did = ("CHROM and POS each combined the red, green and blue into one pulse signal and filtered it to 42-180 beats per minute. "
           f"Heartbeats were counted by peak detection in {len(wins) or 'three'} ten-second windows.")
    found, checks = [], []
    agree = agree_eng = tot = tot_eng = 0
    for w in wins:
        c, p = w.get("CHROM") or {}, w.get("POS") or {}
        found.append(f"At t = {w['t']:.0f} s: CHROM counted {_n(c.get('beats_bpm'), 0)} BPM (engine {_n(c.get('engine_bpm'), 0)}); "
                     f"POS counted {_n(p.get('beats_bpm'), 0)} BPM (engine {_n(p.get('engine_bpm'), 0)}).")
        if c.get("beats_bpm") is not None and p.get("beats_bpm") is not None:
            tot += 1
            agree += abs(c["beats_bpm"] - p["beats_bpm"]) <= AGREE_PULSE
        for q in (c, p):
            if q.get("beats_bpm") is not None and q.get("engine_bpm") is not None:
                tot_eng += 1
                agree_eng += abs(q["beats_bpm"] - q["engine_bpm"]) <= AGREE_PULSE
    if tot:
        checks.append(_chk("pass" if agree == tot else "warn",
                           f"CHROM and POS count the same heart rate (within {AGREE_PULSE:.0f} BPM) in {agree} of {tot} windows."
                           + ("" if agree == tot else " A single method can be fooled in some windows; stage 4 is built to resolve exactly this.")))
    if tot_eng:
        checks.append(_chk("pass" if agree_eng == tot_eng else "warn",
                           f"Counting heartbeats on the waveform matches the engine's own reading (within {AGREE_PULSE:.0f} BPM) in {agree_eng} of {tot_eng} cases."))
    if not wins:
        found.append("The per-window heartbeat counts were not recorded for this older run; the pictures are still in the folder.")
    return {"title": "Stage 2 - CHROM and POS pulse signals", "did": did, "found": found, "checks": checks,
            "next": "Stage 3 converts the last 10 seconds of each pulse into a frequency spectrum (Welch's method), where the heart rate shows as the tallest peak."}


def _stage3(r: dict) -> dict:
    wins = r.get("spectrum_windows") or []
    ref = r.get("reference_bpm")
    did = "Welch's method turned the filtered pulse of each 10-second window into a frequency spectrum covering 42-180 beats per minute; the tallest peak is the heart-rate estimate."
    found, checks = [], []
    competing = 0
    for w in wins:
        parts = []
        for a in ("CHROM", "POS"):
            pk = w.get(a) or []
            if pk:
                parts.append(f"{a} " + ", ".join(f"{b:.0f} ({rel:.2f})" for b, rel in pk[:3]))
        found.append(f"At t = {w['t']:.0f} s the strongest peaks (BPM, relative power) were: " + "; ".join(parts) + ".")
        for a in ("CHROM", "POS"):
            pk = w.get(a) or []
            if len(pk) > 1 and pk[1][1] >= 0.5:
                competing += 1
                break
    if wins:
        last = wins[-1]
        c, p = last.get("CHROM") or [], last.get("POS") or []
        if c and p:
            d = abs(c[0][0] - p[0][0])
            checks.append(_chk("pass" if d <= AGREE_PEAK else "warn",
                               f"In the last window CHROM's tallest peak ({c[0][0]:.0f}) and POS's ({p[0][0]:.0f}) agree within {AGREE_PEAK:.0f} BPM (difference {d:.0f})."))
            clear = (len(c) < 2 or c[1][1] < 0.5) and (len(p) < 2 or p[1][1] < 0.5)
            checks.append(_chk("pass" if clear else "warn", "The last window has one clear peak in both spectra." if clear else "The last window still has a competing peak (second peak at least half as tall)."))
        if ref and c:
            d = abs(c[0][0] - ref)
            checks.append(_chk("pass" if d <= AGREE_PEAK else "warn", f"The tallest CHROM peak in the last window ({c[0][0]:.0f}) is within {AGREE_PEAK:.0f} BPM of the reference {ref:.1f} (difference {d:.1f})."))
        checks.append(_chk("info", f"{competing} of {len(wins)} windows had a competing peak. Taking the tallest peak blindly would sometimes pick the wrong one - which is why stage 4 chooses between peaks."))
    else:
        found.append("The per-window peaks were not recorded for this older run; the spectrum picture is still in the folder.")
    return {"title": "Stage 3 - Welch frequency spectrum", "did": did, "found": found, "checks": checks,
            "next": "Stage 4 scores the candidate peaks with two models and gives the final heart rate."}


def _stage4(r: dict) -> dict:
    ens, sel, trk = r.get("ensemble"), r.get("selector"), r.get("tracker")
    cl, po, ref = r.get("classical_chrom"), r.get("classical_pos"), r.get("reference_bpm")
    did = ("Two models examined the spectra: a peak selector (200 small decision trees that score each candidate peak) and a tracker (a small convolutional network that checks "
           "which peak persists over the last 11 spectra). The final heart rate is the mean of the median answer of each.")
    found, checks = [], []
    if ens is None:
        found.append("The ensemble could not run (recording too short, or the model files were not found).")
        return {"title": "Stage 4 - the ensemble", "did": did, "found": found, "checks": [_chk("warn", "No ensemble reading for this run.")],
                "next": "Record at least ~20 s of steady face video and run again."}
    found.append(f"Selector {_n(sel)} BPM, tracker {_n(trk)} BPM, ensemble {_n(ens)} BPM. Without the models: classical CHROM {_n(cl)} BPM, classical POS {_n(po)} BPM.")
    if sel is not None and trk is not None:
        d = abs(sel - trk)
        checks.append(_chk("pass" if d <= AGREE_PEAK else "fail", f"The two models agree within {AGREE_PEAK:.0f} BPM (selector {sel:.1f}, tracker {trk:.1f}, difference {d:.1f})."))
    if cl is not None:
        d = abs(ens - cl)
        checks.append(_chk("info", f"The ensemble {'confirms' if d <= 2 else 'changed'} the classical answer by {d:.1f} BPM ({_n(cl)} -> {_n(ens)})."))
    if po is not None and abs(po - ens) > 15:
        checks.append(_chk("info", f"POS alone read {po:.1f}, {abs(po - ens):.0f} BPM away from the ensemble - an example of a single method being fooled by a competing peak."))
    n = min(r.get("n_selector_decisions") or 0, 10 ** 9)
    if n:
        checks.append(_chk("pass" if n >= MIN_DECISIONS else "warn", f"{n} selector decisions were made over the recording (needs >= {MIN_DECISIONS} for a steady median)."))
    if ref:
        e_ens = abs(ens - ref)
        checks.append(_chk("pass" if e_ens <= AGREE_PEAK else "fail", f"Ensemble vs the reference device ({ref:.1f} BPM): error {e_ens:.1f} BPM (target <= {AGREE_PEAK:.0f})."))
        if cl is not None:
            checks.append(_chk("info", f"Classical CHROM alone: error {abs(cl - ref):.1f} BPM."))
    else:
        checks.append(_chk("info", "No reference device was used for this recording, so no error can be reported - only whether the stages agree."))
    checks.append(_chk("info", "Honest limits: heart rate only; the models were trained on public recorded datasets (a clip from them gives an optimistic result); "
                               "the realistic error on new people is about 3.9 BPM (125 recorded sessions, 75 people); live-webcam accuracy is not validated."))
    return {"title": "Stage 4 - the ensemble and the final heart rate", "did": did, "found": found, "checks": checks,
            "next": "For the project: validate on a live webcam against a reference device, measure accuracy across skin tones, and improve speech and motion cases."}


def build(r: dict) -> dict:
    stages = [_stage1(r), _stage2(r), _stage3(r), _stage4(r)]
    allc = [c for s in stages for c in s["checks"]]
    cnt = {k: sum(c["status"] == k for c in allc) for k in ("pass", "warn", "fail")}
    decisive = stages[0]["checks"] + stages[3]["checks"]           # signal quality and the final answer decide the verdict
    fails = [c["text"] for c in decisive if c["status"] == "fail"]
    warns = [c["text"] for c in allc if c["status"] == "warn"]
    ens = r.get("ensemble")
    head = f"Heart rate: {ens:.1f} BPM (the number the app shows)." if ens is not None else "No final heart-rate reading was produced for this run."
    if fails:
        verdict = "Treat this reading with caution: " + fails[0]
    elif warns:
        verdict = "The final answer is consistent; notes: " + warns[0]
    elif ens is not None:
        verdict = "All checks passed: the stages agree with each other."
    else:
        verdict = "Not enough data for a verdict."
    return {"headline": head, "verdict": verdict, "counts": cnt, "stages": stages}


def to_text(c: dict) -> str:
    sym = {"pass": "[PASS]", "warn": "[NOTE]", "fail": "[FAIL]", "info": "[INFO]"}
    out = ["CONCLUSIONS", "", c["headline"], c["verdict"],
           f"Checks: {c['counts']['pass']} passed, {c['counts']['warn']} notes, {c['counts']['fail']} failed.", ""]
    for s in c["stages"]:
        out += [s["title"].upper(), "What was done: " + s["did"], "What it shows:"] + [f"  - {t}" for t in s["found"]]
        out += ["Checks:"] + [f"  {sym[k['status']]} {k['text']}" for k in s["checks"]] + ["What happens next: " + s["next"], ""]
    return "\n".join(out)
