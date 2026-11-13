"""
Static HTML pages for the proof-of-concept folders (stdlib only, light enough for the app to import).

    poc/patients/index.html                    every patient and run: status, headline numbers, thumbnail, link
    poc/patients/<name>/<run>/report.html      the walk-through of one run: numbers, then each stage with its pictures and a plain caption

Plain files with relative links and no fetch(), so they open from a double-click (file://) as well as from any web server.
Rebuild everything:  python -m core.poc_site
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
POC_ROOT = ROOT / "poc" / "patients"

CSS = """
:root{--bg:#fffaf4;--card:#fff;--ink:#1f2937;--mute:#6b7280;--line:#eadfd2;--orange:#f26b1d;--orange2:#fde7d6;--green:#16a34a;--red:#dc2626;--blue:#2563eb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "Segoe UI",system-ui,-apple-system,Inter,Arial,sans-serif}
a{color:var(--orange);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px}
header.top{background:#fff;border-bottom:1px solid var(--line)}header.top .wrap{display:flex;align-items:center;gap:14px;padding:16px 20px;flex-wrap:wrap}
.logo{width:34px;height:34px;border-radius:9px;background:var(--orange);color:#fff;display:grid;place-items:center;font-weight:800}
h1{font-size:22px;margin:0}h2{font-size:20px;margin:0 0 4px}h3{font-size:15px;margin:18px 0 6px}
.sub{color:var(--mute);font-size:13px}.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;border:1px solid var(--line);background:#fff}
.pill.done{background:#dcfce7;border-color:#bbf7d0;color:#166534}.pill.running,.pill.queued,.pill.recording{background:#fef3c7;border-color:#fde68a;color:#92400e}
.pill.failed{background:#fee2e2;border-color:#fecaca;color:#991b1b}.pill.live{background:#dbeafe;border-color:#bfdbfe;color:#1e40af}.pill.upload{background:#ede9fe;border-color:#ddd6fe;color:#5b21b6}
.grid{display:grid;gap:16px}.g4{grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}.g2{grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 1px 2px rgba(0,0,0,.03)}
.stat{text-align:center}.stat .v{font-size:34px;font-weight:800;line-height:1.1}.stat .k{color:var(--mute);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.stat.main{background:var(--orange);border-color:var(--orange);color:#fff}.stat.main .k{color:#ffe9da}
.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:14px 0}.flow a{background:#fff;border:1px solid var(--line);border-radius:12px;padding:8px 12px;font-size:13px;color:var(--ink)}
.flow a:hover{border-color:var(--orange);text-decoration:none}.flow i{color:var(--mute);font-style:normal}
nav.stages{position:sticky;top:0;z-index:5;background:rgba(255,250,244,.92);backdrop-filter:blur(6px);border-bottom:1px solid var(--line)}nav.stages .wrap{display:flex;gap:6px;padding:8px 20px;flex-wrap:wrap}
nav.stages a{padding:5px 12px;border-radius:999px;font-size:13px;color:var(--ink);border:1px solid transparent}nav.stages a:hover{background:var(--orange2);text-decoration:none}
section.stage{margin:34px 0}section.stage>h2 .n{display:inline-grid;place-items:center;width:30px;height:30px;border-radius:50%;background:var(--orange);color:#fff;font-size:15px;margin-right:8px}
figure{margin:14px 0;background:#fff;border:1px solid var(--line);border-radius:14px;padding:12px}figure img{width:100%;height:auto;border-radius:8px;cursor:zoom-in;display:block}
figcaption{color:var(--mute);font-size:13px;margin-top:8px}figcaption b{color:var(--ink)}
pre.note{white-space:pre-wrap;word-wrap:break-word;background:#fff;border:1px solid var(--line);border-left:4px solid var(--orange);border-radius:10px;padding:12px 14px;font:13px/1.55 Consolas,"Cascadia Mono",monospace;margin:12px 0}
.callout{background:var(--orange2);border:1px solid #f9c9a4;border-radius:12px;padding:12px 14px;font-size:14px}
.ai-polish{margin-top:18px;background:#f5f8ff;border:1px solid #dbe6fe;border-left:5px solid var(--blue);border-radius:14px;padding:14px 18px}
.ai-polish .tag{display:inline-block;background:var(--blue);color:#fff;font-size:11px;font-weight:700;letter-spacing:.03em;text-transform:uppercase;padding:2px 9px;border-radius:999px;margin-bottom:8px}
.ai-polish p{margin:6px 0}.ai-polish .src{color:var(--mute);font-size:12px;margin-top:8px}
.concl{margin-top:18px;background:#fff;border:1px solid var(--line);border-left:5px solid var(--orange);border-radius:14px;padding:14px 18px}
.concl h3{margin:0 0 8px;font-size:16px}.concl p{margin:6px 0}.concl ul{margin:6px 0 6px 0;padding-left:20px}
ul.checks{list-style:none;padding-left:0!important}ul.checks li{padding:5px 8px 5px 30px;border-radius:8px;margin:4px 0;position:relative;font-size:14px}
ul.checks li:before{position:absolute;left:8px;top:5px;font-weight:800}
ul.checks li.pass{background:#f0fdf4}ul.checks li.pass:before{content:"\\2713";color:var(--green)}ul.checks li.warn{background:#fffbeb}ul.checks li.warn:before{content:"!";color:#d97706}
ul.checks li.fail{background:#fef2f2}ul.checks li.fail:before{content:"\\2717";color:var(--red)}ul.checks li.info{background:#f8fafc}ul.checks li.info:before{content:"i";color:var(--blue);font-style:italic}
.short{border-left:5px solid var(--green)}.short.warn{border-left-color:#d97706}.short.fail{border-left-color:var(--red)}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:50;cursor:zoom-out}#lb img{max-width:96vw;max-height:94vh;border-radius:6px}
.run{display:flex;gap:14px;align-items:center;padding:10px 0;border-top:1px solid var(--line)}.run:first-of-type{border-top:0}.run img{width:96px;height:72px;object-fit:cover;border-radius:8px;background:#eee;flex:none}
.run .meta{flex:1;min-width:0}.run .nums{text-align:right;white-space:nowrap}.run .nums b{font-size:20px}
.btn{display:inline-block;background:var(--orange);color:#fff;border-radius:10px;padding:7px 14px;font-weight:600;font-size:13px}.btn:hover{text-decoration:none;filter:brightness(.95)}
input.search{width:100%;max-width:360px;padding:9px 12px;border-radius:10px;border:1px solid var(--line);font:inherit}
footer{color:var(--mute);font-size:12px;padding:30px 0 40px}
@media print{nav.stages,#lb,.btn{display:none}body{background:#fff}.card,figure{break-inside:avoid}}
"""

LB_JS = """<div id="lb"><img alt=""></div><script>
var lb=document.getElementById('lb'),li=lb.querySelector('img');
document.querySelectorAll('figure img').forEach(function(i){i.addEventListener('click',function(){li.src=i.src;lb.style.display='flex'})});
lb.addEventListener('click',function(){lb.style.display='none'});document.addEventListener('keydown',function(e){if(e.key==='Escape')lb.style.display='none'});
</script>"""


def _read(p: Path, default=""):
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return default


def _json(p: Path) -> dict:
    try:
        return json.loads(_read(p, "{}"))
    except Exception:
        return {}


def _u(rel: str) -> str:
    return "/".join(quote(part) for part in rel.replace("\\", "/").split("/"))


def _fmt(v, unit=""):
    return "&ndash;" if v is None else f"{v:.1f}{unit}"


def _figure(run: Path, rel: str, caption: str) -> str:
    if not (run / rel).exists():
        return ""
    return f'<figure><img src="{_u(rel)}" alt="{html.escape(rel)}" loading="lazy"><figcaption>{caption}</figcaption></figure>'


def _note(run: Path, rel: str) -> str:
    t = _read(run / rel).strip()
    return f'<pre class="note">{html.escape(t)}</pre>' if t else ""


def _thumb(run: Path) -> str | None:
    fr = sorted((run / "01_frames_and_rgb" / "frames").glob("*.jpg")) if (run / "01_frames_and_rgb" / "frames").exists() else []
    return f"{run.parent.name}/{run.name}/01_frames_and_rgb/frames/{fr[len(fr) // 2].name}" if fr else None


def _concl_html(st: dict) -> str:
    found = "".join(f"<li>{html.escape(t)}</li>" for t in st["found"])
    checks = "".join(f'<li class="{c["status"]}">{html.escape(c["text"])}</li>' for c in st["checks"])
    return (f'<div class="concl"><h3>Conclusion of this stage</h3><p><b>What was done.</b> {html.escape(st["did"])}</p>'
            f'{"<p><b>What it shows.</b></p><ul>" + found + "</ul>" if found else ""}'
            f'{"<p><b>Checks (each one can be verified against the numbers above):</b></p><ul class=checks>" + checks + "</ul>" if checks else ""}'
            f'<p><b>What happens next.</b> {html.escape(st["next"])}</p></div>')


def _polish_html(polish: Optional[dict], key: str) -> str:
    """One AI-polished paragraph, if the run has one for this key and it passed the fact-check guardrail.
    Absent entirely (not an error message) whenever polishing was off, failed, or didn't pass — the plain
    deterministic text above/below it already stands on its own either way."""
    item = (polish or {}).get(key)
    if not item or not item.get("text"):
        return ""
    return (f'<div class="ai-polish"><span class="tag">AI-polished, fact-checked</span>'
            f'<p>{html.escape(item["text"])}</p>'
            f'<p class="src">Written by {html.escape(item.get("generated_by") or "?")}; every number independently checked against the measured values above '
            f'by {html.escape(item.get("checked_by") or "?")} before this paragraph was allowed on the page.</p></div>')


def _short_html(con: dict) -> str:
    c = con["counts"]
    cls = "fail" if c["fail"] else ("warn" if c["warn"] else "")
    return (f'<div class="concl short {cls}"><h3>In short</h3><p><b>{html.escape(con["headline"])}</b></p><p>{html.escape(con["verdict"])}</p>'
            f'<p class="sub">Checks: {c["pass"]} passed &middot; {c["warn"]} notes &middot; {c["fail"]} failed. These statements are generated from the measured numbers, not by a language model, '
            f'so they cannot misquote a value. Full text: <a href="conclusions.txt">conclusions.txt</a></p></div>')


# ───────────────────────────── one run ───────────────────────────────────────────────────────────────────────────────
def build_run_page(run: Path) -> None:
    st, rs = _json(run / "status.json"), _json(run / "result.json")
    name = rs.get("patient") or st.get("patient") or run.parent.name
    source = rs.get("source") or st.get("source") or ""
    state = st.get("state", "done" if rs else "queued")
    when = run.name[:15]
    when = f"{when[:4]}-{when[4:6]}-{when[6:8]} {when[9:11]}:{when[11:13]}:{when[13:15]}" if len(when) == 15 else run.name
    ens = rs.get("ensemble")
    cards = [("Ensemble (app's number)", ens, True), ("Classical CHROM", rs.get("classical_chrom"), False), ("Classical POS", rs.get("classical_pos"), False),
             ("Peak selector", rs.get("selector"), False), ("Spectro-temporal tracker", rs.get("tracker"), False)]
    if rs.get("reference_bpm"):
        cards.append(("Reference (oximeter)", rs["reference_bpm"], False))
    stats = "".join(f'<div class="card stat{" main" if main else ""}"><div class="v">{_fmt(v)}</div><div class="k">{html.escape(k)} &middot; BPM</div></div>'
                    for k, v, main in cards if v is not None or main)
    con = None
    if rs and rs.get("status") == "done":
        try:
            from core import poc_conclusions
            con = poc_conclusions.build(rs)
            (run / "conclusions.txt").write_text(poc_conclusions.to_text(con), encoding="utf-8")
        except Exception:
            con = None
    polish = _json(run / "llm_polish.json") or None
    body = []
    if state in ("queued", "running", "recording"):
        body.append(f'<div class="callout">This report is still being built (<b>{html.escape(state)}</b>). Reload in a minute.</div>')
    elif state == "failed":
        body.append(f'<div class="callout"><b>The build failed.</b> Last lines of build.log:<pre class="note">{html.escape(_read(run / "build.log")[-900:])}</pre></div>')
    elif rs.get("status") == "too_short":
        body.append(f'<div class="callout">{html.escape(_read(run / "summary.txt").strip())}</div>')
    stage1 = (_figure(run, "01_frames_and_rgb/contact_sheet_with_patches.png",
                      "<b>Video &rarr; frames.</b> The recording is split into frames. The green box is the detected face; the three small boxes are the skin patches "
                      "(forehead, left cheek, right cheek) whose colours are averaged in every frame.")
              + _figure(run, "01_frames_and_rgb/channel_frames_sheet.png",
                        "<b>Red, green and blue frames.</b> The same frames seen through only one colour channel. The pulse cannot be seen by eye: each heartbeat changes the brightness by about half a percent.")
              + _figure(run, "01_frames_and_rgb/rgb_trace.png",
                        "<b>Average R, G, B per frame.</b> Top: the colour level of the face patches over time. Bottom: the same after removing the slow level &mdash; the tiny wobble that remains is the pulse.")
              + _figure(run, "01_frames_and_rgb/channel_contribution.png",
                        "<b>Which colour contributes more.</b> (a) pulse size, (b) pulse signal-to-noise, (c) by skin patch, (d) the weight CHROM and POS give each colour. Values are printed on the bars.")
              + _note(run, "01_frames_and_rgb/channel_contribution.txt"))
    stage2 = (_figure(run, "02_pulse_signals_chrom_pos/pulse_chrom.png",
                      "<b>CHROM pulse.</b> Grey = raw projection of the three colours; green = after the band-pass filter (what the engine uses). Each red triangle is one heartbeat &mdash; counting them gives the rate.")
              + _figure(run, "02_pulse_signals_chrom_pos/pulse_pos.png", "<b>POS pulse.</b> A second, independent formula for the same pulse. When the two agree the reading is more trustworthy.")
              + _figure(run, "02_pulse_signals_chrom_pos/chrom_vs_pos.png", "<b>CHROM vs POS</b> on the same last 10 seconds."))
    stage3 = (_figure(run, "03_welch_spectrum/welch_spectrum.png",
                      "<b>Welch frequency spectrum.</b> The band-passed pulse converted to frequency (shown in beats per minute). The tallest peak is the heart rate; extra peaks are harmonics or noise &mdash; the reason a model is needed.")
              + _note(run, "03_welch_spectrum/peaks.txt"))
    stage4 = (_figure(run, "04_ensemble/selector_candidates.png", "<b>Peak selector.</b> Candidate peaks taken from the CHROM and POS spectra, each scored by 200 small decision trees. Green = the peak it chose.")
              + _figure(run, "04_ensemble/tracker_posterior.png", "<b>Spectro-temporal tracker.</b> A small convolutional network looks at the last 11 spectra. Bright = it is confident the heart rate is there; it favours peaks that persist over time.")
              + _figure(run, "04_ensemble/decisions_timeline.png", "<b>Every decision.</b> Blue = selector, orange = tracker; the final answer is the median of each, then their mean (green line).")
              + _note(run, "04_ensemble/ensemble_result.txt"))
    downloads = [(r, f) for r, f in (("01_frames_and_rgb/rgb_per_frame.csv", "RGB per frame (CSV)"), ("02_pulse_signals_chrom_pos/pulse_last10s.csv", "Pulse waveforms, last 10 s (CSV)"),
                                      ("summary.txt", "summary.txt"), ("result.json", "result.json")) if (run / r).exists()]
    dl = " &middot; ".join(f'<a href="{_u(r)}">{html.escape(f)}</a>' for r, f in downloads)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(name)} &middot; proof of concept</title><style>{CSS}</style></head><body>
<header class="top"><div class="wrap"><div class="logo">V</div><div><h1>VITAL &middot; Proof of concept</h1><div class="sub">How the heart-rate number was produced, stage by stage</div></div>
<div style="margin-left:auto"><a class="btn" href="../../index.html">&larr; All runs</a></div></div></header>
<nav class="stages"><div class="wrap"><a href="#top">Result</a><a href="#s1">1 Frames &amp; colours</a><a href="#s2">2 Pulse signals</a><a href="#s3">3 Frequency spectrum</a><a href="#s4">4 Ensemble</a></div></nav>
<div class="wrap" id="top">
<section style="margin-top:26px"><h2>{html.escape(name)} <span class="pill {html.escape(source)}">{html.escape(source or 'run')}</span> <span class="pill {html.escape(state)}">{html.escape(state)}</span></h2>
<div class="sub">{html.escape(when)} &middot; {_fmt(rs.get('duration_s'), ' s') if rs.get('duration_s') else ''} &middot; {rs.get('fps_measured', '')} frames/s &middot; face found in {rs.get('frames_with_face', '&ndash;')} of {rs.get('frames_total', '&ndash;')} frames</div>
{''.join(body)}
<div class="grid g4" style="margin-top:14px">{stats}</div>
{_short_html(con) if con else ''}
{_polish_html(polish, 'headline')}
<div class="flow"><a href="#s1">Video</a><i>&rarr;</i><a href="#s1">Frames</a><i>&rarr;</i><a href="#s1">Average R,G,B</a><i>&rarr;</i><a href="#s2">CHROM / POS pulse</a><i>&rarr;</i><a href="#s3">Band-pass + Welch spectrum</a><i>&rarr;</i><a href="#s4">Selector + tracker</a><i>&rarr;</i><a href="#s4"><b>Heart rate</b></a></div></section>
<section class="stage" id="s1"><h2><span class="n">1</span>Video to frames, and the average red, green and blue</h2>{stage1}{_concl_html(con["stages"][0]) if con else ""}{_polish_html(polish, "stage1")}</section>
<section class="stage" id="s2"><h2><span class="n">2</span>CHROM and POS: colours to a pulse signal</h2>{stage2}{_concl_html(con["stages"][1]) if con else ""}{_polish_html(polish, "stage2")}</section>
<section class="stage" id="s3"><h2><span class="n">3</span>Welch transform: pulse to frequency</h2>{stage3}{_concl_html(con["stages"][2]) if con else ""}{_polish_html(polish, "stage3")}</section>
<section class="stage" id="s4"><h2><span class="n">4</span>The ensemble: choosing the heart rate</h2>{stage4}{_concl_html(con["stages"][3]) if con else ""}{_polish_html(polish, "stage4")}</section>
<section class="callout"><b>Read this honestly.</b> Heart rate only. The ensemble models were trained on recorded sessions from public datasets (UBFC-rPPG, UBFC-Phys); if this recording is one of them its result is
optimistic. For a new person or camera the realistic error is the out-of-fold figure of about <b>3.9 BPM</b> (125 recorded sessions, 75 people; classical CHROM alone 11.35, old ML correction 6.69 on the same sessions).
Without a reference device there is no error to report for a single recording. Live-webcam accuracy has not been validated.</section>
<footer>Files: {dl or '&ndash;'}<br>Generated {time.strftime('%Y-%m-%d %H:%M')} by core/poc_site.py</footer></div>{LB_JS}</body></html>"""
    (run / "report.html").write_text(page, encoding="utf-8")


# ───────────────────────────── index ─────────────────────────────────────────────────────────────────────────────────
def build_index() -> None:
    POC_ROOT.mkdir(parents=True, exist_ok=True)
    patients, n_runs, busy = [], 0, False
    for pdir in sorted(p for p in POC_ROOT.glob("*") if p.is_dir()):
        runs = []
        for rdir in sorted((r for r in pdir.glob("*") if r.is_dir()), reverse=True):
            st, rs = _json(rdir / "status.json"), _json(rdir / "result.json")
            state = st.get("state", "done" if rs else "queued")
            busy = busy or state in ("queued", "running", "recording")
            n_runs += 1
            runs.append((rdir, st, rs, state))
        if runs:
            patients.append((pdir, runs))
    cards = []
    for pdir, runs in patients:
        rows = []
        for rdir, st, rs, state in runs:
            th = _thumb(rdir)
            when = rdir.name[:15]
            when = f"{when[:4]}-{when[4:6]}-{when[6:8]} {when[9:11]}:{when[11:13]}" if len(when) == 15 else rdir.name
            src = rs.get("source") or st.get("source") or ""
            ens, cl = rs.get("ensemble"), rs.get("classical_chrom")
            nums = f"<b>{ens:.1f}</b> <span class='sub'>BPM</span><div class='sub'>classical {cl:.1f}</div>" if ens is not None and cl is not None else "<span class='sub'>&ndash;</span>"
            rows.append(f"""<div class="run"><img src="{_u(th) if th else ''}" alt=""><div class="meta"><div><span class="pill {html.escape(src)}">{html.escape(src)}</span> <span class="pill {html.escape(state)}">{html.escape(state)}</span></div>
<div class="sub" style="margin-top:4px">{html.escape(when)}{f" &middot; {rs['duration_s']:.0f} s" if rs.get('duration_s') else ''}</div></div><div class="nums">{nums}</div>
<a class="btn" href="{_u(pdir.name + '/' + rdir.name + '/report.html')}">Open report</a></div>""")
        cards.append(f'<div class="card patient" data-name="{html.escape(pdir.name.lower())}"><h2>{html.escape(pdir.name)}</h2><div class="sub">{len(runs)} run(s)</div><div style="margin-top:8px">{"".join(rows)}</div></div>')
    refresh = '<meta http-equiv="refresh" content="10">' if busy else ""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh}
<title>VITAL &middot; Proof of concept</title><style>{CSS}</style></head><body>
<header class="top"><div class="wrap"><div class="logo">V</div><div><h1>VITAL &middot; Proof of concept</h1><div class="sub">Contactless heart rate from a webcam &mdash; every test, stage by stage</div></div>
<div style="margin-left:auto"><input class="search" id="q" placeholder="Search patient name&hellip;" oninput="f()"></div></div></header>
<div class="wrap"><div class="grid g4" style="margin:22px 0">
<div class="card stat"><div class="v">{len(patients)}</div><div class="k">patients</div></div><div class="card stat"><div class="v">{n_runs}</div><div class="k">runs</div></div>
<div class="card stat"><div class="v">{'building' if busy else 'ready'}</div><div class="k">status{' (auto-refreshing)' if busy else ''}</div></div></div>
<div class="callout" style="margin-bottom:20px">Each run is a folder showing how the heart-rate number was produced: <b>1</b> video &rarr; frames &rarr; average red/green/blue, <b>2</b> CHROM and POS pulse waveforms,
<b>3</b> Welch frequency spectrum, <b>4</b> the ensemble that picks the final rate. Heart rate only; recorded-video evidence (about 3.9 BPM error out-of-fold on 125 sessions), not a live-webcam validation.</div>
<div class="grid g2" id="list">{''.join(cards) or '<div class="card">No runs yet. Upload a video or run a live test in the dashboard.</div>'}</div>
<footer>Updated {time.strftime('%Y-%m-%d %H:%M')}</footer></div>
<script>function f(){{var q=document.getElementById('q').value.toLowerCase();document.querySelectorAll('.patient').forEach(function(c){{c.style.display=c.dataset.name.indexOf(q)>=0?'':'none'}})}}</script></body></html>"""
    (POC_ROOT / "index.html").write_text(page, encoding="utf-8")


def rebuild_all() -> None:
    for pdir in POC_ROOT.glob("*"):
        for rdir in (pdir.glob("*") if pdir.is_dir() else []):
            if rdir.is_dir():
                try:
                    build_run_page(rdir)
                except Exception as exc:                                   # noqa: BLE001
                    print("page failed for", rdir, exc)
    build_index()


if __name__ == "__main__":
    rebuild_all()
    print("rebuilt", POC_ROOT / "index.html")
