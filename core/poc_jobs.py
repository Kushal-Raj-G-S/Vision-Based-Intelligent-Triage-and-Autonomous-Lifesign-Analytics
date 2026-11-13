"""
Background proof-of-concept jobs for the app (light module: no matplotlib / scipy imported here).

Every finished upload and every live session gets its own folder
    poc/patients/<patient name>/<YYYYMMDD_HHMMSS>_<upload|live>/
filled by core/poc_report.py, which runs as a SEPARATE low-priority process. So:
  * the app is never slowed by it (own process, BELOW_NORMAL priority, no console window);
  * a crash / slow job / missing library cannot break a measurement: every entry point here swallows its own errors;
  * several can run in parallel (MAX_PARALLEL), the rest wait in a queue.
VITAL_POC=0 turns the whole thing off.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()                         # don't rely on some other module having loaded .env first (found 2026-11-04:
                                       # llm_polish_enabled() silently always skipped when nothing else had yet)

log = logging.getLogger("vital.poc")

ROOT = Path(__file__).resolve().parent.parent
POC_ROOT = ROOT / "poc" / "patients"
MAX_PARALLEL = 2
MAX_TRACE_ROWS = 200_000            # ~55 min at 60 rows/s; a longer live session is simply not recorded past this
_SEM = threading.Semaphore(MAX_PARALLEL)

# AI-polished wording for the conclusions (core/poc_llm_polish.py) — a SEPARATE, isolated venv (nemoguardrails +
# langchain live only here, never in the app's own venv) and its own semaphore, so it never competes with, slows,
# or is slowed by the report-builder jobs above or the main triage crew's own LLM pool/API usage.
LLM_VENV_PY = ROOT / "tools" / "poc_llm" / "venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
_LLM_SEM = threading.Semaphore(2)


def enabled() -> bool:
    return os.environ.get("VITAL_POC", "1").strip().lower() not in ("0", "false", "off", "no")


def llm_polish_enabled() -> bool:
    return (os.environ.get("VITAL_POC_LLM", "1").strip().lower() not in ("0", "false", "off", "no")
            and LLM_VENV_PY.exists() and bool(os.environ.get("NVIDIA_API_KEY_POC") or os.environ.get("NVIDIA_API_KEY")))


def _safe(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._ -]+", "_", (name or "").strip()).strip(" ._")[:60]
    return s or "Unnamed"


def new_run_dir(patient: str, source: str) -> Path:
    d = POC_ROOT / _safe(patient) / f"{time.strftime('%Y%m%d_%H%M%S')}_{source}"
    n = 1
    base = d
    while d.exists():
        n += 1
        d = Path(f"{base}_{n}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _status(run_dir: Path, **kw) -> None:
    p = run_dir / "status.json"
    cur = {}
    try:
        cur = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        pass
    cur.update(kw)
    cur["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p.write_text(json.dumps(cur, indent=1), encoding="utf-8")


def _run(run_dir: Path, args: list) -> None:
    with _SEM:
        try:
            _status(run_dir, state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))
            refresh(run_dir)
            flags = 0
            if os.name == "nt":
                flags = 0x00004000 | 0x08000000                       # BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW
            with open(run_dir / "build.log", "wb") as lg:
                rc = subprocess.Popen([sys.executable, "-m", "core.poc_report", *args], cwd=str(ROOT), stdout=lg, stderr=subprocess.STDOUT,
                                      creationflags=flags).wait()
            _status(run_dir, state="done" if rc == 0 else "failed", returncode=rc)
        except Exception as exc:                                       # noqa: BLE001
            try:
                _status(run_dir, state="failed", error=f"{type(exc).__name__}: {exc}")
            except Exception:
                pass
            log.warning("POC job failed for %s: %s", run_dir, exc)
            rc = 1
        finally:
            refresh(run_dir)
    if rc == 0 and llm_polish_enabled():
        threading.Thread(target=_run_llm_polish, args=(run_dir,), name="poc-llm-polish", daemon=True).start()


def _run_llm_polish(run_dir: Path) -> None:
    """Best-effort only: the plain report built above is already complete and correct. If this adds nothing
    (disabled, no model reachable, nothing passed the guardrail), the page the viewer sees does not change."""
    with _LLM_SEM:
        try:
            flags = 0
            if os.name == "nt":
                flags = 0x00004000 | 0x08000000                       # BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW
            with open(run_dir / "llm_polish.log", "wb") as lg:
                subprocess.Popen([str(LLM_VENV_PY), "-m", "core.poc_llm_polish", "--run", str(run_dir)], cwd=str(ROOT),
                                 stdout=lg, stderr=subprocess.STDOUT, creationflags=flags).wait()
        except Exception as exc:                                       # noqa: BLE001
            log.info("POC LLM polish skipped for %s: %s", run_dir, exc)


def _submit(run_dir: Path, args: list, meta: dict) -> Optional[Path]:
    try:
        _status(run_dir, state="queued", **meta)
        refresh(run_dir)
        threading.Thread(target=_run, args=(run_dir, args), name="poc-job", daemon=True).start()
        log.info("POC queued: %s", run_dir)
        return run_dir
    except Exception as exc:                                           # noqa: BLE001
        log.warning("POC submit failed: %s", exc)
        return None


def submit_video(video_path: str, patient: str) -> Optional[Path]:
    """An uploaded video finished playing: build its proof folder from the file."""
    if not enabled():
        return None
    try:
        d = new_run_dir(patient or Path(video_path).stem, "upload")
        return _submit(d, ["--video", str(video_path), "--out", str(d), "--name", patient or "Unnamed", "--source", "upload"],
                       {"patient": patient, "source": "upload", "video": str(video_path)})
    except Exception as exc:                                           # noqa: BLE001
        log.warning("POC video submit failed: %s", exc)
        return None


class LiveTraceRecorder:
    """Records one live session cheaply (numbers every frame, one clean JPEG per second) so the proof can be built after Stop."""

    def __init__(self, patient: str):
        self.patient = patient
        self.run_dir = new_run_dir(patient, "live")
        self.frames_dir = self.run_dir / "_live_frames"
        self.frames_dir.mkdir(exist_ok=True)
        self.t, self.rgb, self.motion, self.parts = [], [], [], []
        self.frame_t, self.frame_files, self.frame_boxes = [], [], []
        self._last_save = -1e9
        self._t0 = None
        self._closed = False
        _status(self.run_dir, state="recording", patient=patient, source="live")

    def add(self, now: float, rgb, motion: float, parts, clean_frame, face_box) -> None:
        """now = time.perf_counter(); parts = the 9 per-patch colour means or None; clean_frame = BGR ndarray or None."""
        if self._closed or len(self.t) >= MAX_TRACE_ROWS:
            return
        try:
            if self._t0 is None:
                self._t0 = now
            self.t.append(now); self.rgb.append(tuple(rgb)); self.motion.append(float(motion))
            self.parts.append(list(parts) if parts is not None else [float("nan")] * 9)
            if clean_frame is not None and face_box is not None and now - self._last_save >= 1.0:
                import cv2
                name = f"frame_{len(self.frame_files):05d}.jpg"
                if cv2.imwrite(str(self.frames_dir / name), clean_frame):
                    self.frame_t.append(now - self._t0); self.frame_files.append(name); self.frame_boxes.append(list(face_box[:4]))
                    self._last_save = now
        except Exception as exc:                                       # noqa: BLE001
            log.debug("POC recorder add failed: %s", exc)

    def finish(self) -> Optional[Path]:
        """Stop recording and queue the report. Safe to call twice."""
        if self._closed:
            return None
        self._closed = True
        try:
            if len(self.t) < 90:                                       # < ~3 s: nothing worth analysing
                _status(self.run_dir, state="skipped", reason=f"only {len(self.t)} frames with a face")
                return None
            import numpy as np
            np.savez_compressed(self.run_dir / "live_trace.npz", t=np.array(self.t), rgb=np.array(self.rgb), motion=np.array(self.motion),
                                parts=np.array(self.parts), frame_t=np.array(self.frame_t), frame_files=np.array(self.frame_files),
                                frame_boxes=np.array(self.frame_boxes).reshape(-1, 4))
            return _submit(self.run_dir, ["--trace", str(self.run_dir / "live_trace.npz"), "--out", str(self.run_dir), "--name", self.patient or "Unnamed",
                                          "--source", "live"], {"patient": self.patient, "source": "live", "frames": len(self.t)})
        except Exception as exc:                                       # noqa: BLE001
            log.warning("POC live finish failed: %s", exc)
            return None


def write_index() -> None:
    """poc/patients/index.html (patient cards, status, numbers) - built by core/poc_site.py."""
    from core import poc_site
    poc_site.build_index()


def refresh(run_dir: Path) -> None:
    """Rebuild this run's report page and the index. Never raises."""
    try:
        from core import poc_site
        poc_site.build_run_page(run_dir)
        poc_site.build_index()
    except Exception as exc:                                           # noqa: BLE001
        log.debug("POC page refresh failed: %s", exc)
