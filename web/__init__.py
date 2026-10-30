from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

from core.camera import Camera
from core.rppg import AdvancedRPPG

# ── Environment ───────────────────────────────────────────────────────────────

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vital")


class _QuietPollingFilter(logging.Filter):
    """Hide successful, high-frequency polling GETs (the frontend hits /status
    roughly once a second) so the terminal shows what actually matters —
    POSTs, non-200s, and real pipeline events — instead of a wall of
    identical 200 OK lines. Any failed request still prints normally."""

    _quiet_paths = ("/status", "/api/cameras", "/api/triage_queue")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if ' 200 ' in msg and any(f'"GET {p} ' in msg for p in self._quiet_paths):
            return False
        return True


logging.getLogger("werkzeug").addFilter(_QuietPollingFilter())

# ── Constants ─────────────────────────────────────────────────────────────────

UPLOAD_FOLDER = "uploads"

ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {"mp4", "avi", "mov", "mkv", "webm", "jpg", "jpeg", "png", "bmp", "webp"}
)
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {"jpg", "jpeg", "png", "bmp", "webp"}
)

# None = no Flask-enforced cap. This app runs single-user, localhost-only
# (see "Live camera is single-session by design" in CLAUDE.md) — the 100MB
# limit existed for a hosted-multi-user threat model that doesn't apply here.
# GB-sized real-video test files were hitting it as a hard 413 before the
# upload handler ever ran. If this app is ever exposed beyond localhost,
# reinstate a real cap here first.
MAX_UPLOAD_BYTES = None

# ── ARIA / NVIDIA NIM Config ──────────────────────────────────────────────────

ARIA_SYSTEM_PROMPT = (
    "You are ARIA — Adaptive Real-time Intelligence for Acute-care — a friendly, "
    "supportive clinical AI assistant embedded in the VITAL triage platform. "
    "You serve as a friendly consultor and helper for both clinicians and patients.\n\n"

    "YOUR ROLE:\n"
    "You assist users with friendly conversation, health queries, clinical explanations, "
    "and patient database lookups. Feel free to help with all kinds of questions to maintain "
    "a supportive experience, while gently prioritizing clinical triage and patient records "
    "whenever appropriate.\n\n"

    "RESPONSE FORMATTING CONSTRAINTS:\n"
    "  - Do NOT use markdown formatting characters like asterisks (**), hashtags (#), "
    "or backticks in your output.\n"
    "  - Use standard CAPITALized headers or clean line breaks to separate different topics.\n"
    "  - Use simple bullet points (starting with a dash '-') to present statistics or patient vitals.\n"
    "  - Keep responses highly structured, spaced out, and scannable — avoid dense paragraphs.\n\n"

    "You are ARIA. Friendly. Empathetic. Highly structured. Always clear."
)

chatbot_sys_instruct = ARIA_SYSTEM_PROMPT

nvidia_api_key: Optional[str] = os.getenv("NVIDIA_API_KEY")

if nvidia_api_key:
    log.info("NVIDIA NIM | ARIA Clinical AI configuration loaded.")
else:
    log.warning("NVIDIA_API_KEY not set — ARIA running in mock fallback mode.")

# ── Auth (optional) ───────────────────────────────────────────────────────────
# Off by default so local/dev usage is unchanged. Set API_KEY in .env to
# require an X-API-Key header on destructive endpoints (e.g. wiping the
# entire triage queue) — see require_api_key() in web/routes.py.
API_KEY: Optional[str] = os.getenv("API_KEY")
if API_KEY:
    log.info("API_KEY set — destructive endpoints now require X-API-Key header.")
else:
    log.warning(
        "API_KEY not set — destructive endpoints (e.g. DELETE /api/triage_queue) "
        "are unauthenticated. Set API_KEY in .env before exposing this beyond localhost."
    )

# ── Flask App ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

# CORS: was "origins": "*" — any website could call this API from a browser.
# Restrict to an explicit allowlist (override with CORS_ORIGINS=comma,separated
# in .env for deployment; defaults to the local Next.js dev servers: 3000 = current dashboard,
# 3001 = the previous dashboard kept running side by side).
_cors_origins = [
    o.strip() for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"
    ).split(",") if o.strip()
]
CORS(app, resources={r"/*": {"origins": _cors_origins}})
log.info("CORS allowed origins: %s", _cors_origins)

app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
)

# Resolve upload directory relative to project root (one level above this file)
_upload_dir = os.path.join(app.root_path, "..", UPLOAD_FOLDER)
os.makedirs(_upload_dir, exist_ok=True)
log.info("Upload directory: %s", os.path.abspath(_upload_dir))

# ── Metrics Schema ────────────────────────────────────────────────────────────

@dataclass
class VitalMetrics:
    """Live rPPG metrics snapshot. All fields map 1-to-1 with frontend expectations."""

    # Heart rate
    bpm: float = 0.0
    confidence: float = 0.0
    status: str = "WAITING"
    classification: str = "UNKNOWN"

    # Signal quality
    snr_db: float = 0.0
    sqi: float = 0.0
    ohi: float = 0.0
    stability: float = 0.0
    stability_indicator: str = "--"

    # Respiration
    rr: float = 0.0
    rr_confidence: float = 0.0
    rr_classification: str = "--"

    # Autonomic / stress
    hrv: float = 0.0
    stress_index: float = 0.0

    # Environment
    estimated_lux: float = 0.0
    motion_delta: float = 0.0

    # Session state
    is_live: bool = False
    calibration_done: bool = False
    calibration_progress: float = 0.0

    # Derived / display
    warnings: list[str] = field(default_factory=list)
    remark: str = ""
    ppg_signal: list[float] = field(default_factory=list)

    # Dictionary-like compatibility adapter methods
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def update(self, data: dict) -> None:
        for k, v in data.items():
            setattr(self, k, v)

    def keys(self):
        return self.__dict__.keys()

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON responses."""
        return {k: v for k, v in self.__dict__.items()}


# ── Session State ─────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    """
    Mutable session-level state shared across request handlers.

    Access is not thread-safe for compound operations — callers that need
    atomicity should acquire `processing_lock` explicitly.
    """

    # Synchronisation
    processing_lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)

    # Frame counters
    frame_count: int = 0
    start_time: float = field(default_factory=time.time)

    # Mode flags
    is_live_camera: bool = False
    image_mode: bool = False
    image_frame_bytes: Optional[bytes] = None

    # Rolling session history
    bpm_history: list[float] = field(default_factory=list)
    triage_queue: list[dict] = field(default_factory=list)

    # Per-cycle ML-corrected BPM readings (see CORRECTION_MODEL below) —
    # collected the same way scripts/analyze_video.py's corrected_readings
    # list is, so the session-end IQR-median filter matches what was
    # actually validated (6.48 BPM MAE) rather than a different aggregation.
    corrected_bpm_history: list[float] = field(default_factory=list)

    # Per-session HR ensemble (core/hr_ensemble.py: peak selector + spectro-temporal
    # tracker). None when the model files are missing/disabled — every consumer
    # then falls back to the classical/hybrid numbers above, exactly as before.
    hr_ensemble: object = None

    # Persisted "last good" autonomic values (survive momentary signal loss)
    last_valid_hrv: float = 0.0
    last_valid_stress: float = 0.0

    # Last fully compiled clinical report
    last_compiled_report: Optional[dict] = None

    # Live metrics (mutable in-place by the processing thread)
    metrics: VitalMetrics = field(default_factory=VitalMetrics)

    def reset(self) -> None:
        """Soft-reset session counters and metrics without recreating locks."""
        self.frame_count = 0
        self.start_time = time.time()
        self.is_live_camera = False
        self.image_mode = False
        self.image_frame_bytes = None
        self.bpm_history.clear()
        self.triage_queue.clear()
        self.corrected_bpm_history.clear()
        if self.hr_ensemble is not None:
            self.hr_ensemble.close()
            self.hr_ensemble = None
        self.last_valid_hrv = 0.0
        self.last_valid_stress = 0.0
        self.last_compiled_report = None
        self.metrics = VitalMetrics()
        self.stop_event.clear()
        log.info("Session state reset.")


# ── Core Components ───────────────────────────────────────────────────────────

camera = Camera(source=None)
rppg_engine = AdvancedRPPG(fps=30, window_size=300)
session = SessionState()

log.info(
    "Core components initialised — Camera: %s | rPPG engine: %s",
    camera.__class__.__name__,
    rppg_engine.__class__.__name__,
)

# ── Hybrid ML BPM Correction Model ────────────────────────────────────────────
# Same deployed model (Huber-loss GBR, 6.48 BPM session MAE out-of-fold across
# 71 real sessions — see data/evolution/12-huber-loss.md) that was previously
# only ever applied to pre-recorded videos via scripts/analyze_video.py. Loaded
# once here so the live webcam path (web/routes.py::generate_frames) can apply
# the exact same correction per cycle instead of shipping classical-DSP-only
# numbers on the live path while the offline path stays more accurate.
CORRECTION_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "bpm_correction_gbr.joblib"
)
# Feature order the deployed model was actually trained on — positional, not
# named, so this MUST match scripts/analyze_video.py's _CORRECTION_FEATURE_ORDER.
CORRECTION_FEATURE_ORDER = [
    "bpm", "sqi", "snr_db", "stability_score", "rr", "rr_confidence",
    "hrv", "stress_index", "filt_low", "filt_high",
]
correction_model = None
try:
    from core.ml_compat import load_joblib_compat
    correction_model = load_joblib_compat(CORRECTION_MODEL_PATH)
    log.info("Hybrid BPM correction model loaded — now applied on the live webcam path too.")
except Exception as exc:
    log.warning(
        "Could not load hybrid correction model (%s) — live path stays classical-DSP-only: %s",
        CORRECTION_MODEL_PATH, exc,
    )

# ── HR ensemble (peak selector + spectro-temporal tracker) ────────────────────
# numpy-only models (models/hr_selector.npz, models/hr_tracker.npz). Out-of-fold on
# 125 video sessions the mean of the two scored ~3.8-3.9 BPM MAE vs 4.5 for the hybrid
# correction above (data/evolution/22-hr-ensemble-integration.md — video evidence, not
# a live-webcam validation). None => hybrid/classical path, unchanged. Disable with
# VITAL_HR_ENSEMBLE=0.
from core.hr_ensemble import load_models as _load_hr_models  # noqa: E402
hr_models = _load_hr_models()

# ── Route Registration ────────────────────────────────────────────────────────
# Import routes AFTER app, session, camera, and rppg_engine are defined so that
# web/routes.py can import them without circular-dependency issues.

import web.routes  # noqa: E402  (intentionally late import)

log.info("Routes registered. VITAL is ready.")
