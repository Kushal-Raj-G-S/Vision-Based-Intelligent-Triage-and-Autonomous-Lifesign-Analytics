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

# ── Constants ─────────────────────────────────────────────────────────────────

UPLOAD_FOLDER = "uploads"

ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {"mp4", "avi", "mov", "mkv", "webm", "jpg", "jpeg", "png", "bmp", "webp"}
)
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {"jpg", "jpeg", "png", "bmp", "webp"}
)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

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
# in .env for deployment; defaults to the Next.js dev server only).
_cors_origins = [
    o.strip() for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
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

# ── Route Registration ────────────────────────────────────────────────────────
# Import routes AFTER app, session, camera, and rppg_engine are defined so that
# web/routes.py can import them without circular-dependency issues.

import web.routes  # noqa: E402  (intentionally late import)

log.info("Routes registered. VITAL is ready.")
