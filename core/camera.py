"""
VITAL — Camera Module  (MediaPipe Face Mesh Edition)
=====================================================
468-landmark face mesh for anatomically precise, skin-tone-inclusive
forehead / cheek ROI extraction for rPPG.

Key upgrades vs. previous version
──────────────────────────────────
1. Landmark indices corrected to clinically-validated rPPG anchors:
     Forehead  → 151  (mid-forehead, validated by R2I-rPPG / Nagar et al. 2024)
     Left cheek → 50  (malar surface, same source)
     Right cheek → 280 (malar surface, same source)
   Previous indices [10, 338, 297, 332, 284] were hairline-boundary points
   (too high, prone to hair occlusion).
   Previous cheek indices were jaw-contour points, not malar skin surface.

2. Face-scale-adaptive ROI sizing — box dimensions are derived from the
   inter-eye distance so signal area stays proportional across distances.

3. Yaw-aware cheek gating — if head turns >~25° the occluded cheek is
   dropped and a nose-tip ROI (landmark 4) is substituted, preventing
   specular noise from an angled cheek entering the signal.

4. Weighted spatial pooling updated to a 3-region scheme with tunable
   weights; forehead weight raised to 0.5 (higher vascularity, less hair).
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("vital.camera")

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = hasattr(mp, "solutions")
    if not _HAS_MEDIAPIPE:
        log.info("mediapipe 0.10+ detected — using Haar Cascade fallback (fully operational)")
except ImportError:
    _HAS_MEDIAPIPE = False
    log.info("mediapipe not installed — using Haar Cascade fallback (fully operational)")


# ── Robust spatial pooling ─────────────────────────────────────────────────────

# Fraction of pixels excluded from each ROI before averaging, ranked by
# luminance — darkest (shadows, dark circles, hair strands, acne shading)
# and brightest (specular glare) outliers within the box are dropped so a
# localised blemish doesn't bias the pooled colour used for rPPG. Pixels are
# ranked by luminance (not per-channel independently) so the retained set's
# R/G/B ratio stays intact — CHROM/POS both depend on inter-channel ratios,
# not absolute brightness.
_TRIM_DARK_FRAC   = 0.10   # drop darkest 10% (shadow/dark-spot rejection)
_TRIM_BRIGHT_FRAC = 0.05   # drop brightest 5% (specular-glare rejection)
_TRIM_MIN_PIXELS  = 20     # ROI boxes smaller than this just use a plain mean


def _robust_channel_means(roi: np.ndarray) -> np.ndarray:
    """Trimmed-mean BGR pooling — see module docstring above for rationale."""
    pixels = roi.reshape(-1, 3).astype(np.float64)
    n = pixels.shape[0]
    if n < _TRIM_MIN_PIXELS:
        return np.mean(pixels, axis=0)

    # BT.601 luminance (frame is BGR)
    luminance = 0.114 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.299 * pixels[:, 2]
    order = np.argsort(luminance)
    lo = int(n * _TRIM_DARK_FRAC)
    hi = int(n * (1.0 - _TRIM_BRIGHT_FRAC))
    keep = order[lo:hi]
    if len(keep) < _TRIM_MIN_PIXELS:
        return np.mean(pixels, axis=0)
    return np.mean(pixels[keep], axis=0)


# ── Platform helpers ──────────────────────────────────────────────────────────

def _capture_backend() -> int:
    """Prefer DirectShow on Windows for faster, more reliable webcam open."""
    return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


def _enumerate_windows_cameras() -> list[str]:
    """
    Use PowerShell to enumerate camera devices with friendly names.
    Internal/integrated webcams are sorted to the front.
    """
    import subprocess
    try:
        cmd = (
            "Get-PnpDevice -Class Camera,Image -Status OK "
            "| Where-Object { $_.Present -eq $true } "
            "| Sort-Object -Property @{Expression={"
            "$_.FriendlyName -notmatch 'phone|link|virtual|obs|camo|iriun|droid|ivcam'"
            "};Descending=$true} "
            "| Select-Object -ExpandProperty FriendlyName"
        )
        result = subprocess.run(
            ["powershell", "-Command", cmd],
            capture_output=True, text=True, timeout=8,
        )
        return [n.strip() for n in result.stdout.strip().splitlines() if n.strip()]
    except Exception:
        return []


def _get_camera_name_windows(cv_index: int, win_names: list[str]) -> str:
    return win_names[cv_index] if cv_index < len(win_names) else f"Camera {cv_index}"


# ── Camera list cache ─────────────────────────────────────────────────────────

_cam_cache: list[dict] = []
_cam_cache_populated: bool = False

_VIRTUAL_KEYWORDS = frozenset({
    "phone", "link", "virtual", "obs", "droidcam", "epoccam",
    "iriun", "ivcam", "camo", "ndispi", "snap camera",
    "logi capture", "xsplit", "ndi", "manycam",
})


def find_available_cameras(max_index: int = 5) -> list[dict]:
    """
    Return available camera devices:
        [{'index': int, 'label': str, 'is_virtual': bool}, ...]

    Results are permanently cached — re-probing DirectShow while a capture
    is open causes C++ exceptions on Windows.
    """
    global _cam_cache, _cam_cache_populated

    if _cam_cache_populated:
        return _cam_cache

    win_names = _enumerate_windows_cameras() if sys.platform == "win32" else []
    backend = _capture_backend()
    cam_list: list[dict] = []

    for index in range(max_index):
        cap = None
        try:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                continue
            ret, frame = cap.read()
            cap.release()
            cap = None
            if ret and frame is not None and frame.size > 0:
                name = _get_camera_name_windows(index, win_names)
                is_virtual = any(kw in name.lower() for kw in _VIRTUAL_KEYWORDS)
                cam_list.append({"index": index, "label": name, "is_virtual": is_virtual})
        except Exception:
            pass
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    cam_list.sort(key=lambda c: (c["is_virtual"], c["index"]))

    if cam_list:
        log.info("Detected cameras: %s", [(c["index"], c["label"], c["is_virtual"]) for c in cam_list])
        log.info("Default (built-in): index=%d — %s", cam_list[0]["index"], cam_list[0]["label"])
    else:
        log.warning("No cameras found during probe.")

    _cam_cache = cam_list
    _cam_cache_populated = True
    return cam_list


# ── MediaPipe landmark constants ──────────────────────────────────────────────
#
# Clinically validated anchors for rPPG ROI (Nagar et al., R2I-rPPG, 2024;
# aligned with standard usage in pyVHR / yarppg literature):
#
#   151  — mid-forehead (center of forehead skin, clear of hair)
#   50   — left malar cheek surface
#   280  — right malar cheek surface
#
# Secondary / nose-bridge anchor used when a cheek is yaw-occluded:
#   4    — nose tip (skin always visible, lower rPPG quality but beats noise)
#
# Inter-ocular distance landmarks (for adaptive ROI sizing):
#   33   — outer-left eye corner
#   263  — outer-right eye corner
#
# These are stable across mediapipe 0.8–0.10 legacy solutions.

_LM_FOREHEAD  = 151   # mid-forehead anchor
_LM_LEFT_CHEEK  = 117  # left malar surface (below the eye on the cheek)
_LM_RIGHT_CHEEK = 346  # right malar surface (below the eye on the cheek)
_LM_NOSE_TIP    = 4   # fallback when cheek is occluded
_LM_EYE_LEFT    = 33  # outer left eye corner  (for IOD)
_LM_EYE_RIGHT   = 263 # outer right eye corner (for IOD)

# Yaw threshold (normalised x-delta between left/right eye corners).
# When abs(lm[_LM_EYE_LEFT].x - lm[_LM_EYE_RIGHT].x) < this, the face
# has rotated enough that the near-side cheek becomes unreliable.
_YAW_OCCLUDE_THRESHOLD = 0.12  # ≈ ±25° head rotation

# Spatial pooling weights  [forehead, left cheek, right cheek]
# Forehead has highest vascularity and fewest hair/beard artefacts.
_POOL_WEIGHTS = (0.50, 0.25, 0.25)

# ROI box size expressed as a fraction of inter-ocular distance (IOD).
# This keeps the sampling area proportional regardless of camera distance.
_FH_ROI_W_FACTOR = 0.90   # forehead width  ~ 0.9 × IOD
_FH_ROI_H_FACTOR = 0.45   # forehead height ~ 0.45 × IOD
_CK_ROI_W_FACTOR = 0.55   # cheek width
_CK_ROI_H_FACTOR = 0.45   # cheek height


class Camera:
    """
    rPPG camera module with MediaPipe Face Mesh.
    Falls back to Haar Cascade if mediapipe is unavailable.

    MediaPipe mode  — uses validated landmark anchors (151 / 50 / 280) and
                      face-scale-adaptive ROI sizing.
    Haar mode       — EMA-smoothed bounding box with proportional ROI offsets.
    """

    def __init__(self, source=None, show_face_mesh: bool = True):
        self.video: Optional[cv2.VideoCapture] = None
        self.dummy_mode = True
        self.video_ended = False
        self._source = None

        # Draws all 468 tracked landmarks as small dots ("full mesh"
        # visualisation) — purely cosmetic confirmation that MediaPipe is
        # tracking the whole face; it does NOT change which pixels feed the
        # rPPG signal (that's still just the forehead/cheek ROI boxes below).
        self.show_face_mesh = show_face_mesh

        # Haar-mode EMA state
        self.last_x = self.last_y = self.last_w = self.last_h = 0
        self.alpha = 0.2
        self.is_moving = False

        # Smoothed face box of the most recent frame, in pixels, plus the frame
        # size: (x, y, w, h, frame_w, frame_h). None whenever no face was found
        # on that frame. Read by web/routes.py to give the kiosk its
        # "move closer / left / right" guidance; display-only, never feeds the
        # rPPG signal.
        self.last_face_box: Optional[tuple] = None
        # Research hook (scripts/build_roi_cache.py). When True, _extract_roi_mediapipe also stores per-ROI colour
        # means (as used, and before the debug outline is drawn) in `last_roi_parts`. Off = zero extra work.
        self.record_roi_parts = False
        self.last_roi_parts: Optional[dict] = None

        # Landmark-based motion tracking (MediaPipe mode)
        self._prev_cx: Optional[float] = None
        self._prev_cy: Optional[float] = None

        # ── Detector init ────────────────────────────────────────────
        self.use_mediapipe = _HAS_MEDIAPIPE
        self.face_mesh = None
        self.face_cascade = None

        if self.use_mediapipe:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            log.info("MediaPipe Face Mesh initialised (landmarks: FH=%d LC=%d RC=%d)",
                     _LM_FOREHEAD, _LM_LEFT_CHEEK, _LM_RIGHT_CHEEK)
        else:
            import os
            cascade_path = "haarcascade_frontalface_default.xml"
            if not os.path.exists(cascade_path):
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                log.error("Haar cascade failed to load!")
                self.face_cascade = None
            else:
                log.info("Haar Cascade loaded: %s", cascade_path)

        if source is not None:
            self._open_source(source)
        else:
            log.info("No video source — waiting for upload or live session.")

    # ── Source management ─────────────────────────────────────────────────────

    def _open_source(self, source) -> bool:
        self.release_video()
        label = (
            f"video file: {source}"
            if isinstance(source, str) and not str(source).isdigit()
            else f"camera index: {source}"
        )
        log.info("Opening %s", label)
        try:
            cap = (
                cv2.VideoCapture(source, _capture_backend())
                if isinstance(source, int)
                else cv2.VideoCapture(source)
            )
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    log.info("Successfully opened source: %s", source)
                    self.video = cap
                    self.dummy_mode = False
                    self.video_ended = False
                    self._source = source
                    return True
                log.warning("Failed to read first frame from: %s", source)
                cap.release()
            else:
                log.warning("Failed to open source: %s", source)
        except Exception as exc:
            log.error("Error opening video source: %s", exc)
        self.dummy_mode = True
        self._source = None
        return False

    def get_source_fps(self, default: float = 30.0) -> float:
        """
        Return the ACTUAL frame rate of the opened source, not an assumed
        constant. Every AdvancedRPPG construction site in this codebase used
        to hardcode fps=30 regardless of the real video's frame rate — fine
        for sources that genuinely are ~30fps, but for any video encoded at
        a different rate (e.g. 23.98fps, common for several UBFC-rPPG
        subjects), this caused a deterministic, silent bias: analyzing
        frames at the wrong assumed sample rate scales every frequency in
        the Welch PSD by (assumed_fps / actual_fps), which showed up as a
        confident, rock-stable, but simply WRONG heart rate — a completely
        different failure mode from the harmonic lock-on bug (confirmed via
        real-dataset trace: subject25 at 23.2fps read ~115bpm against a true
        91.5bpm; 91.5 * (30/23.2) ≈ 118, matching almost exactly).
        """
        if self.video is not None:
            try:
                fps = float(self.video.get(cv2.CAP_PROP_FPS))
                if fps and fps > 1.0:  # guard against 0/NaN/nonsense values some backends return
                    return fps
            except Exception:
                pass
        return default

    def get_total_frames(self) -> Optional[int]:
        """
        Total frame count for a video FILE source, or None for a live camera
        (an infinite/unbounded stream, or a device that doesn't report a
        frame count at all). Used only to show upload playback progress
        ("0:32 / 0:52") — never for anything DSP-affecting, so an
        occasionally-inaccurate container-metadata count (a known quirk of
        some codecs) is an acceptable display-only approximation.
        """
        if self.video is None or isinstance(self._source, int):
            return None
        try:
            count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            return count if count > 0 else None
        except Exception:
            return None

    def release_video(self) -> None:
        """Release the capture device; keep face detectors alive."""
        if self.video is not None:
            try:
                self.video.release()
            except Exception:
                pass
            self.video = None
        self.dummy_mode = True
        self.video_ended = False
        self._source = None
        self._prev_cx = self._prev_cy = None
        self.last_x = self.last_y = self.last_w = self.last_h = 0

    def release(self) -> None:
        """Release all hardware and detector resources."""
        self.release_video()
        if self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except Exception:
                pass
            self.face_mesh = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def get_frame(self, annotate: bool = True, pulse: bool = False):
        """
        Capture one frame and extract ROI signals.

        `annotate=False` returns a CLEAN frame (no debug face/ROI rectangles
        or labels) for patient-facing displays; the ROI extraction itself is
        identical either way — it simply draws on a throwaway copy.

        `pulse=True` blends the colour-magnified pulse map over the face in the
        returned JPEG only (display copy; never fed to the rPPG engine).

        Returns
        -------
        (frame_bytes, roi_data, is_moving, motion_delta)
            frame_bytes : bytes | None   — JPEG-encoded annotated frame
            roi_data    : tuple | None   — (r, g, b) spatially pooled means
            is_moving   : bool
            motion_delta: float          — pixel-space centroid drift
        """
        if self.dummy_mode:
            frame = self._create_dummy_frame()
            _, buffer = cv2.imencode(".jpg", frame)
            return buffer.tobytes(), None, False, 0.0

        try:
            success, frame = self.video.read()
            if not success:
                self.video_ended = True
                log.info("End of video reached.")
                return None, None, False, 0.0
        except Exception as exc:
            log.error("Error reading frame: %s", exc)
            return None, None, False, 0.0

        clean = None if (annotate and not pulse) else frame.copy()

        if self.use_mediapipe:
            roi_data, _, motion_delta = self._extract_roi_mediapipe(frame)
        else:
            roi_data, _, motion_delta = self._extract_roi_haar(frame)

        try:
            shown = frame if annotate else clean
            if pulse:
                if getattr(self, "_pulse_viz", None) is None:
                    from core.pulse_viz import PulseViz
                    self._pulse_viz = PulseViz()
                # signal from the un-annotated frame; paint onto whatever is being shown (ROI boxes included)
                shown = self._pulse_viz.apply(clean, self.last_face_box, target=shown)
            _, buffer = cv2.imencode(".jpg", shown)
            return buffer.tobytes(), roi_data, self.is_moving, motion_delta
        except Exception as exc:
            log.error("Error encoding frame: %s", exc)
            return None, None, False, 0.0

    def analyze_image_file(self, image_path: str) -> Optional[dict]:
        """Analyse a still image; return annotated frame bytes and basic signals."""
        frame = cv2.imread(image_path)
        if frame is None:
            return None

        if self.use_mediapipe:
            roi_data, _, _ = self._extract_roi_mediapipe(frame)
        else:
            roi_data, _, _ = self._extract_roi_haar(frame)

        if roi_data is not None:
            r_ch, g_ch, b_ch = roi_data
            lux = int(0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch)
            face_detected = True
        else:
            lux = 0
            face_detected = False

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return None

        return {
            "face_detected": face_detected,
            "estimated_lux": lux,
            "frame_bytes": buffer.tobytes(),
        }

    # ── MediaPipe ROI extraction ──────────────────────────────────────────────

    def _extract_roi_mediapipe(self, frame):
        """
        Extract forehead + cheek ROIs via validated MediaPipe landmark anchors.

        Strategy
        ─────────
        • Compute inter-ocular distance (IOD) from eye-corner landmarks 33 & 263
          so ROI dimensions scale with apparent face size / camera distance.
        • Yaw gating: if the face has turned more than ~25° the far cheek's
          signal becomes unreliable (specular, hair occlusion). That cheek is
          replaced by a small nose-tip ROI which stays skin-visible at wider yaw.
        • Spatial pooling: 50% forehead + 25% left + 25% right (or nose sub).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            cv2.putText(frame, "NO FACE DETECTED", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return None, None, 0.0

        lm = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        def pt(idx: int) -> tuple[int, int]:
            return int(lm[idx].x * w), int(lm[idx].y * h)

        # ── Inter-ocular distance for adaptive sizing ─────────────────
        ex_l, ey_l = pt(_LM_EYE_LEFT)
        ex_r, ey_r = pt(_LM_EYE_RIGHT)
        iod = max(abs(ex_r - ex_l), 20)  # pixels; floor at 20 to avoid div-zero

        fh_rw = max(int(_FH_ROI_W_FACTOR * iod), 20)
        fh_rh = max(int(_FH_ROI_H_FACTOR * iod), 12)
        ck_rw = max(int(_CK_ROI_W_FACTOR * iod), 14)
        ck_rh = max(int(_CK_ROI_H_FACTOR * iod), 10)

        # ── Forehead ROI (landmark 151) ───────────────────────────────
        fh_cx, fh_cy = pt(_LM_FOREHEAD)
        motion_delta = self._compute_motion_delta(fh_cx, fh_cy)
        fh_x1 = max(0, fh_cx - fh_rw // 2)
        fh_y1 = max(0, fh_cy - fh_rh // 2)
        fh_x2 = min(w, fh_cx + fh_rw // 2)
        fh_y2 = min(h, fh_cy + fh_rh // 2)
        fh_roi = frame[fh_y1:fh_y2, fh_x1:fh_x2]

        # ── Yaw detection ─────────────────────────────────────────────
        # Normalised x-distance between eye corners shrinks as face rotates.
        norm_iod = abs(lm[_LM_EYE_RIGHT].x - lm[_LM_EYE_LEFT].x)
        face_turned_left  = lm[_LM_EYE_LEFT].x < lm[_LM_EYE_RIGHT].x and norm_iod < _YAW_OCCLUDE_THRESHOLD
        face_turned_right = lm[_LM_EYE_RIGHT].x < lm[_LM_EYE_LEFT].x and norm_iod < _YAW_OCCLUDE_THRESHOLD

        # ── Left cheek ROI (landmark 50) or nose fallback ─────────────
        lc_label = "L CHEEK"
        if face_turned_left:
            # Left cheek is occluded — use nose tip instead
            lc_cx, lc_cy = pt(_LM_NOSE_TIP)
            lc_label = "NOSE (L)"
        else:
            lc_cx, lc_cy = pt(_LM_LEFT_CHEEK)
        lc_x1 = max(0, lc_cx - ck_rw // 2)
        lc_y1 = max(0, lc_cy - ck_rh // 2)
        lc_x2 = min(w, lc_cx + ck_rw // 2)
        lc_y2 = min(h, lc_cy + ck_rh // 2)
        lc_roi = frame[lc_y1:lc_y2, lc_x1:lc_x2]

        # ── Right cheek ROI (landmark 280) or nose fallback ───────────
        rc_label = "R CHEEK"
        if face_turned_right:
            # Right cheek is occluded — use nose tip instead
            rc_cx, rc_cy = pt(_LM_NOSE_TIP)
            rc_label = "NOSE (R)"
        else:
            rc_cx, rc_cy = pt(_LM_RIGHT_CHEEK)
        rc_x1 = max(0, rc_cx - ck_rw // 2)
        rc_y1 = max(0, rc_cy - ck_rh // 2)
        rc_x2 = min(w, rc_cx + ck_rw // 2)
        rc_y2 = min(h, rc_cy + ck_rh // 2)
        rc_roi = frame[rc_y1:rc_y2, rc_x1:rc_x2]

        if fh_roi.size == 0 or lc_roi.size == 0 or rc_roi.size == 0:
            return None, None, motion_delta

        if self.record_roi_parts:
            # means BEFORE the overlay below draws its outline on the ROI border (the production values are
            # taken after it); recorded so a replay can test whether the outline matters
            _clean = (_robust_channel_means(fh_roi), _robust_channel_means(lc_roi), _robust_channel_means(rc_roi))
            _mouth = abs(lm[14].y - lm[13].y) * h / max(iod, 1)       # inner-lip gap / inter-ocular distance
        # ── Visual overlay ────────────────────────────────────────────
        cv2.rectangle(frame, (fh_x1, fh_y1), (fh_x2, fh_y2), (255, 80, 0), 2)
        cv2.rectangle(frame, (lc_x1, lc_y1), (lc_x2, lc_y2), (0, 220, 220), 2)
        cv2.rectangle(frame, (rc_x1, rc_y1), (rc_x2, rc_y2), (0, 220, 220), 2)
        cv2.putText(frame, "FOREHEAD", (fh_x1, fh_y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 80, 0), 1)
        cv2.putText(frame, lc_label, (lc_x1, lc_y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 220), 1)
        cv2.putText(frame, rc_label, (rc_x1, rc_y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 220), 1)

        # Face bounding box
        all_x = [int(l.x * w) for l in lm]
        all_y = [int(l.y * h) for l in lm]
        cv2.rectangle(frame, (min(all_x), min(all_y)),
                      (max(all_x), max(all_y)), (0, 255, 0), 1)

        # ── Full-mesh visualisation (cosmetic only — confirms whole-face
        # tracking coverage; does NOT change which pixels feed the signal).
        # radius=2 + a colour distinct from every other overlay (forehead
        # orange, cheek yellow, face-box green) — 1px dots were getting
        # lost in JPEG compression at normal webcam resolution. ──────────
        if self.show_face_mesh:
            for px, py in zip(all_x, all_y):
                cv2.circle(frame, (px, py), 2, (255, 0, 255), -1)

        # ── Spatial pooling (trimmed mean — see _robust_channel_means) ────
        fh_bgr = _robust_channel_means(fh_roi)
        lc_bgr = _robust_channel_means(lc_roi)
        rc_bgr = _robust_channel_means(rc_roi)

        w_fh, w_lc, w_rc = _POOL_WEIGHTS
        b = w_fh * fh_bgr[0] + w_lc * lc_bgr[0] + w_rc * rc_bgr[0]
        g = w_fh * fh_bgr[1] + w_lc * lc_bgr[1] + w_rc * rc_bgr[1]
        r = w_fh * fh_bgr[2] + w_lc * lc_bgr[2] + w_rc * rc_bgr[2]

        if self.record_roi_parts:
            self.last_roi_parts = {
                "prod": (fh_bgr, lc_bgr, rc_bgr), "clean": _clean, "mouth_open": float(_mouth),
                "iod": float(iod), "norm_iod": float(norm_iod),
                "box": (min(all_x), min(all_y), max(all_x), max(all_y)),
                "turned": (bool(face_turned_left), bool(face_turned_right)),
            }
        return (r, g, b), fh_roi, motion_delta

    # ── Haar fallback ─────────────────────────────────────────────────────────

    def _extract_roi_haar(self, frame):
        """Extract forehead + cheek ROIs via Haar Cascade with EMA smoothing."""
        if self.face_cascade is None:
            cv2.putText(frame, "NO FACE DETECTOR", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return None, None, 0.0

        self.last_face_box = None
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                cv2.putText(frame, "NO FACE DETECTED", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                return None, None, 0.0

            x, y, bw, bh = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            motion_delta = 0.0

            if self.last_w == 0:
                self.last_x, self.last_y, self.last_w, self.last_h = x, y, bw, bh
                self.is_moving = False
            else:
                new_x = int(self.alpha * x + (1 - self.alpha) * self.last_x)
                new_y = int(self.alpha * y + (1 - self.alpha) * self.last_y)
                motion_delta = ((new_x - self.last_x) ** 2 + (new_y - self.last_y) ** 2) ** 0.5
                self.is_moving = motion_delta > (self.last_w * 0.03)
                self.last_x, self.last_y = new_x, new_y
                self.last_w = int(self.alpha * bw + (1 - self.alpha) * self.last_w)
                self.last_h = int(self.alpha * bh + (1 - self.alpha) * self.last_h)

            sx, sy, sw, sh = self.last_x, self.last_y, self.last_w, self.last_h
            fr = frame
            self.last_face_box = (sx, sy, sw, sh, frame.shape[1], frame.shape[0])

            def _roi(ox_frac, oy_frac, ow_frac, oh_frac):
                x1 = max(0, sx + int(sw * ox_frac))
                y1 = max(0, sy + int(sh * oy_frac))
                rw = min(int(sw * ow_frac), fr.shape[1] - x1)
                rh = min(int(sh * oh_frac), fr.shape[0] - y1)
                return x1, y1, rw, rh

            fh_x, fh_y, fh_w, fh_h = _roi(0.25, 0.05, 0.50, 0.20)
            lc_x, lc_y, lc_w, lc_h = _roi(0.15, 0.45, 0.30, 0.20)
            rc_x, rc_y, rc_w, rc_h = _roi(0.55, 0.45, 0.30, 0.20)

            _clean = None
            if self.record_roi_parts:
                # research hook: colour means BEFORE the debug outlines below are drawn onto the ROI borders
                try:
                    _clean = (_robust_channel_means(frame[fh_y:fh_y + fh_h, fh_x:fh_x + fh_w]),
                              _robust_channel_means(frame[lc_y:lc_y + lc_h, lc_x:lc_x + lc_w]),
                              _robust_channel_means(frame[rc_y:rc_y + rc_h, rc_x:rc_x + rc_w]))
                except Exception:
                    _clean = None

            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
            cv2.rectangle(frame, (fh_x, fh_y), (fh_x + fh_w, fh_y + fh_h), (255, 80, 0), 2)
            cv2.rectangle(frame, (lc_x, lc_y), (lc_x + lc_w, lc_y + lc_h), (0, 220, 220), 2)
            cv2.rectangle(frame, (rc_x, rc_y), (rc_x + rc_w, rc_y + rc_h), (0, 220, 220), 2)
            cv2.putText(frame, "FOREHEAD", (fh_x, fh_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 80, 0), 1)
            cv2.putText(frame, "L CHEEK", (lc_x, lc_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 220), 1)
            cv2.putText(frame, "R CHEEK", (rc_x, rc_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 220), 1)

            fh_roi = frame[fh_y:fh_y + fh_h, fh_x:fh_x + fh_w]
            lc_roi = frame[lc_y:lc_y + lc_h, lc_x:lc_x + lc_w]
            rc_roi = frame[rc_y:rc_y + rc_h, rc_x:rc_x + rc_w]

            if fh_roi.size == 0 or lc_roi.size == 0 or rc_roi.size == 0:
                return None, None, motion_delta

            fh_bgr = _robust_channel_means(fh_roi)
            lc_bgr = _robust_channel_means(lc_roi)
            rc_bgr = _robust_channel_means(rc_roi)

            w_fh, w_lc, w_rc = _POOL_WEIGHTS
            b = w_fh * fh_bgr[0] + w_lc * lc_bgr[0] + w_rc * rc_bgr[0]
            g = w_fh * fh_bgr[1] + w_lc * lc_bgr[1] + w_rc * rc_bgr[1]
            r = w_fh * fh_bgr[2] + w_lc * lc_bgr[2] + w_rc * rc_bgr[2]

            if self.record_roi_parts and _clean is not None:
                self.last_roi_parts = {"prod": (fh_bgr, lc_bgr, rc_bgr), "clean": _clean, "mouth_open": float("nan"),
                                       "iod": float("nan"), "norm_iod": float("nan"),
                                       "box": (sx, sy, sx + sw, sy + sh), "turned": (False, False)}
            return (r, g, b), fh_roi, motion_delta

        except Exception as exc:
            log.error("Error in Haar ROI extraction: %s", exc)
            return None, None, 0.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_motion_delta(self, cx: float, cy: float) -> float:
        """Euclidean centroid drift between consecutive frames (pixels)."""
        if self._prev_cx is None:
            self._prev_cx, self._prev_cy = cx, cy
            self.is_moving = False
            return 0.0
        delta = ((cx - self._prev_cx) ** 2 + (cy - self._prev_cy) ** 2) ** 0.5
        self.is_moving = delta > 3.0
        self._prev_cx, self._prev_cy = cx, cy
        return delta

    @staticmethod
    def _create_dummy_frame() -> np.ndarray:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "WAITING FOR VIDEO SOURCE", (80, 200), font, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, "Upload a video or start a live session", (60, 250), font, 0.7, (200, 200, 200), 1)
        return frame
