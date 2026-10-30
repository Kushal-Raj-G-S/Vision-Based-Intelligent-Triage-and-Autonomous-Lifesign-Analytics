"""
Pulse visualisation — a lightweight, real-time Eulerian-style colour magnification of the face.

This is DISPLAY ONLY. It never feeds the rPPG engine or the clinical engine: `Camera.get_frame()`
runs ROI extraction on the original frame first, then hands a throwaway copy to `PulseViz.apply()`.

Method (after Wu et al. 2012, "Eulerian video magnification"): reduce the face region to a small
fixed grid (spatial pooling lifts the sub-1% colour change above sensor noise), band-pass every
cell over time with two exponential moving averages (a fast one minus a slow one ~ a band around
the heart-rate range), normalise by the running amplitude, then paint positive swings warm and
negative swings cool over the face oval. The viewer sees blood flow pulsing through the skin.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np

GRID = 48                      # pooled grid resolution (cells per side)
LOW_HZ = 0.7                   # slow average cutoff  (~42 bpm)
HIGH_HZ = 3.2                  # fast average cutoff  (~190 bpm)
MAX_ALPHA = 0.9               # strongest blend of the colour map over the face
BOX_PAD = 0.08                 # grow the detected face box slightly
_RMS_DECAY = 0.985             # running amplitude estimate (per frame)


class PulseViz:
    def __init__(self) -> None:
        self._fast: Optional[np.ndarray] = None
        self._slow: Optional[np.ndarray] = None
        self._rms = 0.0
        self._last_t: Optional[float] = None
        self._mask = self._oval_mask(GRID)
        self._prev_y: Optional[np.ndarray] = None
        self._motion = np.zeros((GRID, GRID), np.float32)   # smoothed per-cell motion energy

    @staticmethod
    def _oval_mask(n: int) -> np.ndarray:
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        d = ((xx - (n - 1) / 2) / (n * 0.46)) ** 2 + ((yy - (n - 1) / 2) / (n * 0.50)) ** 2
        return np.clip(1.25 - d, 0.0, 1.0).astype(np.float32)   # soft edge

    def reset(self) -> None:
        self._fast = self._slow = None
        self._rms = 0.0
        self._last_t = None
        self._prev_y = None
        self._motion[:] = 0

    def _artifact_mask(self, cell_bgr: np.ndarray) -> np.ndarray:
        """1 on smooth, still, skin-coloured cells; 0 on eyes, lips, hair edges, background and
        anything moving. Blood-volume change is a uniform tiny colour shift on skin; those other
        cells change far more for reasons unrelated to the pulse and would drown it."""
        u8 = np.clip(cell_bgr, 0, 255).astype(np.uint8)
        ycc = cv2.cvtColor(u8, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
        skin = ((cr > 133) & (cr < 180) & (cb > 77) & (cb < 135)).astype(np.float32)
        skin = cv2.GaussianBlur(skin, (0, 0), 0.8)
        gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.clip(np.sqrt(gx * gx + gy * gy) / 130.0, 0, 1)            # strong edges = features, not skin
        if self._prev_y is not None:
            moved = np.abs(y - self._prev_y)
            self._motion = 0.8 * self._motion + 0.2 * np.clip(moved / 10.0, 0, 1)
        self._prev_y = y
        return np.clip(skin * (1 - edge) * (1 - np.clip(self._motion * 1.2, 0, 1)), 0, 1)

    def apply(self, frame: np.ndarray, face_box: Optional[tuple], target: Optional[np.ndarray] = None) -> np.ndarray:
        """Return a copy of `target` (default `frame`) with the pulse map blended over the face.
        The signal is always read from `frame` (keep it un-annotated); `target` may carry ROI boxes.
        `face_box` is Camera.last_face_box: (x, y, w, h, frame_w, frame_h) or None."""
        out = (frame if target is None else target).copy()
        now = time.time()
        dt = 1 / 30 if self._last_t is None else min(max(now - self._last_t, 1e-3), 0.25)
        self._last_t = now
        if not face_box:
            return out
        x, y, w, h, fw, fh = face_box
        px, py = int(w * BOX_PAD), int(h * BOX_PAD)
        x0, y0 = max(0, int(x) - px), max(0, int(y) - py)
        x1, y1 = min(fw, int(x + w) + px), min(fh, int(y + h) + py)
        if x1 - x0 < 24 or y1 - y0 < 24:
            return out

        roi = frame[y0:y1, x0:x1].astype(np.float32)
        cell = cv2.resize(roi, (GRID, GRID), interpolation=cv2.INTER_AREA)
        # chromatic signal: green share of total light, which is largely illumination-invariant
        sig = cell[..., 1] / (cell.sum(axis=2) + 1e-3)

        a_fast = 1.0 - math.exp(-2 * math.pi * HIGH_HZ * dt)
        a_slow = 1.0 - math.exp(-2 * math.pi * LOW_HZ * dt)
        if self._fast is None:
            self._fast = sig.copy()
            self._slow = sig.copy()
            return out
        self._fast += a_fast * (sig - self._fast)
        self._slow += a_slow * (sig - self._slow)
        band = self._fast - self._slow                       # temporal band-pass, per cell
        keep0 = self._artifact_mask(cell)
        w = self._mask * keep0
        band = band - float((band * w).sum() / (w.sum() + 1e-6))     # drop global flicker

        rms = float(np.sqrt((band * band * w).sum() / (w.sum() + 1e-6)))
        self._rms = max(rms, self._rms * _RMS_DECAY)
        if self._rms < 1e-6:
            return out
        norm = np.clip(band / (0.9 * self._rms), -1.0, 1.0)
        norm = cv2.GaussianBlur(norm, (0, 0), 2.0)

        # colour map: negative -> cool cyan, positive -> hot magenta/red
        pos = np.clip(norm, 0, 1)
        neg = np.clip(-norm, 0, 1)
        colour = np.zeros((GRID, GRID, 3), np.float32)       # BGR
        colour[..., 2] = 255 * pos                           # red
        colour[..., 0] = 90 * pos + 255 * neg                # blue (magenta tint / cyan)
        colour[..., 1] = 255 * neg * 0.85 + 40 * pos         # green (cyan)
        keep = np.sqrt(cv2.GaussianBlur(keep0, (0, 0), 1.0))
        norm = norm * keep
        strength = (np.abs(norm) ** 0.8) * self._mask * MAX_ALPHA

        bw, bh = x1 - x0, y1 - y0
        colour = cv2.resize(colour, (bw, bh), interpolation=cv2.INTER_CUBIC)
        strength = cv2.resize(strength, (bw, bh), interpolation=cv2.INTER_CUBIC)[..., None]
        region = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = np.clip(region * (1 - strength) + colour * strength, 0, 255).astype(np.uint8)
        return out
