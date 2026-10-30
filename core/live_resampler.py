"""
Uniform-rate resampler for LIVE camera frames (any real frame rate -> a steady `fs` Hz stream).

Why. AdvancedRPPG turns every frequency into BPM / breaths-per-minute with ONE fixed `fps` number and ignores timestamps.
A webcam driver reports a nominal 30 fps (CAP_PROP_FPS) but the app's real capture loop delivered 9.6-11.5 fps on the dev
machine (low light, USB webcam, Haar + MJPEG work per frame). With the engine assuming 30, every frequency is scaled by
30/10 = 3x: on recorded sessions thinned to ~10 fps the heart rate came out 65 / 138 / 137 / 130 / 55 against truths of
98-111, and respiration never started (its 900-sample buffer needs 30 s at the assumed rate and takes 90 s at the real
one). Resampling the (timestamp, RGB) stream onto a uniform 30 Hz grid fixes both, and keeps the engine, the HR ensemble
and every threshold counted in cycles on exactly the timing they were validated at (see data/evolution/23-*.md).

Works for any input rate: slower than `fs` is linearly interpolated up (the pulse, < 3 Hz, is far below the input
Nyquist as long as the camera gives >= ~8 fps), faster is linearly sampled down. Big gaps (face lost, stalled camera) are
NOT bridged: the grid restarts instead of inventing samples.
"""
from __future__ import annotations

from typing import List, Optional, Tuple


class UniformResampler:
    def __init__(self, fs: float = 30.0, max_gap_s: float = 0.5, max_per_push: int = 90):
        self.fs = float(fs)
        self.dt = 1.0 / self.fs
        self.max_gap_s = float(max_gap_s)
        self.max_per_push = int(max_per_push)
        self._prev: Optional[Tuple[float, Tuple[float, float, float], float]] = None
        self._next_t = 0.0
        self.measured_fps = 0.0                      # EMA of the real input rate (diagnostics only)

    def reset(self) -> None:
        self._prev = None
        self.measured_fps = 0.0

    def push(self, t: float, rgb, motion: float = 0.0) -> List[Tuple[Tuple[float, float, float], float]]:
        """Add one real frame captured at wall-clock time `t` (seconds). Returns the uniform-grid samples that became
        available, as [((r, g, b), motion), ...] — zero, one or several per call depending on the real frame rate."""
        r, g, b = (float(v) for v in rgb)
        cur = (float(t), (r, g, b), float(motion))
        if self._prev is None:
            self._prev, self._next_t = cur, cur[0] + self.dt
            return [((r, g, b), cur[2])]
        t0, c0, m0 = self._prev
        if t <= t0:                                   # clock hiccup / duplicate timestamp: ignore the frame
            return []
        inst = 1.0 / (t - t0)
        self.measured_fps = inst if self.measured_fps == 0.0 else 0.9 * self.measured_fps + 0.1 * inst
        if t - t0 > self.max_gap_s:                   # stall or face lost: restart the grid, don't bridge
            self._prev, self._next_t = cur, cur[0] + self.dt
            return [((r, g, b), cur[2])]
        out = []
        span = t - t0
        while self._next_t <= t and len(out) < self.max_per_push:
            a = (self._next_t - t0) / span
            out.append((tuple(c0[k] + a * (cur[1][k] - c0[k]) for k in range(3)), m0 + a * (cur[2] - m0)))
            self._next_t += self.dt
        if self._next_t <= t:                         # fell far behind (should not happen): jump the grid forward
            self._next_t = t + self.dt
        self._prev = cur
        return out
