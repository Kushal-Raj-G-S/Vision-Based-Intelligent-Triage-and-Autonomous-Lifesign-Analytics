"""
Adapter: MediaPipe Tasks FaceLandmarker exposed through the legacy FaceMesh interface.

Why: the installed mediapipe (1.0.1) removed `mp.solutions`, so core/camera.py has been silently falling back to the
Haar cascade for every video (it logs "using Haar Cascade fallback"). Haar finds the face in only 6-54% of frames on
several of the worst-error sessions. This adapter lets the camera's ORIGINAL landmark-based ROI code
(`Camera._extract_roi_mediapipe`: landmark-anchored forehead/cheeks, IOD sizing, yaw-aware nose fallback) run
unchanged on the new API — research use first (scripts/build_roi_cache.py --tracker tasks).

    fm = TasksFaceMesh(model_path, fps)        # needs face_landmarker.task
    res = fm.process(rgb_frame)                # -> .multi_face_landmarks[0].landmark[i].x/.y (normalised), or None
"""
from types import SimpleNamespace

import numpy as np


class TasksFaceMesh:
    def __init__(self, model_path: str, fps: float):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,       # tracks across frames like the legacy FaceMesh
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._dt_ms = 1000.0 / float(fps or 30.0)
        self._t_ms = 0.0

    def process(self, rgb: np.ndarray):
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._lm.detect_for_video(image, int(self._t_ms))
        self._t_ms += self._dt_ms
        if not res.face_landmarks:
            return SimpleNamespace(multi_face_landmarks=None)
        pts = [SimpleNamespace(x=p.x, y=p.y, z=p.z) for p in res.face_landmarks[0]]
        return SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=pts)])

    def close(self):
        try:
            self._lm.close()
        except Exception:
            pass
