#!/usr/bin/env python3
"""
Build the HR ROI cache: run the (slow) face tracker ONCE per video and save the per-frame colour traces.

Why: ~all of the time in an HR validation run is MediaPipe face tracking (25 s per UBFC-rPPG clip, 5-7 min per
3-minute UBFC-Phys clip). The signal processing that follows is fast. Everything downstream (CHROM/POS,
tracking, ML features, new algorithms) can therefore be re-run from these traces in seconds, deterministically.

The loop is the production one (scripts/analyze_video.py): same Camera, same default get_frame(), same pooled
(r, g, b) per frame. In addition the tracker is asked to record each ROI separately, the same means before the
debug outline is drawn, the mouth opening, and the face geometry.

    python scripts/build_roi_cache.py                 # all sessions, skips those already cached
    python scripts/build_roi_cache.py --only d2_subject1 --only ph_s11
Run ONE instance at a time (two MediaPipe processes made both crawl in the past).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.camera import Camera           # noqa: E402
import hr_data                           # noqa: E402

CACHE_DIR = hr_data.HR_ROOT.parent / "cache" / "roi"
MAX_FRAMES = 100_000


TRACKER = "production"            # "production" = whatever core/camera.py picks (Haar on mediapipe 1.x); "tasks" = landmarks
TASKS_MODEL = hr_data.HR_ROOT.parent / "models" / "face_landmarker.task"


def build_one(s, out_path: Path) -> dict:
    cam = Camera(source=str(s["video"]))
    if cam.dummy_mode:
        raise RuntimeError(f"could not open {s['video']}")
    cam.record_roi_parts = True
    fps = cam.get_source_fps(default=30.0)
    if TRACKER == "tasks":
        # run the camera's ORIGINAL landmark ROI code on the new MediaPipe API (see scripts/tasks_face_mesh.py)
        from tasks_face_mesh import TasksFaceMesh
        cam.face_mesh = TasksFaceMesh(TASKS_MODEL, fps)
        cam.use_mediapipe = True

    valid, pooled, motion, moving = [], [], [], []
    prod, clean, mouth, iod, norm_iod, box, turned = [], [], [], [], [], [], []
    nan3x3, nan3 = np.full((3, 3), np.nan, np.float32), np.full(3, np.nan, np.float32)
    n = 0
    while n < MAX_FRAMES:
        frame_bytes, roi, is_moving, mdelta = cam.get_frame()
        if frame_bytes is None:
            break
        ok = roi is not None
        valid.append(ok)
        pooled.append(np.array(roi, np.float32) if ok else nan3)
        motion.append(float(mdelta))
        moving.append(bool(is_moving))
        parts = cam.last_roi_parts if ok else None
        if parts is not None:
            prod.append(np.array(parts["prod"], np.float32)); clean.append(np.array(parts["clean"], np.float32))
            mouth.append(parts["mouth_open"]); iod.append(parts["iod"]); norm_iod.append(parts["norm_iod"])
            box.append(parts["box"]); turned.append(parts["turned"])
        else:
            prod.append(nan3x3); clean.append(nan3x3); mouth.append(np.nan); iod.append(np.nan)
            norm_iod.append(np.nan); box.append((np.nan,) * 4); turned.append((False, False))
        cam.last_roi_parts = None
        n += 1
    try:
        cam.release_video()
    except Exception:
        pass

    duration = n / fps
    gt_t, gt_hr = hr_data.gt_series(s, duration)
    meta = dict(id=s["id"], dataset=s["dataset"], subject=s["subject"], condition=s["condition"], fps=fps,
                n_frames=n, duration_s=duration, gt_session=hr_data.session_gt(s), use_mediapipe=bool(cam.use_mediapipe),
                video=str(s["video"]))
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp, meta=json.dumps(meta), valid=np.array(valid), rgb=np.array(pooled), motion=np.array(motion, np.float32),
        moving=np.array(moving), parts_prod=np.array(prod), parts_clean=np.array(clean),
        mouth_open=np.array(mouth, np.float32), iod=np.array(iod, np.float32), norm_iod=np.array(norm_iod, np.float32),
        box=np.array(box, np.float32), turned=np.array(turned), gt_t=gt_t, gt_hr=gt_hr,
    )
    os.replace(tmp, out_path)           # only a finished file ever has the final name
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", help="session id substring; repeatable")
    ap.add_argument("--out", default=str(CACHE_DIR))
    ap.add_argument("--limit", type=int, default=0, help="stop after N new sessions (0 = all)")
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th remaining session starting at i")
    ap.add_argument("--tracker", default="production", choices=["production", "tasks"],
                    help="tasks = MediaPipe FaceLandmarker landmarks via scripts/tasks_face_mesh.py (needs dataset/models/face_landmarker.task)")
    args = ap.parse_args()
    global TRACKER
    TRACKER = args.tracker
    if TRACKER == "tasks" and args.out == str(CACHE_DIR):
        args.out = str(CACHE_DIR.parent / "roi_mp")          # never mix trackers in one cache

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = hr_data.discover()
    if args.only:
        sessions = [s for s in sessions if any(o in s["id"] for o in args.only)]
    todo = [s for s in sessions if not (out_dir / f"{s['id']}.npz").exists()]
    si, sn = (int(x) for x in args.shard.split("/"))
    todo = todo[si::sn]
    print(f"{len(sessions)} sessions found, {len(sessions) - len(todo)} already cached, {len(todo)} to do", flush=True)
    done = 0
    for i, s in enumerate(todo, 1):
        t0 = time.time()
        try:
            m = build_one(s, out_dir / f"{s['id']}.npz")
            print(f"[{i}/{len(todo)}] {s['id']:22s} {m['n_frames']:5d} frames  gt {m['gt_session']:.1f}  "
                  f"{time.time() - t0:5.0f}s", flush=True)
            done += 1
        except Exception as exc:
            print(f"[{i}/{len(todo)}] {s['id']}  FAILED: {exc!r}", flush=True)
        if args.limit and done >= args.limit:
            break
    print("finished", flush=True)


if __name__ == "__main__":
    main()
