from flask import render_template, Response, jsonify, request, redirect
from werkzeug.utils import secure_filename
from functools import wraps
import statistics as stats
import time
import cv2
import numpy as np
import os
import json
import re
import threading

# Import the web package which holds our shared state and configurations
import web
from core.camera import find_available_cameras
from web.neon_db import (init_db, save_patient_record, get_all_patients, clear_all_patients,
                         next_token, set_patient_status, update_narrative,
                         save_new_patient, get_waiting_patients, warm_pool)

# Initialize the Postgres DB on startup
init_db()
warm_pool()

# Start ranking the AI models at boot, so the first patient never waits for the probe.
from agents import llm_pool as _llm_pool
_llm_pool.start_background_probe()

# Setup decorator shortcuts
app = web.app


def require_api_key(fn):
    """
    Gate a destructive endpoint behind X-API-Key when web.API_KEY is set.
    No-op (endpoint stays open) when API_KEY is unset, so local/dev usage is
    unaffected — see web/__init__.py for the warning this logs on startup.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if web.API_KEY and request.headers.get('X-API-Key') != web.API_KEY:
            return jsonify({'success': False, 'error': 'Unauthorized — missing or invalid X-API-Key header.'}), 401
        return fn(*args, **kwargs)
    return wrapper

def _parse_camera_index(raw_source):
    """Parse optional camera index from request; None means auto-detect."""
    if raw_source is None:
        return None
    if isinstance(raw_source, int):
        return raw_source
    source = str(raw_source).strip()
    if source == '':
        return None
    if source.isdigit() or (source.startswith('-') and source[1:].isdigit()):
        return int(source)
    return None


def _resolve_camera_source(raw_source):
    """Pick a working camera index — explicit choice or first real (non-virtual) device."""
    index = _parse_camera_index(raw_source)
    if index is not None:
        return index
    cam_list = find_available_cameras()
    if not cam_list:
        return None
    # Prefer first non-virtual camera (built-in webcam)
    for cam in cam_list:
        if not cam.get('is_virtual', False):
            return cam['index']
    # Fall back to first available even if virtual
    return cam_list[0]['index']


def release_camera_hardware():
    """Stop streaming and release the video device so the OS unlocks the camera."""
    try:
        web.session.stop_event.set()
        time.sleep(0.1)
        web.session.stop_event.clear()
        with web.session.processing_lock:
            if web.camera is not None:
                web.camera.release_video()
    except Exception as e:
        print(f"[APP] Camera release error: {e}")


def _median_iqr_filtered(values):
    """Same outlier-robust aggregation agents/tools.py::get_vitals_dict and
    scripts/analyze_video.py already use for the clinical/triage numbers —
    duplicated here (not imported) to match this codebase's existing
    pattern of keeping this small helper local to each call site.

    Used so the LIVE dashboard display (web.session.metrics, read by
    /status and generate_report) shows the same validated, session-
    aggregated number the clinical engine actually reasons about, instead
    of a single noisy instantaneous reading — see the live-vs-triage BPM
    discrepancy this was written to fix (dashboard showed 88.4 while the
    triage engine used a validated 50 for the same session)."""
    if not values:
        return None
    sorted_v = sorted(values)
    n = len(sorted_v)
    q1, q3 = sorted_v[n // 4], sorted_v[(3 * n) // 4]
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered = [v for v in sorted_v if lo <= v <= hi] or sorted_v
    return stats.median(filtered)


def _classify(bpm):
    """
    Clinical heart rate classification (resting, adult) aligned with project PDF:
      < 60        → BRADYCARDIA   (slow)
      60 – 100    → NORMAL
      > 100       → TACHYCARDIA   (fast)
    """
    if not isinstance(bpm, (int, float)):
        return '--'
    if bpm <= 0:
        return 'INVALID'
    if bpm < 60:
        return 'BRADYCARDIA'
    if bpm <= 100:
        return 'NORMAL'
    return 'TACHYCARDIA'


def _classify_respiratory_rate(rr):
    """Adult resting respiratory rate classification (breaths/min) aligned with project PDF:
      < 12        → BRADYPNEA
      12 – 20     → NORMAL
      > 20        → TACHYPNEA
    """
    if not isinstance(rr, (int, float)):
        return '--'
    if rr <= 0:
        return 'INVALID'
    if rr < 12:
        return 'BRADYPNEA'
    if rr <= 20:
        return 'NORMAL'
    return 'TACHYPNEA'


def _face_box_payload():
    """Smoothed face box as fractions of the frame (x, y, w, h in 0..1), in the
    camera's RAW (un-mirrored) orientation, or None when no face is visible.
    The kiosk mirrors its preview, so it converts x -> 1 - (x + w)."""
    b = getattr(web.camera, 'last_face_box', None)
    if not b:
        return None
    x, y, w, h, fw, fh = b
    if fw <= 0 or fh <= 0:
        return None
    return {'x': round(x / fw, 3), 'y': round(y / fh, 3),
            'w': round(w / fw, 3), 'h': round(h / fh, 3)}


def generate_frames(clean: bool = False, pulse: bool = False):
    """MJPEG generator. Runs for the lifetime of the /video_feed request.
    clean=True streams frames without the debug face/ROI rectangles (kiosk)."""
    calibration_done = False

    while not web.session.stop_event.is_set():
        frame_bytes, roi_data, is_moving, motion_delta = web.camera.get_frame(annotate=not clean, pulse=pulse)

        # ── Video file ended ─────────────────────────────────────────
        if frame_bytes is None:
            print("[APP] Video ended — computing final summary")
            with web.session.processing_lock:
                final = web.rppg_engine.get_final_summary()
                web.session.metrics.update(final)
                web.session.metrics['status']       = 'VIDEO_ENDED'
                web.session.metrics['classification'] = _classify(final.get('final_bpm'))
                web.session.metrics['calibration_done'] = calibration_done
                if not web.session.is_live_camera:
                    web.camera.release_video()
            break

        with web.session.processing_lock:
            web.session.frame_count += 1
            elapsed = time.time() - web.session.start_time

            # Perceptual luminance (ITU-R BT.601) — accurate pseudo-lux
            if roi_data is not None:
                r_ch, g_ch, b_ch = roi_data
                lux = int(0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch)
            else:
                lux = 0
            web.session.metrics['estimated_lux'] = lux
            web.session.metrics['motion_delta']  = int(motion_delta)
            web.session.metrics['face_box']      = _face_box_payload()
            # False while the camera is showing its idle placeholder (no live camera / video loaded):
            # lets the dashboard cover that placeholder with a proper empty state.
            web.session.metrics['has_source']    = not getattr(web.camera, 'dummy_mode', False)

            # Playback progress for an uploaded video file — display only,
            # never used by the DSP/clinical pipeline. None for a live
            # camera (Camera.get_total_frames() returns None for an int
            # source), so the frontend can hide the overlay entirely there.
            total_frames = web.camera.get_total_frames()
            if total_frames is not None and web.rppg_engine.fps > 0:
                video_current_sec = round(web.session.frame_count / web.rppg_engine.fps, 1)
                video_total_sec   = round(total_frames / web.rppg_engine.fps, 1)
            else:
                video_current_sec = None
                video_total_sec   = None
            web.session.metrics['video_current_sec'] = video_current_sec
            web.session.metrics['video_total_sec']   = video_total_sec

            # Warnings list
            warnings = []
            if is_moving or motion_delta > 15.0:
                warnings.append("Excessive motion — please stay still")
            if roi_data is not None and (lux < 50 or lux > 210):
                warnings.append("Poor lighting — move to better light")

            web.rppg_engine.add_frame(roi_data, elapsed)
            results = web.rppg_engine.process_ppg_signal()

            if results['ready']:
                calibration_done = True
                bpm = results.get('bpm', 0)
                if isinstance(bpm, (int, float)) and bpm > 0:
                    web.session.bpm_history.append(float(bpm))

                # Hybrid ML correction — same gate (sqi > 15) and feature
                # order scripts/analyze_video.py already validated offline.
                # Appends to a session-long list so the eventual triage run
                # can take the same IQR-filtered median over corrected
                # readings that produced the validated 6.48 BPM session MAE,
                # instead of correcting a single noisy instantaneous cycle.
                if web.correction_model is not None and results.get('sqi', 0) > 15:
                    filt_band = results.get('filter_band_hz', [0.0, 0.0])
                    feat_row = [[
                        results['bpm'], results.get('sqi', 0.0), results.get('snr_db', 0.0),
                        results.get('stability_score', 0.0), results.get('rr', 0.0),
                        results.get('rr_confidence', 0.0), results.get('hrv', 0.0),
                        results.get('stress_index', 0.0), filt_band[0], filt_band[1],
                    ]]
                    scaler = web.correction_model.get('scaler')
                    feat_arr = scaler.transform(feat_row) if scaler is not None else feat_row
                    corrected_bpm = float(web.correction_model['model'].predict(feat_arr)[0])
                    if 40.0 <= corrected_bpm <= 180.0:
                        web.session.corrected_bpm_history.append(corrected_bpm)

                # Persist last known-good HRV and stress with EMA smoothing.
                # EMA: new = alpha * raw + (1-alpha) * prev  — damps wild jumps.
                # Stress index (very noisy) → alpha=0.15 (heavy smoothing)
                # HRV → alpha=0.25 (moderate smoothing)
                hrv_now    = results.get('hrv', 0) or 0
                stress_now = results.get('stress_index', 0) or 0
                if hrv_now > 0:
                    if web.session.last_valid_hrv > 0:
                        web.session.last_valid_hrv = 0.25 * hrv_now + 0.75 * web.session.last_valid_hrv
                    else:
                        web.session.last_valid_hrv = hrv_now
                if stress_now > 0:
                    if web.session.last_valid_stress > 0:
                        web.session.last_valid_stress = 0.15 * stress_now + 0.85 * web.session.last_valid_stress
                    else:
                        web.session.last_valid_stress = stress_now


                rr_val = results.get('rr', 0) or 0
                rr_conf = results.get('rr_confidence', 0) or 0
                # Match core/clinical.py's own RR_CONFIDENCE_LOW (30%) reliability
                # floor — that engine already treats anything below it as an
                # unreliable reading in its reasoning text; this gate used to be a
                # much looser 10%, so a borderline noisy edge-lock (respiration's
                # RSA modulation is weak in webcam PPG; a bad read tends to land
                # right at the 6 or 30 br/min search-band edge — see
                # AdvancedRPPG._estimate_rr's docstring) could slip past here and
                # display as a solid-looking number the clinical engine had
                # already privately flagged as untrustworthy.
                if rr_conf < 30:
                    rr_val = 0

                # Display values — the SAME session-aggregated, outlier-
                # filtered numbers agents/tools.py::get_vitals_dict feeds to
                # the clinical/triage engine, not the single instantaneous
                # reading above. Before this, web.session.metrics['bpm']/
                # ['rr'] carried the raw per-frame value, so the live gauge
                # and the generated report (both read metrics here) could
                # show a wildly different number than the ESI/triage result
                # computed moments later from the same session — observed
                # live as an 88.4 BPM dashboard card next to a 50 BPM triage
                # figure for the identical session. bpm_history already
                # accumulates the raw per-frame values above (line ~200), so
                # this only changes what's DISPLAYED, not what's recorded.
                bpm_hist = web.session.bpm_history
                avg_bpm_disp = (
                    stats.mean(bpm_hist) if len(bpm_hist) >= 2
                    else (bpm_hist[-1] if bpm_hist else bpm)
                )
                corrected_bpm_disp = (
                    _median_iqr_filtered(web.session.corrected_bpm_history)
                    if len(web.session.corrected_bpm_history) >= 3 else None
                )
                bpm_display = corrected_bpm_disp if corrected_bpm_disp is not None else avg_bpm_disp

                rr_hist_live = list(getattr(web.rppg_engine, 'rr_history', []) or [])
                # >=1, not >=3 — every entry in rr_history already
                # individually cleared _RR_HISTORY_MIN_CONF (30%) to get
                # there (see core/rppg.py), so showing the median of even
                # one or two of them is not showing anything the clinical
                # engine considers unreliable. The >=3 floor stays enforced
                # separately in agents/tools.py::get_vitals_dict() for the
                # actual Compensated Shock determination, which legitimately
                # wants more session-level evidence before acting on it —
                # this only changes what the LIVE gauge displays while the
                # session is still running. Without this, the gauge sat on
                # "--" for nearly the whole session, since RR often only
                # crosses the >=3 mark in the last few seconds (Finding 10).
                rr_session_disp = (
                    _median_iqr_filtered(rr_hist_live) if len(rr_hist_live) >= 1 else None
                )
                rr_display = rr_session_disp if rr_session_disp is not None else rr_val

                web.session.metrics = {
                    'bpm':                  round(bpm_display, 1) if isinstance(bpm_display, (int, float)) else 0,
                    'bpm_instant':          int(bpm) if isinstance(bpm, (int, float)) else 0,
                    'confidence':           int(results.get('confidence', 0)),
                    'status':               results.get('status', 'OK'),
                    'snr_db':               results.get('snr_db', 0),
                    'sqi':                  results.get('sqi', 0),
                    'classification':       _classify(bpm_display),
                    'ohi':                  results.get('confidence', 0),
                    'stability':            results.get('stability_score', 0),
                    'stability_indicator':  results.get('stability_indicator', '--'),
                    'rr':                   round(rr_display, 1) if rr_display else 0,
                    'rr_instant':           rr_val,
                    'rr_confidence':        rr_conf,
                    'rr_classification':    _classify_respiratory_rate(rr_display),
                    'hrv':                  web.session.last_valid_hrv,
                    'stress_index':         web.session.last_valid_stress,
                    'warnings':             warnings,
                    'remark':               results.get('remark', ''),
                    'estimated_lux':        lux,
                    'motion_delta':         int(motion_delta),
                    'is_live':              web.session.is_live_camera,
                    'calibration_done':     True,
                    'face_detected':        roi_data is not None,
                    'face_box':             _face_box_payload(),
                    'has_source':           not getattr(web.camera, 'dummy_mode', False),
                    'ppg_signal':           results.get('ppg_signal', [])[-150:],
                    'calibration_progress': 100,
                    'video_current_sec':    video_current_sec,
                    'video_total_sec':      video_total_sec,
                }
            else:
                web.session.metrics['status']           = 'CALIBRATING'
                web.session.metrics['warnings']         = warnings
                web.session.metrics['is_live']          = web.session.is_live_camera
                web.session.metrics['calibration_done'] = False
                web.session.metrics['estimated_lux']    = lux
                web.session.metrics['motion_delta']     = int(motion_delta)
                web.session.metrics['face_detected']    = roi_data is not None
                web.session.metrics['calibration_progress'] = results.get('calibration_progress', 0)
                web.session.metrics['rr']               = results.get('rr', 0)
                web.session.metrics['rr_confidence']    = results.get('rr_confidence', 0)
                web.session.metrics['rr_classification'] = _classify_respiratory_rate(results.get('rr', 0))

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.02)


# ── App Routes ───────────────────────────────────────────────────────

@app.route('/')
def index():
    # Redirect browser visits to the Next.js SaaS frontend on port 3000
    # The Flask server on port 5002 is purely the REST API backend
    return jsonify({
        'service': 'VITAL Opti-Screen API Server',
        'version': '2.0',
        'status': 'running',
        'frontend': 'http://localhost:3000',
        'message': 'Open http://localhost:3000 for the VITAL SaaS dashboard.'
    })


@app.route('/video_feed')
def video_feed():
    clean = request.args.get('clean', '') in ('1', 'true', 'yes')
    pulse = request.args.get('pulse', '') in ('1', 'true', 'yes')
    return Response(generate_frames(clean=clean, pulse=pulse),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    with web.session.processing_lock:
        payload = dict(web.session.metrics)
    return jsonify(payload)


@app.route('/session_summary')
def session_summary():
    """Returns final session statistics built from bpm_history."""
    with web.session.processing_lock:
        history = list(web.session.bpm_history)
        metrics = dict(web.session.metrics)

    if web.session.image_mode and metrics.get('status') == 'IMAGE_READY':
        return jsonify({
            'avg_bpm':        round(metrics.get('bpm', 0), 1),
            'min_bpm':        round(metrics.get('bpm', 0), 1),
            'max_bpm':        round(metrics.get('bpm', 0), 1),
            'stability_pct':  0,
            'rr':             round(metrics.get('rr', 0), 1),
            'rr_classification': metrics.get('rr_classification', '--'),
            'hrv':            round(metrics.get('hrv', 0), 1),
            'stress_index':   round(metrics.get('stress_index', 0), 1),
            'classification': metrics.get('classification', '--'),
            'remark':         metrics.get('remark', ''),
            'sample_count':   0,
        })

    if len(history) >= 2:
        sorted_h = sorted(history)
        n = len(sorted_h)
        q1 = sorted_h[n // 4]
        q3 = sorted_h[(3 * n) // 4]
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        clean = [x for x in sorted_h if lo <= x <= hi] or sorted_h
        avg_bpm = round(stats.mean(clean), 1)
        min_bpm = round(min(clean), 1)
        max_bpm = round(max(clean), 1)
        std_bpm = stats.stdev(clean) if len(clean) > 1 else 0
        stability_pct = round(max(0, min(100, 100 - std_bpm * 4)))
    elif len(history) == 1:
        avg_bpm = min_bpm = max_bpm = round(history[0], 1)
        stability_pct = 0
    else:
        # INSUFFICIENT_DATA — no fake values
        return jsonify({
            'avg_bpm': None, 'min_bpm': None, 'max_bpm': None,
            'stability_pct': 0,
            'rr': round(metrics.get('rr', 0), 1),
            'rr_classification': metrics.get('rr_classification', '--'),
            'hrv': round(metrics.get('hrv', 0), 1),
            'stress_index': round(metrics.get('stress_index', 0), 1),
            'classification': '--',
            'remark': 'INSUFFICIENT_DATA',
            'sample_count': 0,
        })

    payload = {
        'avg_bpm':        avg_bpm,
        'min_bpm':        min_bpm,
        'max_bpm':        max_bpm,
        'stability_pct':  stability_pct,
        'rr':             round(metrics.get('rr', 0), 1),
        'rr_classification': metrics.get('rr_classification', '--'),
        'hrv':            round(metrics.get('hrv', 0), 1),
        'stress_index':   round(metrics.get('stress_index', 0), 1),
        'classification': _classify(avg_bpm),
        'remark':         metrics.get('remark', ''),
        'sample_count':   len(history),
    }

    return jsonify(payload)


def _reset(source, live):
    web.session.stop_event.set()
    time.sleep(0.15)
    web.session.stop_event.clear()
    with web.session.processing_lock:
        if web.camera is not None:
            web.camera.release_video()
        from core.camera import Camera
        from core.rppg import AdvancedRPPG
        web.camera = Camera(source=source)
        # Use the source's ACTUAL frame rate, not a hardcoded 30 — an
        # uploaded video not encoded at ~30fps (confirmed on several real
        # UBFC-rPPG subjects, ~23.2fps) previously produced a confident but
        # deterministically wrong BPM, scaled by (30 / real_fps). Live
        # webcams usually do negotiate close to 30fps, but there's no
        # reason to assume it when we can just ask the device.
        source_fps = web.camera.get_source_fps(default=30.0)
        web.rppg_engine = AdvancedRPPG(fps=int(round(source_fps)), window_size=max(300, int(source_fps * 10)))
        web.session.frame_count    = 0
        web.session.start_time     = time.time()
        web.session.is_live_camera = live
        web.session.image_mode      = False
        web.session.image_frame_bytes = None
        web.session.bpm_history    = []
        web.session.corrected_bpm_history = []
        web.session.last_valid_hrv    = 0.0
        web.session.last_valid_stress = 0.0
        web.session.metrics.update({
            'bpm': 0, 'status': 'WAITING', 'classification': 'UNKNOWN',
            'rr': 0, 'rr_confidence': 0, 'rr_classification': '--',
            'hrv': 0, 'stress_index': 0, 'warnings': [],
            'estimated_lux': 0, 'motion_delta': 0,
            'is_live': live, 'calibration_done': False,
        })


def _release_session():
    """End active capture and return camera to standby without destroying detectors."""
    web.session.is_live_camera = False
    web.session.stop_event.set()
    time.sleep(0.15)
    web.session.stop_event.clear()
    with web.session.processing_lock:
        web.session.is_live_camera = False
        if web.camera is not None:
            web.camera.release_video()
        web.session.metrics.update({
            'status': 'WAITING',
            'is_live': False,
            'calibration_done': False,
        })


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in web.ALLOWED_EXTENSIONS


def _is_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in web.IMAGE_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    filename  = secure_filename(file.filename)
    # Upload folder relative to package root
    filepath  = os.path.join(app.root_path, '..', web.UPLOAD_FOLDER, filename)
    file.save(filepath)

    if _is_image_file(filename):
        web.session.stop_event.set()
        time.sleep(0.15)
        web.session.stop_event.clear()
        if web.camera is not None:
            web.camera.release_video()
        web.session.is_live_camera = False
        analysis = web.camera.analyze_image_file(filepath)
        if analysis is None:
            return jsonify({'error': 'Invalid image file'}), 400

        with web.session.processing_lock:
            web.session.image_frame_bytes = analysis['frame_bytes']
            web.session.image_mode = True
            web.session.bpm_history.clear()
            web.session.metrics.update({
                'bpm': 72,
                'confidence': 15,
                'status': 'IMAGE_READY',
                'classification': 'NORMAL',
                'rr': 16,
                'rr_confidence': 15,
                'rr_classification': _classify_respiratory_rate(16),
                'hrv': 35,
                'stress_index': 40,
                'warnings': ['Image demo estimate — use live/video for real vitals'],
                'remark': 'IMAGE_DEMO',
                'estimated_lux': analysis['estimated_lux'],
                'motion_delta': 0,
                'is_live': False,
                'calibration_done': True,
                'face_detected': analysis['face_detected'],
                'ppg_signal': [],
                'calibration_progress': 100,
            })

        return jsonify({
            'success': True,
            'message': f'Image uploaded: {filename}',
            'mode': 'image',
        })

    # Store only — do NOT start playback here. Uploading used to immediately
    # call _reset() and start streaming the video, which meant there was no
    # way to upload several videos ahead of time and choose which one to run
    # later; every new upload silently pre-empted whatever had just started.
    # Playback now starts explicitly via /api/start_uploaded_video, selected
    # from /api/uploaded_videos' listing (see the "Uploaded Video" device
    # mode in the frontend).
    return jsonify({'success': True, 'message': f'Video saved: {filename}', 'mode': 'video', 'filename': filename})


@app.route('/api/uploaded_videos')
def list_uploaded_videos():
    """List previously uploaded video files (not images) available to play."""
    upload_dir = os.path.join(app.root_path, '..', web.UPLOAD_FOLDER)
    videos = []
    try:
        for fname in sorted(os.listdir(upload_dir)):
            full_path = os.path.join(upload_dir, fname)
            if not os.path.isfile(full_path):
                continue
            if allowed_file(fname) and not _is_image_file(fname):
                videos.append({
                    'filename': fname,
                    'size_bytes': os.path.getsize(full_path),
                    'modified': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(full_path))),
                })
    except FileNotFoundError:
        pass
    # Most recently uploaded first.
    videos.sort(key=lambda v: v['modified'], reverse=True)
    return jsonify({'videos': videos})


@app.route('/api/start_uploaded_video', methods=['POST'])
def start_uploaded_video():
    """Start playing a specific, already-uploaded video by filename."""
    data = request.get_json(silent=True) or {}
    filename = secure_filename((data.get('filename') or '').strip())
    if not filename:
        return jsonify({'success': False, 'error': 'No filename provided.'}), 400

    upload_dir = os.path.join(app.root_path, '..', web.UPLOAD_FOLDER)
    filepath = os.path.join(upload_dir, filename)
    # secure_filename() already strips path-traversal components, but
    # resolve-and-contain is the actual guarantee against escaping the
    # upload directory, not just filename sanitization.
    if os.path.commonpath([os.path.abspath(filepath), os.path.abspath(upload_dir)]) != os.path.abspath(upload_dir):
        return jsonify({'success': False, 'error': 'Invalid filename.'}), 400
    if not os.path.isfile(filepath):
        return jsonify({'success': False, 'error': f'No uploaded video named "{filename}" found.'}), 404

    _reset(filepath, live=False)
    return jsonify({'success': True, 'message': f'Playing: {filename}'})


@app.route('/api/cameras')
def list_cameras():
    """Return indices and names of locally available camera devices."""
    cam_list = find_available_cameras()  # list of {index, label, is_virtual}
    # Default = first non-virtual camera
    default_cam = next((c for c in cam_list if not c.get('is_virtual', False)), None)
    default_index = default_cam['index'] if default_cam else (cam_list[0]['index'] if cam_list else None)
    return jsonify({
        'cameras': cam_list,
        'default': default_index,
    })


@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    try:
        payload = request.get_json(silent=True) or {}
        source = _resolve_camera_source(payload.get('source'))
        if source is None:
            return jsonify({
                'success': False,
                'error': 'No camera detected. Connect a webcam and ensure it is not in use by another app.',
            }), 400
        _reset(source, live=True)
        if web.camera.dummy_mode:
            return jsonify({
                'success': False,
                'error': f'Could not open camera index {source}. Try another device from the list.',
            }), 400
        return jsonify({
            'success': True,
            'message': f'Live camera started (index {source})',
            'source': source,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/reset_camera', methods=['POST'])
def reset_camera():
    _release_session()
    return jsonify({'success': True, 'message': 'Camera released'})


@app.route('/api/scan/begin', methods=['POST'])
def scan_begin():
    """Start a CLEAN measurement window without touching the camera: new rPPG
    engine + cleared histories. The kiosk uses this after positioning, so frames
    captured while the patient was still moving into place never reach the
    session medians. (_reset() also rebuilds the camera, which is slower and
    drops the video stream.)"""
    from core.rppg import AdvancedRPPG
    if web.camera is None or web.camera.dummy_mode:
        return jsonify({'success': False, 'error': 'Camera is not running.'}), 409
    with web.session.processing_lock:
        fps = web.camera.get_source_fps(default=30.0)
        web.rppg_engine = AdvancedRPPG(fps=int(round(fps)), window_size=max(300, int(fps * 10)))
        web.session.frame_count = 0
        web.session.start_time = time.time()
        web.session.bpm_history = []
        web.session.corrected_bpm_history = []
        web.session.last_valid_hrv = 0.0
        web.session.last_valid_stress = 0.0
        web.session.metrics.update({
            'bpm': 0, 'status': 'WAITING', 'classification': 'UNKNOWN',
            'rr': 0, 'rr_confidence': 0, 'rr_classification': '--',
            'hrv': 0, 'stress_index': 0, 'warnings': [], 'calibration_done': False,
        })
    return jsonify({'success': True})


STT_URL = os.getenv('STT_URL', 'http://127.0.0.1:5003')


@app.route('/api/stt', methods=['POST'])
def stt_proxy():
    """Forward a short WAV clip to the local GPU Whisper server (tools/gpu_lab/stt_server.py).
    503 tells the kiosk to fall back to typing — it must never dead-end a patient."""
    import requests as _rq
    audio = request.files.get('audio')
    if audio is None:
        return jsonify({'success': False, 'error': 'no audio'}), 400
    try:
        r = _rq.post(f"{STT_URL}/transcribe",
                     files={'audio': ('clip.wav', audio.read(), 'audio/wav')},
                     data={'hint': request.form.get('hint', '')}, timeout=30)
        return jsonify(r.json()), r.status_code
    except Exception as exc:
        return jsonify({'success': False, 'error': 'stt_unavailable', 'detail': str(exc)[:120]}), 503


@app.route('/release_camera', methods=['POST'])
def release_camera():
    _release_session()
    return jsonify({'success': True, 'message': 'Camera released'})


@app.route('/image_feed')
def image_feed():
    if not web.session.image_mode or web.session.image_frame_bytes is None:
        return jsonify({'error': 'No image loaded'}), 404
    return Response(web.session.image_frame_bytes, mimetype='image/jpeg')


@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.json or {}
    user_message = data.get('message', '')
    history = data.get('history', [])
    image_data = data.get('image', None)  # Base64 data: "data:image/jpeg;base64,..."
    
    # Check if NVIDIA NIM key is present
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    
    # Retrieve current active patient records from Neon. get_all_patients()
    # now raises on a real DB failure instead of silently returning [] (that
    # used to make transient Neon connection hiccups indistinguishable from
    # "no patients exist") — catch it here specifically so ARIA degrades to
    # "database unreachable" rather than the chat endpoint 500ing outright.
    try:
        patients = get_all_patients()
        db_unreachable = False
    except Exception as db_exc:
        print(f"[CHAT] Patient DB unreachable: {db_exc}")
        patients = []
        db_unreachable = True

    patient_context = "\n\n=== CURRENT ACTIVE TRIAGE QUEUE & PATIENT RECORDS ===\n"
    if db_unreachable:
        patient_context += (
            "Patient database is temporarily unreachable (transient connection error) — "
            "do not tell the user their records were deleted; this is a connectivity issue, not data loss.\n"
        )
    elif not patients:
        patient_context += "No patient records currently exist in the triage database.\n"
    else:
        for p in patients:
            patient_context += (
                f"- Patient Name: {p.get('name')}\n"
                f"  Record ID: {p.get('id')}\n"
                f"  Triage Timestamp: {p.get('timestamp')}\n"
                f"  Acuity Rating: ESI Level {p.get('esi_level')} (Priority Score: {p.get('priority_score')}/100)\n"
                f"  Vitals Telemetry: HR {p.get('heart_rate', '--')} BPM, Respiration {p.get('respiration', '--')} breaths/min, "
                f"HRV {p.get('hrv', '--')} ms, Stress Index {p.get('stress_index', '--')}\n"
                f"  Primary Diagnosis: {p.get('primary_diagnosis', 'N/A')}\n"
                f"  Shock Alert Criteria: {'CRITICAL / ACTIVE SHOCK' if p.get('is_shock') else 'NORMAL / NO SHOCK'}\n"
                f"  AI Clinical Assessment Trace: {p.get('triage_summary', 'N/A')}\n\n"
            )
            
    try:
        if nvidia_key:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
            
            # Setup messages list with system instructions and patient data injected
            instructions = (
                web.chatbot_sys_instruct + "\n\n"
                "CLINICAL DATABASE PRIVILEGES:\n"
                "You have read-only query access to the active patient database. Use the records listed below "
                "to answer questions from triage staff or patients about specific individuals. When asked about a "
                "patient by name, match it against the records, summarize their ESI score, vital statistics, "
                "and explain their triage significance. Remain highly professional and outline any warnings (e.g. shock risk).\n"
                + patient_context
            )
            messages = [SystemMessage(content=instructions)]
            
            # Append historical messages
            for msg in history[-6:]:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))
            
            # Append new user message with optional image part for multimodal OCR
            if image_data:
                header, encoded = image_data.split(",", 1) if "," in image_data else ("", image_data)
                payload = [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
                ]
                messages.append(HumanMessage(content=payload))
            else:
                messages.append(HumanMessage(content=user_message))
                
            # Model choice comes from the self-healing pool (agents/llm_pool.py): NVIDIA's
            # hosted models flip between working / 404 / 410 / timing out within hours,
            # so the chat walks the currently-live ranking instead of a fixed list.
            # timeout=30 keeps one stalled model from freezing the chat for a minute.
            from agents import llm_pool
            llm_pool.wait_ready(30)           # right after startup, use the real ranking
            live = llm_pool.ranked()[:4] or [m for m, _ in llm_pool.CANDIDATES[:4]]
            response_text, last_err = None, None
            for m in live:
                try:
                    reply = ChatNVIDIA(model=m, nvidia_api_key=nvidia_key,
                                       max_completion_tokens=2000, timeout=30).invoke(messages)
                    response_text = reply.content
                    break
                except Exception as exc:          # record the failure so the NEXT request skips this model
                    last_err = exc
                    if llm_pool.is_dead_error(exc):
                        llm_pool.mark_dead(m)
                    print(f"[CHAT] {m} failed ({type(exc).__name__}: {str(exc)[:100]}) — trying next.")
            if response_text is None:
                raise last_err if last_err else RuntimeError("no model available")
            
        else:
            # Fallback simulator with database parsing
            matched_patient = None
            for p in patients:
                p_name = str(p.get('name', '')).lower()
                if p_name and p_name in user_message.lower():
                    matched_patient = p
                    break
            
            if matched_patient:
                response_text = (
                    f"[en-US] [DEMO MODE: DATABASE DETECTED] I found a matching record for patient **{matched_patient['name']}**.\n\n"
                    f"* **Triage Acuity:** ESI Level {matched_patient['esi_level']} (Score: {matched_patient['priority_score']}/100)\n"
                    f"* **Vitals:** HR {matched_patient.get('heart_rate', '--')} BPM, Respiration {matched_patient.get('respiration', '--')} breaths/min, HRV {matched_patient.get('hrv', '--')} ms, Stress Index {matched_patient.get('stress_index', '--')}\n"
                    f"* **Clinical Indication:** {matched_patient.get('primary_diagnosis', 'N/A')}\n"
                    f"* **AI Assessment:** {matched_patient.get('triage_summary', 'N/A')}\n\n"
                    f"To enable complete clinical reasoning on this record, please configure your NVIDIA_API_KEY."
                )
            else:
                if image_data:
                    response_text = (
                        "[en-US] [DEMO MODE: NVIDIA_API_KEY not set] I have successfully parsed your clinical document image. "
                        "The text indicates a patient intake chart with blood pressure 125/80 mmHg, resting heart rate 72 bpm, "
                        "and a chief complaint of mild chest tightness. These vitals map to ESI Level 3. "
                        "Please proceed with live sensor calibration or video upload for precise physiological analysis."
                    )
                else:
                    response_text = (
                        f"[en-US] [DEMO MODE: NVIDIA_API_KEY not set] Thank you for your clinical query. You asked: '{user_message}'. "
                        "ARIA is currently in fallback mode. Please configure NVIDIA_API_KEY in your .env to enable the full "
                        "NVIDIA Llama Nemotron reasoning engine."
                    )
                
        return jsonify({'response': response_text})
        
    except Exception as e:
        import traceback
        print("\n" + "!"*60)
        print(f"[CHAT ERROR] ARIA chatbot query failed: {str(e)}")
        traceback.print_exc()
        print("!"*60 + "\n")
        return jsonify({'error': f"ARIA query failed: {str(e)}"}), 500


@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    """Generate a clinical session summary report from current rPPG metrics."""
    try:
        with web.session.processing_lock:
            metrics = dict(web.session.metrics)
            history = list(web.session.bpm_history)

        bpm = metrics.get('bpm', 0)
        rr  = metrics.get('rr', 0)
        hrv = metrics.get('hrv', web.session.last_valid_hrv or 0)
        stress = metrics.get('stress_index', web.session.last_valid_stress or 0)
        classification = metrics.get('classification', 'UNKNOWN')
        rr_classification = metrics.get('rr_classification', '--')
        confidence = metrics.get('confidence', 0)
        snr_db = metrics.get('snr_db', 0)
        status = metrics.get('status', 'UNKNOWN')
        lux = metrics.get('estimated_lux', 0)
        stability = metrics.get('stability_indicator', '--')

        # Session BPM statistics
        if len(history) >= 2:
            import statistics as _stats
            avg_bpm = round(_stats.mean(history), 1)
            min_bpm = round(min(history), 1)
            max_bpm = round(max(history), 1)
        elif bpm > 0:
            avg_bpm = min_bpm = max_bpm = round(float(bpm), 1)
        else:
            avg_bpm = min_bpm = max_bpm = None

        # Clinical interpretation
        if classification == 'NORMAL' and (rr_classification in ('NORMAL', '--')):
            clinical_summary = (
                f"Vitals within normal physiological range. "
                f"Heart rate {bpm} BPM classified as {classification}. "
                f"Respiratory rate {round(float(rr), 1) if rr else '--'} br/min. "
                f"HRV (RMSSD) {round(float(hrv), 1) if hrv else '--'} ms indicates "
                f"{'adequate autonomic balance' if hrv and float(hrv) > 20 else 'reduced autonomic variability'}."
            )
        elif classification == 'TACHYCARDIA':
            clinical_summary = (
                f"Elevated heart rate detected ({bpm} BPM — TACHYCARDIA). "
                f"Requires clinical assessment. Rule out fever, anxiety, dehydration, or cardiac arrhythmia. "
                f"HRV: {round(float(hrv), 1) if hrv else '--'} ms."
            )
        elif classification == 'BRADYCARDIA':
            clinical_summary = (
                f"Low heart rate detected ({bpm} BPM — BRADYCARDIA). "
                f"May indicate high athletic conditioning or AV conduction block. "
                f"Clinical evaluation advised. HRV: {round(float(hrv), 1) if hrv else '--'} ms."
            )
        else:
            clinical_summary = (
                f"Session vitals recorded. Signal confidence {confidence}%. "
                f"Further calibration or improved lighting may be needed for clinical-grade readings."
            )

        stress_label = (
            'CRITICAL' if float(stress or 0) > 150 else
            'ELEVATED' if float(stress or 0) > 80 else
            'OPTIMAL' if float(stress or 0) > 0 else 'UNKNOWN'
        )

        report = {
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'session_duration_s': 30,
            'vitals': {
                'heart_rate_bpm': bpm,
                'heart_rate_avg': avg_bpm,
                'heart_rate_min': min_bpm,
                'heart_rate_max': max_bpm,
                'classification': classification,
                'respiratory_rate': round(float(rr), 1) if rr else None,
                'rr_classification': rr_classification,
                'hrv_rmssd_ms': round(float(hrv), 1) if hrv else None,
                'stress_index': round(float(stress), 1) if stress else None,
                'stress_label': stress_label,
            },
            'signal_quality': {
                'confidence_pct': confidence,
                'snr_db': round(float(snr_db), 1) if snr_db else None,
                'stability': stability,
                'luminance_lux': lux,
                'status': status,
            },
            'clinical_summary': clinical_summary,
            'disclaimer': (
                'This report is generated by rPPG optical analysis and is intended as a screening aid only. '
                'It is NOT a substitute for clinical examination or certified medical diagnostics.'
            ),
        }

        web.session.last_compiled_report = report
        return jsonify({'success': True, 'report': report})

    except Exception as e:
        return jsonify({'success': False, 'error': f'Report generation failed: {str(e)}'}), 500


def _extract_json_blob(text):
    """Best-effort JSON extraction from LLM free text (plain, fenced, or embedded)."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    return None


def _find_key_anywhere(obj, key):
    """Depth-first search for `key` in a possibly-nested dict (the 8B model
    sometimes wraps fields in a 'triage_recommendation' sub-object instead of
    the flat schema asked for — search rather than assume one shape)."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key_anywhere(v, key)
            if found is not None:
                return found
    return None


def _extract_triage_summary(result_text, assessment):
    """
    Pull a clean, human-readable summary out of the Coordinator Agent's free
    text for dashboard display. Previously this field was set to the raw LLM
    output (a fenced ```json {...}``` blob), which rendered as a literal JSON
    dump in the UI — a real display bug, not intentional. Falls back to the
    deterministic assessment's own reasoning if nothing parses.
    """
    parsed = _extract_json_blob(result_text)
    if parsed is not None:
        summary = _find_key_anywhere(parsed, 'triage_summary')
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
    # Nothing usable parsed out of the LLM text — fall back to the
    # deterministic engine's own reasoning rather than showing raw JSON.
    return f"{assessment.primary_diagnosis}. " + "; ".join(assessment.reasoning)


@app.route('/api/triage_run', methods=['POST'])
def triage_run_api():
    """
    Runs the deterministic clinical engine (core.clinical) on this patient's
    vitals, then kicks off the CrewAI 3-Agent pipeline to narrate/explain
    that assessment against retrieved guideline context.

    The ESI level / priority score / shock flag saved to the record are
    ALWAYS the deterministic engine's output — never parsed from LLM free
    text — so a malformed or hallucinated LLM response can degrade the
    narrative quality but can never change a patient's triage priority.
    """
    patient_name = None
    intake = {}
    fast = False
    # Support JSON POST request for live session
    if request.is_json:
        data = request.get_json(silent=True) or {}
        source = data.get('source', '')
        patient_name = data.get('patient_name', '')
        intake = _clean_intake(data)
        # fast=True (kiosk): save the record + token right after the quick
        # deterministic assessment and write the LLM narrative in later.
        fast = bool(data.get('fast'))
        if source.lower() in ['live', 'webcam']:
            filepath = 'live'
        else:
            return jsonify({'error': 'Invalid source in JSON body. Expected "live" or "webcam".'}), 400
    else:
        patient_name = request.form.get('patient_name', '')
        if 'video' not in request.files:
            return jsonify({'error': 'No video file or JSON source parameter provided'}), 400
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'Invalid file'}), 400

        # Support sending "live" as filename
        if file.filename.lower() in ['live', 'webcam']:
            filepath = 'live'
        else:
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.root_path, '..', web.UPLOAD_FOLDER, filename)
            file.save(filepath)

    # ── Step 1: deterministic vitals + clinical assessment (always runs,
    #    regardless of whether the LLM crew below succeeds) ──────────────
    from agents.tools import get_vitals_dict
    from core import clinical, guidelines

    vitals = get_vitals_dict(filepath)
    assessment = clinical.assess(
        hr=float(vitals.get('bpm', 0) or 0),
        rr=float(vitals.get('rr', 0) or 0),
        hrv=float(vitals.get('hrv', 0) or 0),
        stress_index=float(vitals.get('stress_index', 0) or 0),
        sqi=float(vitals.get('sqi', vitals.get('confidence', 0)) or 0),
        rr_confidence=float(vitals.get('rr_confidence', 0) or 0),
        age_years=intake.get('age'),          # kiosk intake; None (adult limits) from the dashboard
    )
    grounding = guidelines.grounding_context(assessment)

    # ── Step 2: LLM crew narrates/explains the deterministic assessment ──
    def _narrate():
        """Run the 3-agent crew. Returns (result_text, crew_failed_reason, triage_summary).
        The deterministic assessment above is already complete and valid — an
        LLM/API outage degrades the narrative, it never takes down triage."""
        text, failed = "", None
        try:
            from agents.crew import build_triage_crew
            crew_result = build_triage_crew().kickoff(inputs={
                "video_path": filepath,
                "precomputed_vitals": str(vitals),
                "clinical_assessment": assessment.narrative_context(),
                "grounding_context": grounding,
            })
            text = str(crew_result)
        except Exception as e:
            failed = str(e)
            text = ("[AI narrative unavailable — LLM crew execution failed: "
                    f"{failed}]\n\n" + assessment.narrative_context())
        summary = (
            _extract_triage_summary(text, assessment) if failed is None else
            f"{assessment.primary_diagnosis}. " + "; ".join(assessment.reasoning)
        )
        return text, failed, summary

    if fast:
        crew_failed_reason = None
        result_text = "[AI narrative is being generated — this record is already triaged and queued.]"
        triage_summary = f"{assessment.primary_diagnosis}. " + "; ".join(assessment.reasoning)
    else:
        result_text, crew_failed_reason, triage_summary = _narrate()

    # Generate patient details
    import uuid
    patient_id = str(uuid.uuid4())[:8]
    if not patient_name or str(patient_name).strip() == '':
        patient_name = "Patient " + patient_id
        if filepath != 'live':
            base_name = os.path.basename(filepath)
            name_part = os.path.splitext(base_name)[0]
            patient_name = name_part.replace('_', ' ').replace('-', ' ').title()

    patient_record = {
        'id': patient_id,
        'name': str(patient_name).strip(),
        'video_path': filepath,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        # Authoritative fields — always from the deterministic engine, never
        # from LLM-parsed JSON.
        'esi_level': assessment.esi_level,
        'priority_score': assessment.priority_score,
        'primary_diagnosis': assessment.primary_diagnosis,
        'is_shock': assessment.is_shock,
        'triage_summary': triage_summary,
        'agent_output': result_text,
        'heart_rate': float(vitals.get('bpm', 0) or 0),
        'respiration': float(vitals.get('rr', 0) or 0),
        'hrv': float(vitals.get('hrv', 0) or 0),
        'stress_index': float(vitals.get('stress_index', 0) or 0),
        **intake,
    }

    # Token allocation and the save happen together in ONE transaction (one pooled
    # connection) so the fast kiosk path stays quick and two patients finishing
    # at the same moment can never receive the same token. save_patient_record()
    # swallows DB errors and returns False; this does the same, so the response
    # still says plainly when the record was NOT persisted.
    try:
        db_saved = bool(save_new_patient(patient_record))
    except Exception as exc:
        print(f"[DB] ERROR saving record {patient_record.get('id')}: {exc}")
        db_saved = False

    if fast and db_saved:
        def _fill_narrative(pid=patient_id):
            text, failed, summary = _narrate()
            try:
                update_narrative(pid, summary, text)
            except Exception as exc:
                print(f"[NARRATIVE] could not update record {pid}: {exc}")
        threading.Thread(target=_fill_narrative, daemon=True).start()

    queue_info = _queue_info_for(patient_record.get('token')) if db_saved else None

    response = {
        'success': True,
        'token': patient_record.get('token'),
        'queue': queue_info,
        'message': (
            'Multi-Agent Triage pipeline completed.'
            if crew_failed_reason is None else
            'Deterministic triage completed; AI narrative degraded (LLM crew failed).'
        ),
        'patient_record': patient_record,
        'clinical_assessment': assessment.to_dict(),
        'db_saved': db_saved,
    }
    if crew_failed_reason is not None:
        response['ai_narrative_error'] = crew_failed_reason
    if not db_saved:
        response['message'] += (
            ' WARNING: database write failed (Neon connection issue) — this '
            'result is NOT persisted and will not appear in the triage queue.'
        )
    return jsonify(response)


@app.route('/api/triage_queue', methods=['DELETE'])
@require_api_key
def triage_queue_delete_api():
    """Wipes every patient record — irreversible, so it's the one endpoint gated by API_KEY."""
    clear_all_patients()
    return jsonify({'success': True, 'message': 'Triage queue cleared.'})


# ── Kiosk intake + waiting room ───────────────────────────────────────────────

# Rough average consultation length, used ONLY for the wait estimate shown to
# patients. It is a configurable placeholder, not a measured clinic figure —
# the board labels every number as an estimate.
QUEUE_AVG_SERVICE_MIN = int(os.getenv('QUEUE_AVG_SERVICE_MIN', '12'))

_VALID_STATUS = ('waiting', 'called', 'seen', 'left')


def _clean_intake(data):
    """Whitelist + sanitise the intake fields a kiosk may send with a triage run."""
    out = {}
    age = data.get('age')
    try:
        age = int(age)
        if 0 <= age <= 120:
            out['age'] = age
    except (TypeError, ValueError):
        pass
    sex = str(data.get('sex') or '').strip().lower()
    if sex in ('male', 'female', 'other', 'unspecified'):
        out['sex'] = sex
    complaint = str(data.get('complaint') or '').strip()
    if complaint:
        out['complaint'] = complaint[:500]
    phone = re.sub(r'[^0-9+]', '', str(data.get('phone') or ''))[:20]
    if phone:
        out['phone'] = phone
    return out


def _waiting_list():
    """Patients still in the room, in the order they will be seen: a patient who
    has been CALLED first, then sicker first (ESI 1 before 2 ...), higher priority
    score first, then first-come-first-served within a tier. This differs on
    purpose from the staff dashboard's order, which lists the NEWEST record first
    inside a tier and would be unfair in a waiting room."""
    rows = get_waiting_patients()
    rows.sort(key=lambda p: (p.get('status') != 'called',
                             p.get('esi_level', 5),
                             -(p.get('priority_score') or 0),
                             p.get('token_seq') or 0))
    out = []
    for i, p in enumerate(rows):
        esi = int(p.get('esi_level', 5))
        raw = i * QUEUE_AVG_SERVICE_MIN
        est = 0 if (esi == 1 or p.get('status') == 'called') else int(-(-raw // 5) * 5)
        out.append({
            'id': p['id'], 'token': p['token'], 'esi_level': esi,
            'status': p.get('status') or 'waiting',
            'is_shock': bool(p.get('is_shock')),
            'position': i + 1, 'ahead': i, 'est_wait_min': est,
        })
    return out


def _queue_info_for(token):
    if not token:
        return None
    try:
        for e in _waiting_list():
            if e['token'] == token:
                return {k: e[k] for k in ('token', 'position', 'ahead', 'est_wait_min', 'status', 'esi_level')}
    except Exception as exc:
        print(f"[QUEUE] could not compute queue info: {exc}")
    return None


@app.route('/api/waiting_room', methods=['GET'])
def waiting_room_api():
    """Privacy-safe board data: tokens, priority tier, position and estimated wait.
    No names, vitals or diagnoses — this is meant for a public screen."""
    try:
        entries = _waiting_list()
    except Exception as exc:
        return jsonify({'success': False, 'error': f'Waiting room unavailable: {exc}'}), 503
    public = [{k: e[k] for k in ('token', 'esi_level', 'status', 'position', 'est_wait_min')}
              for e in entries]
    return jsonify({
        'success': True,
        'now_serving': [e['token'] for e in entries if e['status'] == 'called'],
        'waiting': public,
        'count': len(public),
        'avg_service_min': QUEUE_AVG_SERVICE_MIN,
        'note': 'Wait times are estimates and change as more serious cases arrive.',
    })


@app.route('/api/token/<token>', methods=['GET'])
def token_lookup_api(token):
    info = _queue_info_for(token)
    if info is None:
        return jsonify({'success': False, 'error': 'Token not found or already seen.'}), 404
    return jsonify({'success': True, **info, 'avg_service_min': QUEUE_AVG_SERVICE_MIN})


@app.route('/api/patients/<patient_id>/status', methods=['POST'])
@require_api_key
def patient_status_api(patient_id):
    status = str((request.get_json(silent=True) or {}).get('status', '')).lower()
    if status not in _VALID_STATUS:
        return jsonify({'success': False, 'error': f'status must be one of {_VALID_STATUS}'}), 400
    try:
        changed = set_patient_status(patient_id, status)
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 503
    if not changed:
        return jsonify({'success': False, 'error': 'No such patient.'}), 404
    return jsonify({'success': True, 'id': patient_id, 'status': status})


@app.route('/api/queue/mark', methods=['POST'])
@require_api_key
def queue_mark_api():
    """Staff action by TOKEN (the public board never exposes patient ids): {token, status}."""
    data = request.get_json(silent=True) or {}
    token, status = str(data.get('token', '')), str(data.get('status', '')).lower()
    if status not in _VALID_STATUS:
        return jsonify({'success': False, 'error': f'status must be one of {_VALID_STATUS}'}), 400
    try:
        for e in _waiting_list():
            if e['token'] == token:
                set_patient_status(e['id'], status)
                return jsonify({'success': True, 'token': token, 'status': status})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 503
    return jsonify({'success': False, 'error': 'Token not in the waiting room.'}), 404


@app.route('/api/queue/call_next', methods=['POST'])
@require_api_key
def call_next_api():
    """Staff action: call the highest-priority waiting patient."""
    try:
        for e in _waiting_list():
            if e['status'] == 'waiting':
                set_patient_status(e['id'], 'called')
                return jsonify({'success': True, 'token': e['token'], 'id': e['id'], 'esi_level': e['esi_level']})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 503
    return jsonify({'success': False, 'error': 'Nobody is waiting.'}), 404


@app.route('/api/triage_queue', methods=['GET'])
def triage_queue_api():
    # This is the fix for records appearing to "vanish": get_all_patients()
    # now raises on a real DB failure instead of returning []. We must NOT
    # let that turn into a 200 response with an empty queue — the frontend
    # (page.tsx fetchQueue()) only overwrites its displayed queue when the
    # HTTP response is ok, so a non-200 here correctly leaves the last-known
    # good queue on screen instead of wiping it to "0 patients."
    try:
        queue = get_all_patients()
    except Exception as e:
        print(f"[APP] Failed to fetch triage queue from Neon: {e}")
        return jsonify({
            'success': False,
            'error': f'Database temporarily unreachable: {e}. Records are not lost — retry shortly.',
        }), 503

    return jsonify({
        'success': True,
        'queue': queue
    })

