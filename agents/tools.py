#tools.py
import statistics
from crewai.tools import tool
from scripts.analyze_video import analyze_video_complete


def _median_iqr_filtered(values):
    """Same outlier-robust aggregation scripts/analyze_video.py uses for
    ML-corrected readings — applied here so the live path's corrected BPM is
    computed the same validated way, not a plain mean of noisier per-cycle
    predictions."""
    if not values:
        return None
    sorted_v = sorted(values)
    n = len(sorted_v)
    q1, q3 = sorted_v[n // 4], sorted_v[(3 * n) // 4]
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered = [v for v in sorted_v if lo <= v <= hi] or sorted_v
    return statistics.median(filtered)


def get_vitals_dict(video_path: str) -> dict:
    """
    Plain-Python vitals extraction — the single source of truth used both by
    the CrewAI tool wrapper below (extract_vitals) and directly by
    web/routes.py to compute the deterministic clinical assessment
    (core.clinical.assess) *before* the LLM crew runs. Kept as one function
    so both callers can never see different numbers for the same patient.

    Returns a normalised dict always containing: success, bpm, rr, hrv,
    stress_index, sqi, mode — regardless of whether the source was a live
    webcam session or a recorded video file.
    """
    if video_path.lower() in ["live", "webcam"]:
        try:
            # Import web locally to prevent circular dependencies
            import web
            with web.session.processing_lock:
                metrics = dict(web.session.metrics)
                history = list(web.session.bpm_history)
                corrected_history = list(web.session.corrected_bpm_history)
                last_report = web.session.last_compiled_report
                # The engine's own accumulated RR readings — see below for why
                # the instantaneous metrics['rr'] is not what should reach the
                # clinical engine.
                rr_history = list(getattr(web.rppg_engine, 'rr_history', []) or [])

            if not history and metrics.get('status') == 'WAITING':
                if last_report:
                    return {
                        'success': True,
                        'bpm': last_report['vitals']['heart_rate_bpm'] or last_report['vitals']['heart_rate_avg'] or 0.0,
                        'confidence': last_report['signal_quality']['confidence_pct'],
                        'sqi': last_report['signal_quality']['confidence_pct'],
                        'stability': 50.0,
                        'rr': last_report['vitals']['respiratory_rate'] or 0.0,
                        # The compiled report doesn't separately track rr_confidence —
                        # default to a conservative 50 (not 100) so a stale/borderline RR
                        # reading isn't silently trusted as fully reliable.
                        'rr_confidence': 50.0,
                        'hrv': last_report['vitals']['hrv_rmssd_ms'] or 0.0,
                        'stress_index': last_report['vitals']['stress_index'] or 0.0,
                        'estimated_lux': last_report['signal_quality']['luminance_lux'],
                        'motion_delta': 0,
                        'mode': 'WEBCAM_COMPLETED_SESSION',
                    }
                return {
                    'success': False,
                    'bpm': 0, 'rr': 0, 'hrv': 0, 'stress_index': 0, 'sqi': 0,
                    'mode': 'WEBCAM_NO_DATA',
                    'error': 'No active live webcam session data or completed report found. Start a webcam session first.',
                }

            import statistics as stats
            if len(history) >= 2:
                avg_bpm = stats.mean(history)
            elif len(history) == 1:
                avg_bpm = history[0]
            else:
                avg_bpm = metrics.get('bpm', 0)

            # Hybrid ML correction — prefer the IQR-filtered median over the
            # session's corrected per-cycle readings (same aggregation
            # validated offline at 6.48 BPM session MAE) over the classical
            # mean whenever enough corrected readings exist. Falls back to
            # the classical average if the model isn't loaded or the session
            # hasn't produced enough high-SQI cycles yet — never silently
            # worse than the previously-shipped behaviour.
            corrected_bpm = _median_iqr_filtered(corrected_history) if len(corrected_history) >= 3 else None
            final_bpm = corrected_bpm if corrected_bpm is not None else avg_bpm

            # HR ensemble (peak selector + spectro-temporal tracker, core/hr_ensemble.py) takes priority when
            # it has enough decisions; otherwise the hybrid/classical value above stands, as before.
            ens_est = None
            ens = getattr(web.session, 'hr_ensemble', None)
            if ens is not None:
                ens.flush(timeout=5.0)           # let the worker drain what the video loop already queued (stays alive)
                e = ens.estimate()
                if not e['error'] and e['bpm'] is not None and e['n_selector'] >= 10:
                    ens_est = e
                    final_bpm = e['bpm']

            # Respiratory rate gets the SAME session-level aggregation heart
            # rate already gets, and for the same reason. metrics['rr'] is the
            # latest single frame's estimate; feeding that to
            # core.clinical.assess() meant the tachypnea threshold (RR > 20,
            # which combines with tachycardia into the Compensated Shock flag)
            # was decided by one instant of the noisiest channel this engine
            # produces, while HR got a median over the whole session.
            #
            # Measured on BIDMC (scripts/validate_rr_bidmc.py, 53 recordings,
            # human-annotated breaths): per-window MAE 3.09 br/min vs.
            # session-level MAE 2.14 (median 0.34). Aggregating is worth ~1
            # br/min of accuracy on a threshold that sits at 20.
            session_rr = _median_iqr_filtered(rr_history) if len(rr_history) >= 3 else None
            final_rr = session_rr if session_rr is not None else metrics.get('rr', 0)

            return {
                'success': True,
                'bpm': round(final_bpm, 1),
                'bpm_classical': round(avg_bpm, 1),
                'bpm_corrected': round(corrected_bpm, 1) if corrected_bpm is not None else None,
                'bpm_ensemble': round(ens_est['bpm'], 1) if ens_est else None,
                'bpm_selector': round(ens_est['selector'], 1) if ens_est and ens_est['selector'] is not None else None,
                'bpm_tracker': round(ens_est['tracker'], 1) if ens_est and ens_est['tracker'] is not None else None,
                'confidence': metrics.get('confidence', 0),
                'sqi': metrics.get('sqi', metrics.get('confidence', 0)),
                'stability': metrics.get('stability', 0),
                'rr': round(final_rr, 1) if final_rr else 0,
                'rr_instantaneous': metrics.get('rr', 0),
                'rr_readings': len(rr_history),
                'rr_confidence': metrics.get('rr_confidence', 0),
                'hrv': metrics.get('hrv', 0),
                'stress_index': metrics.get('stress_index', 0),
                'estimated_lux': metrics.get('estimated_lux', 0),
                'motion_delta': metrics.get('motion_delta', 0),
                'mode': 'WEBCAM',
            }
        except Exception as e:
            return {
                'success': False,
                'bpm': 0, 'rr': 0, 'hrv': 0, 'stress_index': 0, 'sqi': 0,
                'mode': 'WEBCAM_ERROR',
                'error': f"Error reading live webcam session data: {str(e)}",
            }

    try:
        import web
        correction_path = web.CORRECTION_MODEL_PATH if web.correction_model is not None else None
        results = analyze_video_complete(video_path, correction_model_path=correction_path, use_ensemble=True)
        # analyze_video_complete() only estimates bpm/confidence today — no
        # rr/hrv/stress_index for pre-recorded files yet (those currently
        # only exist in the live/session pipeline). Surface that honestly
        # rather than inventing zeros that read as "measured normal."
        results.setdefault('rr', 0)
        results.setdefault('rr_confidence', 0)
        results.setdefault('hrv', 0)
        results.setdefault('stress_index', 0)
        results.setdefault('sqi', results.get('confidence', 0))
        # Prefer the hybrid-corrected BPM over classical, same as the live
        # path — 'bpm' stays the single field core.clinical.assess reads,
        # 'bpm_classical'/'final_bpm_corrected' stay available for honesty.
        if results.get('final_bpm_corrected') is not None:
            results['bpm_classical'] = results.get('final_bpm_classical', results['bpm'])
            results['bpm'] = results['final_bpm_corrected']
        # HR ensemble beats both when it produced an answer (core/hr_ensemble.py).
        if results.get('final_bpm_ensemble') is not None:
            results.setdefault('bpm_classical', results['bpm'])
            results['bpm'] = results['final_bpm_ensemble']
        return results
    except Exception as e:
        return {
            'success': False,
            'bpm': 0, 'rr': 0, 'hrv': 0, 'stress_index': 0, 'sqi': 0,
            'mode': 'FILE_ERROR',
            'error': f"Error executing vitals extraction: {str(e)}",
        }


@tool("Vitals Extraction Tool")
def extract_vitals(video_path: str) -> str:
    """
    Extracts physiological vitals (BPM, Respiratory Rate, HRV, and Stress Index)
    from a pre-recorded patient video file (e.g., 'uploads/face.mp4') OR from the active live webcam session
    if video_path is set to 'live' or 'webcam'.

    Args:
        video_path (str): Path to patient video file, OR 'live'/'webcam' for active session.

    Returns:
        str: A stringified dictionary containing the extracted metrics.
    """
    return str(get_vitals_dict(video_path))
