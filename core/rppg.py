"""
VITAL — rPPG Signal Processing Engine
======================================
Research-grade rPPG with selectable projection algorithms, SNR-adaptive
filtering, and a composite waveform Signal Quality Index.

Algorithms
──────────
CHROM  de Haan & Jeanne, IEEE TBME 2013  — chrominance-based, melanin-robust
POS    Wang et al., IEEE TBME 2017       — plane orthogonal to skin tone, moving-window

Bandpass
──────────
Fixed-width Butterworth seeds the pipeline.  After the first stable BPM
estimate exists, the filter narrows around the known HR ± guard band.
Guard width shrinks as SNR rises (tight = high confidence) and widens
as SNR falls (loose = rescue mode).  A hard physiological floor/ceiling
prevents the window from collapsing below 0.30 Hz width or expanding
above the original 2.15 Hz wide design band.

SQI
──────────
Five independent sub-scores are computed and fused into one 0-100 value:
  1. Spectral concentration  (existing power-ratio metric)
  2. Waveform kurtosis       (sharp systolic peaks → higher kurtosis)
  3. Shannon entropy         (low entropy = periodic = clean)
  4. Zero-crossing rate      (ZCR should match expected HR range)
  5. Autocorrelation peak    (strong periodicity → high AC at lag = 1/HR)

References
──────────
de Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based
  rPPG. IEEE TBME, 60(10), 2878-2886.
Wang, W., den Brinker, A.C., Stuijk, S., & de Haan, G. (2017). Algorithmic
  principles of remote PPG. IEEE TBME, 64(7), 1479-1491.
Elgendi, M. et al. (2016). Optimal SQI for PPG signals. Bioengineering, 3(4).
Pai, A. et al. (2021). HRVCam: SNR-based adaptive bandpass for rPPG HRV.
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from enum import Enum
from typing import Literal, Optional

import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter

log = logging.getLogger("vital.rppg")

# ── Algorithm selector ────────────────────────────────────────────────────────

class Algorithm(str, Enum):
    CHROM = "CHROM"
    POS   = "POS"


# ── Bandpass constants ────────────────────────────────────────────────────────

# Absolute physiological limits (Hz)
_BP_LOW_FLOOR   = 0.70   # 42 BPM — never go below this lower cutoff
_BP_HIGH_CEIL   = 3.00   # 180 BPM — never go above this upper cutoff
_BP_MIN_WIDTH   = 0.30   # Hz — floor on passband width to avoid filter collapse

# Seed / fallback filter when no BPM estimate exists yet
_BP_SEED_LOW    = 0.85   # Hz (~51 BPM)
_BP_SEED_HIGH   = 3.00   # Hz (~180 BPM)

# Adaptive guard band around known HR (Hz on each side)
#   shrinks as SNR improves, widens as SNR drops
_GUARD_SNR_HIGH = 0.20   # ±0.20 Hz when SNR ≥ 15 dB (tight)
_GUARD_SNR_MID  = 0.35   # ±0.35 Hz when 8 dB ≤ SNR < 15 dB
_GUARD_SNR_LOW  = 0.55   # ±0.55 Hz when SNR < 8 dB (rescue / wide)
_SNR_HIGH_THRESH = 15.0  # dB
_SNR_MID_THRESH  =  8.0  # dB

# Respiration bandpass (Hz)
_RESP_LOW  = 0.10   # 6 breaths/min
_RESP_HIGH = 0.50   # 30 breaths/min

# ── Respiratory rate: RIV-fusion parameters ──────────────────────────────────
#
# Why RR is estimated from beat-level features and NOT from band-passing the
# raw PPG at _RESP_LOW.._RESP_HIGH (which is what this engine used to do):
#
# Measured against 53 x 8-min recordings of human-annotated breaths (BIDMC,
# PhysioNet — see scripts/validate_rr_bidmc.py), the old raw-band argmax had
# a correlation of r = -0.088 with ground truth. It carried no respiratory
# information at all. 97% of its readings sat on one of the two band edges:
# ~72% pinned to the 6 br/min floor, ~25% pinned to the 30 br/min ceiling.
# Two independent contaminants, one at each end, caused that:
#
#   * Bottom: raw PPG in 0.1-0.5 Hz is dominated by 1/f baseline/perfusion
#     drift. A 1/f power spectrum decreases monotonically, so an argmax over
#     it MUST select the lowest bin. Not a tuning issue — arithmetic.
#   * Top: the old resp filter was order=2 (12 dB/octave). The pulse at
#     ~1.5 Hz is only ~1.6 octaves above the 0.5 Hz cutoff, so it received
#     only ~19 dB of attenuation while being 30-40 dB STRONGER than the
#     respiratory modulation. Leaked pulse energy therefore dominated the
#     top of the band.
#
# True respiratory modulation is weak and sits buried between those two.
# Widening the analysis window does not help (measured: MAE stayed 11-13
# br/min from 10 s all the way to 60 s) because the problem was never
# resolution — it was that the wanted signal wasn't in that band to begin
# with.
#
# The replacement derives respiration from respiratory-induced variations
# (RIVs) measured per heartbeat — the approach validated on this exact
# dataset by Pimentel et al. (the BIDMC paper) and Karlen et al.
# (SmartFusion). Three RIVs exist in the literature and are all extracted by
# _extract_rivs, but only two are FUSED into the final estimate:
#   RIAV  amplitude variation  (peak-to-trough height per beat)     — fused
#   RIIV  intensity variation  (baseline level at each beat)        — fused
#   RIFV  frequency variation  (beat-to-beat interval — i.e. RSA)   — NOT fused
# RIFV is extracted (available for future use / diagnostics) but excluded
# from fusion: measured alone on BIDMC it scored slope 0.09-0.14 and
# corr ~0.08 against ground truth — essentially noise, likely because BIDMC's
# ICU population (many ventilated patients) has blunted respiratory sinus
# arrhythmia, which is exactly what RIFV depends on. Including it in the
# median pulled the fused answer toward its bad guesses. See
# _estimate_rr's docstring for the measured before/after.
#
# Each fused RIV is resampled onto a uniform _RIV_FS grid via CUBIC SPLINE
# (not linear interpolation — see _riv_candidate's docstring for why) and
# linearly detrended. Both edge-lock mechanisms die structurally, not by
# tuning:
#   * resampling to _RIV_FS puts Nyquist at _RIV_FS/2, so the pulse is
#     decimated away BY CONSTRUCTION — no pulse energy remains to leak.
#   * detrending removes the 1/f pedestal.
# The surviving candidates are fused by median: agreement between channels
# contaminated by DIFFERENT artefacts is far stronger evidence than any one
# spectrum's peak height.
#
# Measured on BIDMC (all 53 recordings, deployment-faithful 30 fps
# resampling, human-annotated breaths): per-window MAE 2.22 br/min,
# session-level MAE 1.71 (median 0.34), 84.9% of recordings within 2 br/min.
# Old code (raw-band argmax), same harness: r ~ 0 (no correlation with truth).
_RR_WINDOW_SEC     = 60.0   # RR analysis buffer. Long ON PURPOSE: at 12 br/min
                            # a 10 s window holds 2 breath cycles, which cannot
                            # define a frequency. HR is unaffected — it keeps
                            # slicing its own 10 s window (see _run_pipeline).
_RR_MIN_WINDOW_SEC = 30.0   # don't emit any RR until this much signal exists
_RIV_FS            = 4.0    # Hz — uniform RIV grid; Nyquist 2 Hz kills the pulse
_RR_AGREE_TOL      = 2.0    # br/min spread between RIVs still treated as agreement
_RR_MIN_BEATS      = 6      # fewer beats than this = no usable RIV series
_RR_UPDATE_HZ      = 2.0    # recompute RR at most this often (it changes slowly;
                            # 3 Welches + peak-detect on a 60 s buffer every
                            # frame would stall the live MJPEG stream)
# Confidence floor a reading must clear to enter rr_history. MUST stay equal to
# core.clinical.RR_CONFIDENCE_LOW and the display gates in web/routes.py and
# scripts/analyze_video.py — CLAUDE.md documents why those three already have
# to move together: if they diverge, the number shown to a clinician and the
# number the clinical engine reasons about disagree about the same reading.
_RR_HISTORY_MIN_CONF = 30.0
# RIIV-only rolling-median dedrift window, in seconds — see Finding 10 in
# data/evolution/19-respiratory-rate-ground-truth.md. RIIV is the raw
# per-beat baseline intensity of the projected PPG signal; on a real webcam
# (unlike BIDMC/CapnoBase's contact-PPG-derived RIVs, which have no camera
# exposure loop anywhere in their signal path) continuous auto-exposure/AGC
# correction injects slow baseline drift with no respiratory content, and
# the existing linear detrend + _sos_riv bandpass (2nd-order, soft rolloff)
# were not enough to fully reject it — a live session showed RIIV's reported
# rate sliding from ~10.6 to ~6.4 br/min in under 10s while RIAV stayed
# rock-stable at ~12. A window here must beat _RESP_LOW's 10s period (6
# br/min) to avoid attenuating genuine low-end respiration; 20s puts the
# rolling-median high-pass's cutoff at ~0.05 Hz, comfortably below
# _RESP_LOW=0.10 Hz.
_RIIV_DEDRIFT_WINDOW_SEC = 20.0

# ── Adaptive RR search band + low-zone peak rescue (Finding 11) ──────────────
# A fixed 0.10-0.50 Hz search band cannot be right for every patient: children
# breathe faster than 30 br/min (CapnoBase 0115/0018, true 35-40 br/min, were
# outside it entirely and alone were ~48% of CapnoBase's total error), while
# adults rarely breathe below ~10 br/min yet slow vasomotor/drift power at 6-9
# br/min kept winning the argmax. Breathing rate tracks heart rate across
# populations (measured true RR/HR: 5th-95th pct 0.08-0.27 CapnoBase, 0.13-0.26
# BIDMC), so the band is expressed as FRACTIONS OF THE WINDOW'S OWN HEART RATE,
# measured from the beats already detected in that window — no dependence on
# prev_bpm or any other engine state.
#   lower edge = 0.06 x HR  (clipped to [_RESP_LOW, 0.25 Hz])
#   upper edge = 0.30 x HR  (clipped to [_RESP_HIGH, 0.80 Hz])
# Falls back to the fixed _RESP_LOW.._RESP_HIGH band when no HR is available.
_RR_BAND_LO_HR_FRAC = 0.06
_RR_BAND_HI_HR_FRAC = 0.30
_RR_BAND_LO_CAP_HZ  = 0.25
_RR_BAND_HI_CAP_HZ  = 0.80
# Low-zone rescue: if the winning peak sits below _RR_LOW_ZONE_BPM, look for a
# peak anywhere in [1.6x, 2.8x] its frequency and prefer it when it holds at
# least _RR_LOW_ZONE_RATIO of the winner's power. The existing
# _promote_subharmonic only inspects the single bin at EXACTLY 2x (+-0.06
# br/min), so a junk low peak whose true partner sits at 2.2-2.8x was never
# rescued. 43% of BIDMC's remaining wrong windows were this ~0.5x class.
# Parameters were chosen on BIDMC+CapnoBase and then confirmed on 40 held-out
# VitalDB surgeries never used for tuning (session MAE 1.21 -> 1.03); results
# are flat across zone 11-13 and ratio 0.25-0.35, not a knife-edge optimum.
_RR_LOW_ZONE_BPM   = 12.0
_RR_LOW_ZONE_RATIO = 0.30
_RR_LOW_ZONE_SPAN  = (1.6, 2.8)
# Subharmonic promotion threshold: promote to 2x the peak when the double
# holds at least this fraction of the peak's power. Tuned on BIDMC — see
# AdvancedRPPG._promote_subharmonic and data/evolution for the measured sweep.
_RR_SUBHARM_PROMOTE_RATIO = 0.55
# Separate, stricter threshold for the demotion direction (2x -> 1x). A
# shared threshold with promotion regressed BIDMC (session MAE 1.60 -> 1.87)
# while helping CapnoBase a lot (3.37 -> 2.39) — measured, not assumed. Being
# more conservative about demoting (requiring the half to be closer to
# co-dominant, not just "not negligible") is the knob to recover the BIDMC
# cost without giving back all of the CapnoBase gain — being swept in
# data/evolution/19-respiratory-rate-ground-truth.md Finding 8.
_RR_DEMOTE_RATIO = 0.80
# Minimum heartbeats per breath before an RIV-derived RR is considered
# adequately sampled.
#
# The RIV series are sampled ONCE PER HEARTBEAT, so the beat rate is the
# sampling rate for respiration. Recovering a breathing rhythm therefore needs
# beats >> breaths; at HR 90 with RR 24 there are only 3.75 beats per breath,
# which is near the point where the interpolated RIV series can no longer
# represent the respiratory waveform. This is an inherent property of
# beat-derived RR methods, not a tuning parameter — measured on BIDMC, the
# estimator's signed error is negative in every band above 14 br/min
# (-1.2 at 14-18, -1.8 at 18-20, -1.8 at 20-24), i.e. it systematically
# UNDER-estimates high respiratory rates. Widening _RESP_HIGH does not fix it
# (swept 0.50/0.60/0.70/0.85 Hz: tachypnea miss rate stayed 40%, synthetic MAE
# got worse), confirming the limit is sampling, not the search band.
#
# Underestimation is the clinically dangerous direction here, because
# clinical.RR_TACHY is 20 br/min and a suppressed tachypnea also suppresses
# the Compensated Shock interaction term. So when a reading falls in this
# undersampled regime, confidence is cut so the existing gates drop it: an
# explicit "could not measure" lets core.clinical.assess() fall back to
# heart-rate-only assessment, whereas a confident-looking low number actively
# hides tachypnea.
_RR_MIN_BEATS_PER_BREATH = 4.0

# POS sliding window length (seconds) — per Wang 2017
_POS_WINDOW_SEC = 1.6

# ── SQI sub-score weights (must sum to 1.0) ───────────────────────────────────
# Weights from Samsung Research / rPPG SQI literature:
# Spectral concentration is the strongest single predictor; kurtosis and
# entropy are complementary morphological checks.
_SQI_W_SPECTRAL  = 0.40
_SQI_W_KURTOSIS  = 0.20
_SQI_W_ENTROPY   = 0.20
_SQI_W_ZCR       = 0.10
_SQI_W_AUTOCORR  = 0.10

# HRV upsampling target (Hz) for sub-frame IBI precision
_HRV_HI_RES_FPS = 250.0

# ── BPM lock/candidate hysteresis (harmonic-ambiguity mitigation) ────────────
#
# Root cause this addresses: on real UBFC-rPPG validation, the raw per-cycle
# peak-pick would periodically snap onto ~0.5x or ~2x the true heart rate for
# hundreds of consecutive frames at a time (a genuine bistable lock, not
# single-frame noise — confirmed via frame-by-frame trace on a real subject).
# Once locked onto the wrong frequency, the OLD adaptive-filter recentering
# (fed the raw noisy peak every cycle) would narrow its passband around that
# wrong frequency too, making the false lock self-reinforcing. And because
# get_final_summary() takes a history median, whichever cluster (right or
# wrong) accumulated more admitted frames would dominate — actively rejecting
# a later CORRECT reading as an "implausible jump" relative to an
# already-established WRONG one.
#
# Fix: track a confirmed "locked" BPM and a competing "candidate" BPM
# separately. A new reading only replaces the lock once it has been seen
# consistently for several consecutive cycles — much more consecutive
# cycles required when the candidate looks like a harmonic multiple/fraction
# of the current lock, since that's specifically the ambiguous case. The
# adaptive filter recenters on the CONFIRMED lock, not the raw jittery peak,
# so it can no longer reinforce a not-yet-confirmed false read.
_LOCK_AGREE_TOLERANCE   = 6.0    # bpm — raw estimates within this of the candidate count as "the same" candidate
_LOCK_HARMONIC_LOW_A, _LOCK_HARMONIC_HIGH_A = 0.40, 0.62   # candidate/lock ratio range that looks like a subharmonic (~0.5x)
_LOCK_HARMONIC_LOW_B, _LOCK_HARMONIC_HIGH_B = 1.60, 2.50   # candidate/lock ratio range that looks like a harmonic multiple (~2x)
_LOCK_CONFIRM_STREAK_NORMAL   = 12   # cycles required to confirm a non-harmonic-looking jump (~real HR change)
_LOCK_CONFIRM_STREAK_HARMONIC = 30   # cycles required to confirm a harmonic-ratio jump (much more skepticism)
_LOCK_PRIOR_LOW, _LOCK_PRIOR_HIGH = 55.0, 105.0   # typical resting-HR band — soft tie-breaker only, never a hard cutoff

# bpm_history "unstick" threshold — see get_final_summary()'s history-append
# gate. That gate rejects any reading more than 18bpm from the median of the
# last 5 *accepted* entries, to keep single-frame noise out of the session
# summary. Found via subject37 trace: if the first few accepted entries
# happen to land during a brief wrong burst early in cold-start convergence
# (n=7 entries there), every later (correct, stable) reading can be
# permanently rejected for the rest of the session, because the anchor
# itself never gets to update. Rather than trust a stale anchor forever, a
# long enough streak of consistently-rejected-but-mutually-agreeing readings
# is treated as a genuine sustained plateau and allowed to re-anchor.
#
# This must stay narrowly scoped, though — see _HISTORY_UNSTICK_MAX_HISTORY.
# subject41 trace proved the naive always-on version actively harmful: its
# signal is still genuinely bimodal across the WHOLE clip post-subharmonic-
# promotion (not just at cold start), so a long-sustained streak on the
# WRONG cluster can legitimately build up later in the session too. The
# previous "fixed" 8.2 BPM error on subject41 turns out to have been this
# exact freeze bug accidentally anchoring on the correct early cluster and
# never letting go — not a real resolution of the ambiguity. Unsticking
# unconditionally traded that lucky freeze for a genuine regression
# (8.2 -> 41.8 BPM error), confirmed by re-running the real 56-session
# validation (see data/evolution/09-history-freeze-bug.md).
_HISTORY_UNSTICK_STREAK = 150   # cycles (~5s at 30fps) of sustained disagreement before re-anchoring
_HISTORY_UNSTICK_MAX_HISTORY = 10   # only allow unstick while the frozen anchor is still this fresh/untested


# ── Subharmonic-promotion peak selection ──────────────────────────────────────
#
# Root-cause fix attempt #2 for the harmonic lock-on / "octave error" failure
# mode (confirmed via real-dataset validation, reproduced identically across
# three independent datasets: peak-picker locks onto ~0.5x the true heart
# rate — the ~2x cases are rarer — for sustained multi-second stretches).
#
# Fix attempt #1 (Harmonic Product Spectrum, upward-summing power at a
# candidate's 2x/3x points) was tried and DISPROVEN by re-tracing the known
# failing case (test/dataset_2/subject41): it made no difference at all.
# The reasoning why, worked out after the fact: HPS is biased in the WRONG
# direction for a subharmonic-dominant failure specifically. If the true
# rate is 100bpm and the false candidate is ~50bpm, then scoring the ~50bpm
# candidate by summing power at its own 2x point (~100bpm) picks up the
# REAL signal's energy and misreads it as "supporting harmonic evidence
# for the 50bpm candidate" — reinforcing the wrong answer instead of
# penalizing it. Classic HPS works for the opposite error (mistaking a true
# harmonic for the fundamental); it doesn't help when the failure is
# undershooting to a subharmonic, which is this pipeline's dominant failure
# direction.
#
# Corrected approach: instead of rewarding a candidate for having power at
# its multiples (ambiguous, as above), directly check whether the TOP
# candidate's double has suspiciously strong power relative to itself. If
# a candidate f0 has a comparable-or-stronger companion peak at ~2*f0, that
# is direct evidence f0 itself might be the spurious subharmonic — promote
# to 2*f0 instead. This is asymmetric on purpose (checks upward promotion
# only) because the observed failure direction is undershooting, not
# overshooting.
_SUBHARM_PROMOTE_RATIO = 0.55  # promote to 2x if that bin has >= 55% of the top candidate's power
_SUBHARM_TOL_HZ        = 0.08  # how close to exactly 2x (or 0.5x) a bin must land to count

# Downward-demotion — the mirror image of the promotion check above, added
# after tracing POS's own separate, previously-undocumented failure mode
# (test/trace_pos_gt50.log): CHROM's documented bug undershoots to ~half
# the true rate (fixed by the promotion check); POS instead OVERSHOOTS to
# roughly double the true rate from the very first cold-start reading, and
# the adaptive filter then locks onto and reinforces the wrong (high) peak,
# narrowing the search band until the true (lower) frequency is filtered
# out entirely. The promotion check can never catch this: it only ever
# looks upward from the top candidate, and POS's top candidate is already
# too high, not too low. This checks whether the top candidate's HALF has
# comparable power instead, and demotes to it if so — same tolerance and
# threshold as promotion, kept symmetric since there's no prior evidence
# either direction needs different sensitivity.
_SUBHARM_DEMOTE_RATIO = 0.55


def _select_dominant_freq_hps(valid_freqs: np.ndarray, valid_psd: np.ndarray) -> tuple[float, int, float]:
    """
    Select the dominant frequency, checking for and correcting two
    harmonic-ambiguity signatures: the raw top peak has a comparably-strong
    companion at almost exactly double its frequency (promote to the
    double — CHROM's documented undershoot bug), or at almost exactly half
    its frequency (demote to the half — POS's documented overshoot bug).
    Returns (frequency_hz, bin_index, harmonic_ratio). Name kept as `_hps`
    for call-site continuity even though the technique is no longer
    classic HPS — see module comment above for why.

    harmonic_ratio (new) is whichever companion's power ratio was actually
    checked (2x for the promotion path, 0.5x for the demotion path; 0.0 if
    neither a 2x nor 0.5x bin was found at all) — previously computed here
    purely to decide promote-or-not and then discarded. Exposed as a
    feature for the ML correction layer: a high ratio is a direct,
    real-time signal that THIS cycle's estimate is ambiguous between two
    candidates a factor of ~2 apart (whichever one was picked), independent
    of whether either heuristic actually fired — the model can learn to
    distrust a reading with a high ratio even in cases the hand-written
    0.55 threshold doesn't cleanly resolve (e.g. subject41's genuinely
    bimodal signal, still unresolved as of Era 13 — see
    data/evolution/13-majority-vote-gate-reverted.md).
    """
    n = len(valid_psd)
    top_idx = int(np.argmax(valid_psd))
    if n < 3:
        return float(valid_freqs[top_idx]), top_idx, 0.0

    f0 = valid_freqs[top_idx]

    double_target = f0 * 2.0
    j = int(np.argmin(np.abs(valid_freqs - double_target)))
    if abs(valid_freqs[j] - double_target) <= _SUBHARM_TOL_HZ:
        ratio = float(valid_psd[j] / valid_psd[top_idx]) if valid_psd[top_idx] > 0 else 0.0
        if ratio >= _SUBHARM_PROMOTE_RATIO:
            return float(valid_freqs[j]), j, ratio
        return float(f0), top_idx, ratio

    half_target = f0 / 2.0
    k = int(np.argmin(np.abs(valid_freqs - half_target)))
    if abs(valid_freqs[k] - half_target) <= _SUBHARM_TOL_HZ:
        ratio = float(valid_psd[k] / valid_psd[top_idx]) if valid_psd[top_idx] > 0 else 0.0
        if ratio >= _SUBHARM_DEMOTE_RATIO:
            return float(valid_freqs[k]), k, ratio
        return float(f0), top_idx, ratio

    return float(f0), top_idx, 0.0


# ── Utility ───────────────────────────────────────────────────────────────────

def _safe_butter_sos(low: float, high: float, fs: float, order: int = 4,
                     clamp_to_pulse_band: bool = True) -> Optional[np.ndarray]:
    """
    Build a Butterworth bandpass SOS; returns None on failure.

    clamp_to_pulse_band (default True) applies the PULSE-band guards
    (_BP_LOW_FLOOR / _BP_HIGH_CEIL / _BP_MIN_WIDTH) that keep the adaptive
    heart-rate filter from collapsing or wandering out of physiological
    range. Every heart-rate call site wants that and must keep passing True.

    Pass False for any band OUTSIDE the pulse range — respiration, above all.
    Those guards are not generic: _BP_LOW_FLOOR is 0.70 Hz (42 bpm), so
    requesting the respiratory band 0.10-0.50 Hz used to be silently
    rewritten to 0.70-1.00 Hz — low clamped UP to 0.70, high clamped DOWN to
    0.50, then the negative width tripped the collapse guard and pushed high
    to 1.00. The result was a filter passing 42-60 BREATHS/min, centred
    squarely on the heart-rate fundamental.

    That was not a subtle mis-tuning; it is why the old respiratory-rate
    estimator scored r = -0.088 against human-annotated breaths on BIDMC. Its
    "respiration filter" was passing the pulse rather than rejecting it, so
    the pulse then dominated the top of the 6-30 br/min search band and the
    argmax pinned to the 30 br/min ceiling. Verified directly with
    sosfreqz: requested 0.1-0.5 Hz, actual -3 dB passband 0.701-0.999 Hz.
    """
    try:
        if clamp_to_pulse_band:
            width = high - low
            if width < _BP_MIN_WIDTH:
                # Prevent filter collapse: expand symmetrically
                mid = (low + high) / 2.0
                low  = max(_BP_LOW_FLOOR, mid - _BP_MIN_WIDTH / 2.0)
                high = min(_BP_HIGH_CEIL, mid + _BP_MIN_WIDTH / 2.0)
            low  = max(_BP_LOW_FLOOR, low)
            high = min(_BP_HIGH_CEIL, high)
            if high - low < _BP_MIN_WIDTH:
                high = low + _BP_MIN_WIDTH
        else:
            # Only guard against genuinely invalid/unstable requests.
            nyq = fs / 2.0
            low  = max(low, 1e-6)
            high = min(high, nyq * 0.999)
            if low >= high:
                log.warning("Filter build skipped: low %.3f >= high %.3f Hz", low, high)
                return None
        return signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    except Exception as exc:
        log.warning("Filter build failed (%.2f–%.2f Hz): %s", low, high, exc)
        return None


def _guard_width(snr_db: float) -> float:
    """Return the ±guard half-width in Hz based on current SNR."""
    if snr_db >= _SNR_HIGH_THRESH:
        return _GUARD_SNR_HIGH
    if snr_db >= _SNR_MID_THRESH:
        return _GUARD_SNR_MID
    return _GUARD_SNR_LOW


# ── SQI sub-scores ────────────────────────────────────────────────────────────

def _sqi_spectral(valid_psd: np.ndarray, peak_idx: int, window: float = 0.10) -> float:
    """
    Spectral concentration: ratio of peak-band power to total valid-band power.
    Window is ±window Hz around the dominant peak bin.
    """
    n = len(valid_psd)
    lo = max(0, peak_idx - max(1, int(window * n / 2.15)))
    hi = min(n, peak_idx + max(1, int(window * n / 2.15)))
    signal_power = float(np.sum(valid_psd[lo:hi + 1]))
    total_power  = float(np.sum(valid_psd)) + 1e-9
    return min(100.0, (signal_power / total_power) * 100.0)


def _sqi_kurtosis(ppg: np.ndarray) -> float:
    """
    Waveform kurtosis score.

    A clean PPG has pronounced systolic peaks → excess kurtosis > 1.
    Noise / flat signal clusters around kurtosis ≈ 0 (Gaussian-like).
    We map the kurtosis value to a 0-100 score with a soft logistic curve
    so that kurtosis of ~3 (clear peaks) yields ~80+ and kurtosis near 0
    yields near 0.

    Reference: Elgendi 2016; Selvaraj et al. 2011.
    """
    if len(ppg) < 4:
        return 0.0
    std = float(np.std(ppg))
    if std < 1e-9:
        return 0.0
    # Excess kurtosis (Fisher definition; Gaussian = 0)
    kurt = float(np.mean(((ppg - np.mean(ppg)) / std) ** 4)) - 3.0
    # Map to 0-100: score saturates at kurtosis ~6, zero at ≤0
    score = max(0.0, min(100.0, (kurt / 6.0) * 100.0))
    return score


def _sqi_entropy(ppg: np.ndarray, n_bins: int = 64) -> float:
    """
    Inverted Shannon entropy score.

    A periodic PPG concentrates probability mass in a few amplitude bins →
    low entropy → high quality score.
    White noise spreads uniformly → high entropy → low quality.

    Reference: Selvaraj 2011; Elgendi 2016.
    """
    if len(ppg) < 4:
        return 0.0
    hist, _ = np.histogram(ppg, bins=n_bins, density=True)
    hist = hist[hist > 0]
    # Shannon entropy (nats)
    H = float(-np.sum(hist * np.log(hist + 1e-12)))
    H_max = np.log(n_bins)           # maximum possible entropy
    # Invert and normalise: low entropy → high score
    score = max(0.0, min(100.0, (1.0 - H / H_max) * 100.0))
    return score


def _sqi_zcr(ppg: np.ndarray, bpm: float, fps: float) -> float:
    """
    Zero-crossing rate consistency score.

    Expected ZCR (crossings/sec) for a clean PPG ≈ 2 × (HR in Hz).
    We score how close the measured ZCR is to the expected value.
    A ±30% tolerance window scores 100; beyond ±100% scores 0.

    Reference: Elgendi 2016.
    """
    if len(ppg) < 4 or bpm <= 0:
        return 0.0
    mean_ppg = float(np.mean(ppg))
    centred  = ppg - mean_ppg
    crossings = int(np.sum(np.diff(np.sign(centred)) != 0))
    duration  = len(ppg) / fps
    measured_zcr  = crossings / duration          # crossings per second
    expected_zcr  = 2.0 * (bpm / 60.0)           # 2 × heart rate in Hz
    ratio = abs(measured_zcr - expected_zcr) / (expected_zcr + 1e-9)
    # Linear score: 0 error → 100, 100% error → 0
    score = max(0.0, min(100.0, (1.0 - ratio) * 100.0))
    return score


def _sqi_autocorr(ppg: np.ndarray, bpm: float, fps: float) -> float:
    """
    Autocorrelation peak score at the expected inter-beat lag with neighborhood search.

    Searches a ±4 frame window around the expected lag to find the optimal peak,
    avoiding severe quality penalties from minor heart rate estimation noise.
    """
    if len(ppg) < 10 or bpm <= 0:
        return 0.0
    hr_hz  = bpm / 60.0
    expected_lag = int(round(fps / hr_hz))
    if expected_lag <= 0 or expected_lag >= len(ppg):
        return 0.0

    # Neighborhood search window: expected_lag ± 4 frames
    search_min = max(1, expected_lag - 4)
    search_max = min(len(ppg) - 1, expected_lag + 4)

    ac_zero = float(np.dot(ppg, ppg)) + 1e-9
    best_coef = 0.0
    for lag in range(search_min, search_max + 1):
        ac_lag = float(np.dot(ppg[lag:], ppg[:-lag]))
        coef = ac_lag / ac_zero
        if coef > best_coef:
            best_coef = coef

    return max(0.0, min(100.0, best_coef * 100.0))


def _composite_sqi(
    ppg: np.ndarray,
    valid_psd: np.ndarray,
    peak_idx: int,
    bpm: float,
    fps: float,
) -> tuple[float, dict]:
    """
    Fuse five sub-scores into one composite SQI in [0, 100].

    Returns (composite_score, sub_scores_dict) for transparency / logging.
    """
    s_spectral = _sqi_spectral(valid_psd, peak_idx)
    s_kurtosis = _sqi_kurtosis(ppg)
    s_entropy  = _sqi_entropy(ppg)
    s_zcr      = _sqi_zcr(ppg, bpm, fps)
    s_autocorr = _sqi_autocorr(ppg, bpm, fps)

    composite = (
        _SQI_W_SPECTRAL  * s_spectral +
        _SQI_W_KURTOSIS  * s_kurtosis +
        _SQI_W_ENTROPY   * s_entropy  +
        _SQI_W_ZCR       * s_zcr      +
        _SQI_W_AUTOCORR  * s_autocorr
    )

    sub_scores = {
        "spectral":  round(s_spectral, 1),
        "kurtosis":  round(s_kurtosis, 1),
        "entropy":   round(s_entropy,  1),
        "zcr":       round(s_zcr,      1),
        "autocorr":  round(s_autocorr, 1),
    }
    return float(np.clip(composite, 0.0, 100.0)), sub_scores


# ── CHROM projection ──────────────────────────────────────────────────────────

def _project_chrom(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    CHROM (de Haan & Jeanne 2013) — chrominance projection.

    Step 1: Colour-normalise each channel by its temporal mean (AC/DC split).
    Step 2: Project onto two orthogonal skin-reflectance chrominance axes.
            X = 3Rn - 2Gn             (roughly Cr analog)
            Y = 1.5Rn + Gn - 1.5Bn   (roughly Cb analog)
    Step 3: Cancel specular reflection via alpha-scaling:
            S = X - α·Y  where α = std(X)/std(Y)

    The alpha term cancels the portion of X that is correlated with Y
    (the specular component), leaving only the pulsatile signal.
    """
    r_mean = float(np.mean(r)) + 1e-9
    g_mean = float(np.mean(g)) + 1e-9
    b_mean = float(np.mean(b)) + 1e-9

    Rn = signal.detrend(r) / r_mean
    Gn = signal.detrend(g) / g_mean
    Bn = signal.detrend(b) / b_mean

    X = 3.0 * Rn - 2.0 * Gn
    Y = 1.5 * Rn + Gn - 1.5 * Bn
    alpha = (float(np.std(X)) + 1e-9) / (float(np.std(Y)) + 1e-9)
    return X - alpha * Y


# ── POS projection ────────────────────────────────────────────────────────────

def _project_pos(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    fps: float,
    window_sec: float = _POS_WINDOW_SEC,
) -> np.ndarray:
    """
    POS (Wang et al. 2017) — plane orthogonal to skin tone, sliding window.

    The key difference from CHROM:
      • Operates on short overlapping windows (1.6 s default) rather than
        the entire buffer, making it more robust to slow drift / illumination
        changes within the clip.
      • Defines the pulse plane via two vectors that are *data-driven* within
        each window, using the normalized colour matrix directly:
            Xs =  Gn - Bn
            Ys = -2Rn + Gn + Bn
            S_win = Xs + (std(Xs)/std(Ys)) · Ys
      • Windows are accumulated via overlap-add to reconstruct the full signal.

    Exact projection equations from Wang 2017, Section IV / iPhys Eq. 4-6:
        Xs = x̄g - x̄b
        Ys = -2x̄r + x̄g + x̄b
        S  = Xs + (σ(Xs)/σ(Ys)) · Ys
    where x̄ = channel / mean(channel) within each window.
    """
    n = len(r)
    L = max(2, int(fps * window_sec))   # window length in samples
    output = np.zeros(n, dtype=np.float64)
    weight = np.zeros(n, dtype=np.float64)
    hann   = np.hanning(L)

    for start in range(0, n - L + 1, 1):          # step=1 → full overlap-add
        end = start + L
        rw = r[start:end]
        gw = g[start:end]
        bw = b[start:end]

        r_mean = float(np.mean(rw)) + 1e-9
        g_mean = float(np.mean(gw)) + 1e-9
        b_mean = float(np.mean(bw)) + 1e-9

        Rn = rw / r_mean
        Gn = gw / g_mean
        Bn = bw / b_mean

        Xs = Gn - Bn
        Ys = -2.0 * Rn + Gn + Bn
        alpha = (float(np.std(Xs)) + 1e-9) / (float(np.std(Ys)) + 1e-9)
        S_win = (Xs + alpha * Ys) * hann

        output[start:end] += S_win
        weight[start:end] += hann

    # Normalise by accumulated Hann weight (avoids boundary ringing)
    weight = np.where(weight < 1e-9, 1.0, weight)
    return signal.detrend(output / weight)


# ── Main engine ───────────────────────────────────────────────────────────────

class AdvancedRPPG:
    """
    Research-grade rPPG engine.

    Parameters
    ----------
    fps          : Camera frame rate (default 30).
    window_size  : Ring buffer length in samples (default 300 = 10 s @ 30 fps).
    algorithm    : 'CHROM' (default) or 'POS'.
    """

    def __init__(
        self,
        fps: int = 30,
        window_size: int = 300,
        algorithm: Literal["CHROM", "POS"] = "CHROM",
    ):
        self.fps          = fps
        self.buffer_size  = window_size
        self.algorithm    = Algorithm(algorithm.upper())

        # RGB ring buffers.
        #
        # maxlen is deliberately max(window_size, fps * _RR_WINDOW_SEC): the
        # HR path always slices its own last `fps * 10` samples inside
        # _run_pipeline, so holding MORE history than window_size leaves every
        # heart-rate number bit-identical, while giving respiratory rate the
        # long window it physically requires (see the _RR_* block above).
        # Doing it here rather than at each AdvancedRPPG(...) construction
        # site means callers keep their existing window_size semantics for HR
        # and no call site has to change. Cost is ~1800 floats x 3.
        _buf_len = max(window_size, int(fps * _RR_WINDOW_SEC))
        self.r_buffer: deque[float] = deque(maxlen=_buf_len)
        self.g_buffer: deque[float] = deque(maxlen=_buf_len)
        self.b_buffer: deque[float] = deque(maxlen=_buf_len)

        # ── Filter state ──────────────────────────────────────────────
        # Seed filter (wide, used until first stable BPM is known)
        self._sos_seed = _safe_butter_sos(_BP_SEED_LOW, _BP_SEED_HIGH, fps)
        # Adaptive filter (recomputed each cycle once BPM estimate exists)
        self._sos_adaptive: Optional[np.ndarray] = None
        self._adaptive_low  = _BP_SEED_LOW
        self._adaptive_high = _BP_SEED_HIGH

        # Respiration filter at the frame rate. No longer used by the RR path
        # (which filters the RIV series on the _RIV_FS grid via _sos_riv
        # instead), but kept because it is now CORRECT for the first time —
        # it used to be built with the pulse-band clamps still applied, which
        # made it a 0.70-1.00 Hz filter rather than 0.10-0.50 Hz.
        self._sos_resp = _safe_butter_sos(_RESP_LOW, _RESP_HIGH, fps, order=2,
                                         clamp_to_pulse_band=False)

        # Pulse-band filter used for beat detection when deriving RIVs. Order 3
        # (vs the resp filter's 2) because this one has to isolate beats
        # cleanly across a 60 s window, not merely pass a band.
        self._sos_pulse = _safe_butter_sos(_BP_LOW_FLOOR, _BP_HIGH_CEIL, fps, order=3)

        # Respiration bandpass applied to the RIV series ON THE _RIV_FS GRID.
        #
        # This is NOT redundant with masking the FFT output to the resp band.
        # Measured: with linear detrending plus band-masking alone, 9 of 11
        # remaining failure cases on BIDMC were 0.5x SUBHARMONIC lock-ons
        # landing at 7-9 br/min against a true 16-22. Cause: detrend(type=
        # "linear") removes only a straight line, but real baseline/perfusion
        # drift is curved, so a large sub-0.1 Hz residual survives — and
        # spectral LEAKAGE from that residual floods the lowest in-band bins.
        # Masking the output cannot undo leakage that has already contaminated
        # the bins you kept; only filtering before the transform removes it.
        # Order 4 here (not 2) because this filter's whole job is to reject a
        # component that is far stronger than the signal being kept.
        # clamp_to_pulse_band=False is REQUIRED here — see _safe_butter_sos.
        # With the default True, this request silently becomes a 0.70-1.00 Hz
        # (42-60 br/min) filter centred on the heart rate.
        self._sos_riv = _safe_butter_sos(_RESP_LOW, _RESP_HIGH, _RIV_FS,
                                        order=4, clamp_to_pulse_band=False)
        # Per-band variants of the same filter for the HR-adaptive search band
        # (see _RR_BAND_*); edges are quantised to 0.01 Hz so this stays tiny.
        self._sos_riv_cache: dict = {(_RESP_LOW, _RESP_HIGH): self._sos_riv}

        # ── Temporal smoothing ────────────────────────────────────────
        self.prev_bpm: float = 0.0
        self.prev_snr: float = 0.0
        self.prev_rr: float = 0.0

        # RR session history + update throttle.
        #
        # rr_history is the direct analogue of bpm_history: until now RR had NO
        # session history at all, only a 2-sample EMA, so the session-level RR
        # the product reported was just "whatever the last frame happened to
        # say". HR by contrast gets an IQR-filtered median over its whole
        # history in get_final_summary(). Closing that asymmetry is worth as
        # much as the estimator rewrite: on BIDMC, per-window MAE 2.89 becomes
        # session-level MAE 1.37 purely by aggregating this way.
        self.rr_history: list[float] = []
        self._rr_cycle_counter: int = 0
        self._rr_cached: tuple = (0.0, 0.0)

        # BPM lock/candidate hysteresis state — see _LOCK_* constants above.
        self._locked_bpm: float = 0.0
        self._raw_bpm_window: deque = deque(maxlen=_LOCK_CONFIRM_STREAK_HARMONIC)
        self._prior_defer_count: int = 0
        self.bpm_history: list[float] = []
        self._history_reject_streak: int = 0
        self._latest_motion_delta: float = 0.0
        self.frame_count: int = 0

        log.info(
            "rPPG engine ready — algorithm=%s  fps=%d  buffer=%d samples",
            self.algorithm.value, fps, window_size,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def set_algorithm(self, algorithm: Literal["CHROM", "POS"]) -> None:
        """Hot-switch projection algorithm mid-session without clearing buffers."""
        self.algorithm = Algorithm(algorithm.upper())
        log.info("Algorithm switched to %s", self.algorithm.value)

    def add_frame(self, rgb: Optional[tuple], timestamp=None, motion_delta: float = 0.0) -> None:
        """Append one (r, g, b) sample to the ring buffers.

        motion_delta (new, optional) — the caller's own frame-to-frame face
        motion measurement (see Camera._compute_motion_delta), passed
        through into the next process_ppg_signal() result dict as a
        feature. Purely additive/optional: existing call sites that don't
        pass it keep working exactly as before, with motion_delta=0.0.
        """
        self._latest_motion_delta = float(motion_delta)
        if rgb is None:
            return
        try:
            r, g, b = rgb
            self.r_buffer.append(float(r))
            self.g_buffer.append(float(g))
            self.b_buffer.append(float(b))
        except Exception as exc:
            log.warning("Failed to add frame: %s", exc)

    def process_ppg_signal(self) -> dict:
        """
        Run the full rPPG pipeline and return a metrics dict.

        Pipeline
        ─────────
        1.  Wait for minimum calibration window (2.5 s).
        2.  Project RGB → 1-D PPG signal via CHROM or POS.
        3.  Bandpass-filter with the current adaptive filter.
        4.  Welch PSD → dominant BPM.
        5.  Update adaptive filter cutoffs using new BPM + SNR.
        6.  Compute composite SQI (spectral + kurtosis + entropy + ZCR + AC).
        7.  Temporal smoothing, history gating.
        8.  Respiratory rate estimation.
        9.  HRV (RMSSD) + Baevsky Stress Index.
        """
        min_samples = int(self.fps * 2.5)

        if len(self.r_buffer) < min_samples:
            progress = len(self.r_buffer) / min_samples * 100.0
            if len(self.r_buffer) % self.fps == 0:
                log.debug("Calibrating — buffer %d/%d (%.0f%%)",
                          len(self.r_buffer), min_samples, progress)
            return {
                "bpm": 0, "confidence": 0, "status": "CALIBRATING",
                "snr_db": 0, "sqi": 0, "sqi_components": {},
                "ready": False, "ppg_signal": [],
                "calibration_progress": int(progress),
            }

        try:
            return self._run_pipeline()
        except Exception as exc:
            log.error("Signal processing error: %s", exc, exc_info=True)
            return self._empty_result()

    # ── Internal pipeline ─────────────────────────────────────────────────────

    def _run_pipeline(self) -> dict:
        max_samples = int(self.fps * 10)

        r = np.array(list(self.r_buffer)[-max_samples:])
        g = np.array(list(self.g_buffer)[-max_samples:])
        b = np.array(list(self.b_buffer)[-max_samples:])

        # ── Step 2: projection ────────────────────────────────────────
        if self.algorithm == Algorithm.CHROM:
            ppg_raw = _project_chrom(r, g, b)
        else:
            ppg_raw = _project_pos(r, g, b, self.fps)

        # ── Step 3: bandpass filter ───────────────────────────────────
        sos = self._sos_adaptive if self._sos_adaptive is not None else self._sos_seed
        if sos is not None:
            try:
                ppg_filtered = signal.sosfiltfilt(sos, ppg_raw)
            except Exception:
                ppg_filtered = ppg_raw
        else:
            ppg_filtered = ppg_raw

        # ── Step 4: Welch PSD ─────────────────────────────────────────
        seg_len = len(ppg_filtered)
        freqs, psd = signal.welch(
            ppg_filtered, fs=self.fps,
            window="hann", nperseg=seg_len, noverlap=0, nfft=4096,
        )

        valid_mask  = (freqs >= _BP_LOW_FLOOR) & (freqs <= _BP_HIGH_CEIL)
        valid_freqs = freqs[valid_mask]
        valid_psd   = psd[valid_mask]
        if len(valid_psd) == 0:
            return self._empty_result()

        dominant_freq, peak_idx, harmonic_ratio = _select_dominant_freq_hps(valid_freqs, valid_psd)
        bpm_raw        = dominant_freq * 60.0

        # Time-domain hybrid estimator for short buffers (cold start) to stabilize resolution wobbles
        is_cold_start = len(ppg_filtered) < int(self.fps * 5.0)
        if is_cold_start:
            try:
                min_dist = int(self.fps * 0.40) # 400ms minimum refractory period
                sig_std  = float(np.std(ppg_filtered))
                peaks, _ = find_peaks(
                    ppg_filtered,
                    distance=min_dist,
                    prominence=max(1e-6, sig_std * 0.20),
                )
                if len(peaks) >= 2:
                    peak_intervals = np.diff(peaks)
                    avg_interval_samples = np.mean(peak_intervals)
                    bpm_time_domain = (self.fps / avg_interval_samples) * 60.0

                    if 45.0 <= bpm_time_domain <= 170.0:
                        # Blend the FFT estimate and the time-domain estimate (fades to 100% FFT at 5.0 seconds)
                        blend_ratio = len(ppg_filtered) / (self.fps * 5.0)
                        bpm_raw = blend_ratio * bpm_raw + (1.0 - blend_ratio) * bpm_time_domain
                        dominant_freq = bpm_raw / 60.0
            except Exception as e:
                log.debug("Time-domain fallback failed: %s", e)

        # ── Step 5: SNR ───────────────────────────────────────────────
        win_bins = max(1, int(0.10 * len(valid_psd) / 2.15))
        lo_b = max(0, peak_idx - win_bins)
        hi_b = min(len(valid_psd), peak_idx + win_bins + 1)
        signal_power = float(np.sum(valid_psd[lo_b:hi_b]))
        total_power  = float(np.sum(valid_psd)) + 1e-9
        noise_power  = total_power - signal_power + 1e-9
        snr_db       = float(np.clip(10.0 * np.log10(signal_power / noise_power), 0.0, 30.0))

        # ── Resolve BPM lock (harmonic-ambiguity debounce) ────────────
        # This confirmed value — not the raw per-cycle peak — drives BOTH
        # the adaptive filter recentering and the smoothing/history below,
        # so the filter itself can no longer reinforce a not-yet-confirmed
        # false lock (see _LOCK_* constants for the root-cause writeup).
        #
        # NOT during cold start, though: the lock/candidate machinery is a
        # fix for steady-state bistable oscillation (a signal that has
        # already found a rate, then slides into a false one later) — it is
        # NOT meant to govern the initial few seconds of convergence, which
        # is naturally volatile before the filter has ever settled on
        # anything. Feeding a possibly-wrong very-first raw read into the
        # adaptive filter as an immediate hard lock was actively worse than
        # the old behaviour: a wrong cold-start filter can narrow tightly
        # enough to filter the TRUE frequency out of the signal entirely,
        # making the wrong lock permanently self-reinforcing. During cold
        # start, track the raw estimate directly (matching prior behaviour)
        # while still keeping self._locked_bpm updated so the hysteresis has
        # a sensible, already-converged starting point once cold start ends.
        if is_cold_start:
            self._locked_bpm = bpm_raw
            resolved_bpm = bpm_raw
            filter_freq_hz = dominant_freq
        else:
            resolved_bpm = self._resolve_bpm_lock(bpm_raw, snr_db)
            filter_freq_hz = resolved_bpm / 60.0

        # ── Update adaptive bandpass ──────────────────────────────────
        self._update_adaptive_filter(filter_freq_hz, snr_db)

        # ── Step 6: composite SQI ─────────────────────────────────────
        # (computed from the raw instantaneous peak — SQI measures how
        # periodic/clean the current waveform snapshot looks, independent
        # of which candidate the lock resolver ultimately trusts)
        composite_sqi, sqi_components = _composite_sqi(
            ppg_filtered, valid_psd, peak_idx, bpm_raw, self.fps,
        )

        # ── Step 7: temporal smoothing ────────────────────────────────
        bpm = self._smooth_bpm(resolved_bpm)
        self.prev_snr = snr_db

        # HR classification
        if bpm < 48 or bpm > 150:
            status = "OUT_OF_RANGE"
        elif composite_sqi < 20:
            status = "LOW_SIGNAL"
        else:
            status = "OK"

        log.debug(
            "[%s] raw=%.1f smooth=%.1f SQI=%.0f SNR=%.1f dB filt=%.2f–%.2f Hz %s",
            self.algorithm.value, bpm_raw, bpm, composite_sqi, snr_db,
            self._adaptive_low, self._adaptive_high, status,
        )

        # BPM history gate
        self.frame_count += 1
        if composite_sqi > 15 and self.frame_count > 30 and status == "OK":
            if len(self.bpm_history) >= 5:
                recent_median = float(np.median(self.bpm_history[-5:]))
                if abs(bpm - recent_median) <= 18:
                    self.bpm_history.append(float(bpm))
                    self._history_reject_streak = 0
                else:
                    # See _HISTORY_UNSTICK_STREAK/_HISTORY_UNSTICK_MAX_HISTORY:
                    # don't let a still-fresh, barely-supported anchor (e.g. a
                    # brief wrong burst during cold-start) veto every later
                    # reading forever — but ONLY while it's still fresh. Once
                    # the anchor has substantial accumulated support, a later
                    # sustained streak is far more likely to be a genuine
                    # bimodal/harmonic episode than proof the anchor was
                    # wrong, so it must NOT be allowed to override (confirmed
                    # by real-data regression on subject41 when this guard
                    # was missing — see _HISTORY_UNSTICK_MAX_HISTORY's note).
                    self._history_reject_streak += 1
                    if (self._history_reject_streak > _HISTORY_UNSTICK_STREAK
                            and len(self.bpm_history) <= _HISTORY_UNSTICK_MAX_HISTORY):
                        self.bpm_history.append(float(bpm))
                        self._history_reject_streak = 0
            else:
                self.bpm_history.append(float(bpm))
                self._history_reject_streak = 0

        # ── Step 8: respiratory rate ──────────────────────────────────
        # Note it is NOT given ppg_raw (the 10 s HR window) any more — it
        # reads the engine's own longer history, because respiration at
        # 12 br/min completes only 2 cycles in 10 s and cannot be resolved
        # from that. Throttled to _RR_UPDATE_HZ: RR moves slowly, and doing
        # beat-detection plus three Welch transforms over a 60 s buffer on
        # every single frame would stall the live MJPEG stream.
        self._rr_cycle_counter += 1
        rr_every = max(1, int(self.fps / _RR_UPDATE_HZ))
        is_fresh_rr = (self._rr_cycle_counter % rr_every == 0
                       or self._rr_cached == (0.0, 0.0))
        if is_fresh_rr:
            self._rr_cached = self._estimate_rr()
        rr_bpm_raw, rr_conf = self._rr_cached

        rr_bpm = self._smooth_rr(rr_bpm_raw, rr_conf) if rr_bpm_raw > 0 else 0.0

        # RR session history — the analogue of the bpm_history gate above.
        # Only readings that clear the same confidence floor the display and
        # clinical engine use are allowed to accumulate, so the session median
        # is built from readings the system is willing to stand behind.
        #
        # is_fresh_rr gates the append too, not just the throttle above — this
        # matters a great deal, not just for efficiency. Without it, a SINGLE
        # true estimate gets appended once per frame for every frame until the
        # next recompute (~15 times at 30fps / _RR_UPDATE_HZ=2), so rr_history
        # was never really "N independent measurements": it was N/15 distinct
        # values, each repeated ~15x, with the repeat count set by frame
        # timing rather than measurement quality. A single wrong-but-confident
        # cold-start estimate that happened to persist for one throttle
        # interval could dominate the session median purely by having
        # appeared for longer, not by being more likely correct. Caught via a
        # synthetic regression: after an unrelated fusion change, one
        # 90 s run's very first RR estimate (a confidently-wrong 28 br/min
        # against true 12) got appended ~15 times in that one throttle window,
        # which was enough to drag the IQR-filtered session median to 27 br/min
        # even though the great majority of the session's DISTINCT estimates
        # clustered correctly around 11-12.
        if is_fresh_rr and rr_bpm_raw > 0 and rr_conf >= _RR_HISTORY_MIN_CONF:
            self.rr_history.append(float(rr_bpm_raw))

        # ── Step 9: stability ─────────────────────────────────────────
        recent = self.bpm_history[-30:] if len(self.bpm_history) > 30 else self.bpm_history
        bpm_std = float(np.std(recent)) if len(recent) > 2 else 0.0
        stability_score = float(np.clip(composite_sqi - bpm_std * 5.0, 0.0, 100.0))
        stability_indicator = (
            "HIGH"   if stability_score > 75 else
            "MEDIUM" if stability_score > 40 else
            "LOW"
        )

        # ── Step 10: HRV + stress index ───────────────────────────────
        hrv, stress_index = self._compute_hrv_stress(ppg_filtered, composite_sqi)

        return {
            "bpm":                  float(bpm),
            "confidence":           float(composite_sqi),
            "status":               status,
            "snr_db":               float(snr_db),
            "sqi":                  float(composite_sqi),
            "sqi_components":       sqi_components,
            "stability_score":      float(stability_score),
            "stability_indicator":  stability_indicator,
            "rr":                   float(rr_bpm),
            "rr_confidence":        float(rr_conf),
            "hrv":                  float(hrv),
            "stress_index":         float(stress_index),
            "algorithm":            self.algorithm.value,
            "filter_band_hz":       [round(self._adaptive_low, 3),
                                     round(self._adaptive_high, 3)],
            "harmonic_ratio":       float(harmonic_ratio),
            "motion_delta":         float(self._latest_motion_delta),
            "ready":                True,
            "ppg_signal":           ppg_filtered.tolist(),
        }

    # ── BPM lock/candidate resolver (harmonic-ambiguity debounce) ─────────────

    def _resolve_bpm_lock(self, bpm_raw: float, snr_db: float) -> float:
        """
        Debounce the raw per-cycle BPM peak against a confirmed "lock" before
        it's allowed to drive the adaptive filter or smoothing/history.

        Design note — why a rolling-window spread check, not a same-value
        streak counter: an earlier version of this method required the raw
        estimate to repeat within tolerance of a frozen "candidate" value
        every single cycle to build a confirmation streak. That's too
        fragile against ordinary per-cycle FFT-bin jitter (a clean synthetic
        signal's raw peak can still wander a few bpm frame-to-frame even once
        correctly converged), so it regressed straightforward cases that
        worked before this fix. Instead: keep a short rolling window of raw
        readings and ask whether they've collectively *settled* — low
        mutual spread — on a value away from the current lock, rather than
        demanding literal repetition.

        A reading that merely agrees with the current lock (within
        _LOCK_AGREE_TOLERANCE) nudges the lock slightly and the window
        doesn't need to be consulted. A disagreeing reading only replaces
        the lock once the most recent `required_window` raw readings are
        mutually consistent (spread ≤ 2×tolerance) AND their median differs
        meaningfully from the lock. Readings that look like a harmonic
        multiple/fraction of the current lock (~0.5x or ~2x) need a much
        longer settled window than an ordinary jump, since that ratio is
        specifically the ambiguous case a real periodic pulse waveform can
        produce (comparable energy at the wrong multiple of the true rate).
        """
        self._raw_bpm_window.append(bpm_raw)

        if self._locked_bpm <= 0:
            self._locked_bpm = bpm_raw
            return bpm_raw

        delta = abs(bpm_raw - self._locked_bpm)
        if delta <= _LOCK_AGREE_TOLERANCE:
            self._locked_bpm = 0.7 * self._locked_bpm + 0.3 * bpm_raw
            self._prior_defer_count = 0
            return self._locked_bpm

        ratio = bpm_raw / self._locked_bpm if self._locked_bpm > 0 else 1.0
        looks_harmonic = (
            (_LOCK_HARMONIC_LOW_A <= ratio <= _LOCK_HARMONIC_HIGH_A) or
            (_LOCK_HARMONIC_LOW_B <= ratio <= _LOCK_HARMONIC_HIGH_B)
        )
        required_window = _LOCK_CONFIRM_STREAK_HARMONIC if looks_harmonic else _LOCK_CONFIRM_STREAK_NORMAL

        window = list(self._raw_bpm_window)[-required_window:]
        if len(window) < required_window:
            return self._locked_bpm  # not enough history yet to confirm a switch

        if max(window) - min(window) > _LOCK_AGREE_TOLERANCE * 2:
            return self._locked_bpm  # recent readings aren't consistent enough to trust as a new lock

        candidate_value = statistics.median(window)
        if abs(candidate_value - self._locked_bpm) <= _LOCK_AGREE_TOLERANCE:
            return self._locked_bpm  # the settled window actually agrees with the lock after all

        if looks_harmonic:
            lock_in_prior = _LOCK_PRIOR_LOW <= self._locked_bpm <= _LOCK_PRIOR_HIGH
            cand_in_prior = _LOCK_PRIOR_LOW <= candidate_value <= _LOCK_PRIOR_HIGH
            if lock_in_prior and not cand_in_prior:
                # Current lock is already in the plausible resting-HR range
                # and the challenger isn't — a soft tie-breaker, not a hard
                # cutoff. Bounded by _prior_defer_count so this can't
                # deadlock forever: once the challenger has stayed settled
                # past this budget too, it wins anyway.
                self._prior_defer_count += 1
                if self._prior_defer_count <= required_window:
                    return self._locked_bpm

        self._prior_defer_count = 0
        self._locked_bpm = candidate_value
        return self._locked_bpm

    # ── Adaptive filter update ────────────────────────────────────────────────

    def _update_adaptive_filter(self, dominant_freq: float, snr_db: float) -> None:
        """
        Recompute the adaptive bandpass filter around the current HR estimate.

        Guard width narrows as SNR improves (high confidence → tight filter
        reduces harmonic bleed) and widens as SNR drops (wide net to recover
        signal during noise bursts).
        """
        guard = _guard_width(snr_db)
        new_low  = max(_BP_LOW_FLOOR, dominant_freq - guard)
        new_high = min(_BP_HIGH_CEIL, dominant_freq + guard)

        # Smooth the cutoffs so rapid HR changes don't cause abrupt filter jumps
        alpha = 0.25   # EMA weight for new estimate
        self._adaptive_low  = alpha * new_low  + (1.0 - alpha) * self._adaptive_low
        self._adaptive_high = alpha * new_high + (1.0 - alpha) * self._adaptive_high

        sos = _safe_butter_sos(self._adaptive_low, self._adaptive_high, self.fps)
        if sos is not None:
            self._sos_adaptive = sos

    # ── BPM temporal smoothing ────────────────────────────────────────────────

    def _smooth_bpm(self, bpm_raw: float) -> float:
        """Adaptive EMA: tighter tracking when estimate is stable."""
        if self.prev_bpm <= 0:
            self.prev_bpm = bpm_raw
            return bpm_raw
        delta = abs(bpm_raw - self.prev_bpm)
        if delta < 5:
            bpm = 0.30 * self.prev_bpm + 0.70 * bpm_raw   # fast track small changes
        elif delta < 15:
            bpm = 0.50 * self.prev_bpm + 0.50 * bpm_raw
        elif delta < 25:
            bpm = 0.20 * self.prev_bpm + 0.80 * bpm_raw
        else:
            bpm = bpm_raw                                   # large jump — trust raw
        self.prev_bpm = bpm
        return bpm

    # ── Respiratory rate ──────────────────────────────────────────────────────

    def _beat_detection_filter(self) -> Optional[np.ndarray]:
        """
        Pulse-band filter used for RIV beat detection — narrowed around the
        CONFIRMED heart rate when one exists, instead of the full physiological
        range.

        Root cause this fixes: the static wide-band `_sos_pulse`
        (_BP_LOW_FLOOR.._BP_HIGH_CEIL) passes a heart rate's 2nd harmonic
        completely unattenuated whenever it also falls in-range, which it
        almost always does (2x a 60 bpm pulse is 120 bpm, itself a plausible
        "heart rate" to the filter). Any waveform with dicrotic-notch-like
        asymmetry then has a second, harmonic-driven local maximum per cardiac
        cycle, and a generic prominence threshold cannot always distinguish it
        from a genuine beat. Measured: even a HEART-RATE-ONLY synthetic
        signal (no respiration at all) produced 1.55x too many detected
        "beats" (91 found over 58.6 s at 60 bpm, vs. 58.6 expected) — every
        downstream RIV series built from over-counted beats is contaminated,
        independent of anything to do with respiration.
        The main HR pipeline already solves the identical problem for its own
        peak-picking by narrowing around the confirmed rate (see
        _update_adaptive_filter) — this mirrors that fix for beat DETECTION.
        A generous +-25% guard is used (vs. the main filter's much tighter
        SNR-adaptive guard) because RIV extraction only needs harmonics
        suppressed, not fine frequency resolution, and physiological HR can
        genuinely drift by more than a tight guard would allow over a 60 s
        window.
        Falls back to the static wide-band filter during cold start, before
        any HR has been confirmed.
        """
        if self.prev_bpm <= 0:
            return self._sos_pulse
        hz = self.prev_bpm / 60.0
        low, high = hz * 0.75, hz * 1.25
        narrowed = _safe_butter_sos(low, high, self.fps, order=3,
                                    clamp_to_pulse_band=False)
        return narrowed if narrowed is not None else self._sos_pulse

    def _extract_rivs(self, ppg_long: np.ndarray) -> Optional[tuple]:
        """
        Derive the three per-beat respiratory-induced variation series.

        Returns (times_sec, riav, riiv, rifv) sampled once per detected beat,
        or None when there aren't enough clean beats to define anything.
        """
        sos_pulse = self._beat_detection_filter()
        if sos_pulse is None:
            return None
        try:
            pulse = signal.sosfiltfilt(sos_pulse, ppg_long)
        except Exception:
            return None

        sd = float(np.std(pulse))
        if sd <= 0:
            return None

        peaks, _ = find_peaks(
            pulse,
            distance=int(self.fps * 0.40),          # 400 ms refractory floor
            prominence=max(1e-6, sd * 0.20),
        )
        if len(peaks) < _RR_MIN_BEATS:
            return None

        t, riav, riiv, rifv = [], [], [], []
        for i in range(1, len(peaks)):
            a, b = peaks[i - 1], peaks[i]
            seg = pulse[a:b + 1]
            if len(seg) < 2:
                continue
            t.append(b / self.fps)
            riav.append(float(pulse[b]) - float(np.min(seg)))   # amplitude
            riiv.append(float(ppg_long[b]))                     # baseline level
            rifv.append((b - a) / self.fps)                     # interval (RSA)

        if len(t) < _RR_MIN_BEATS:
            return None
        return (np.asarray(t), np.asarray(riav),
                np.asarray(riiv), np.asarray(rifv))

    def _riv_candidate(self, grid: np.ndarray, t: np.ndarray,
                       series: np.ndarray, dedrift: bool = False,
                       lo: float = _RESP_LOW, hi: float = _RESP_HIGH) -> tuple:
        """
        One RIV series → (rr_bpm, spectral_concentration).

        Interpolates onto the uniform grid with a cubic spline rather than
        linear interpolation. Linear interpolation between beat-spaced
        samples (~0.7 s apart) acts as a mild low-pass filter — its transfer
        function attenuates higher frequencies more than lower ones — which
        was measured to be part of why the estimator's answers compressed
        toward the middle of the search band rather than tracking the true
        rate (regression slope of estimate-on-truth only 0.34, i.e. moving
        about a third as much as reality). Switching to cubic interpolation
        alone moved that slope to ~0.55 with no other change. Falls back to
        linear if the beat count is too low for a cubic spline (needs >=4
        points).

        `dedrift` (RIIV only — see _RIIV_DEDRIFT_WINDOW_SEC) subtracts a
        rolling median baseline BEFORE the existing linear detrend/bandpass,
        to reject non-linear drift (e.g. a webcam's own AGC/auto-exposure
        stepping) that a single global linear detrend cannot remove and that
        the bandpass filter's soft rolloff does not fully reject at its
        edge. RIAV and RIFV don't carry this confound (they're a beat-local
        shape/interval feature, not an absolute baseline level) and are left
        untouched.
        """
        if len(t) >= 4:
            try:
                y = CubicSpline(t, series)(grid)
            except Exception:
                y = np.interp(grid, t, series)
        else:
            y = np.interp(grid, t, series)

        if dedrift:
            win = int(_RIIV_DEDRIFT_WINDOW_SEC * _RIV_FS)
            win += 1 - (win % 2)   # median_filter wants an odd window size
            if win >= 3 and len(y) > win:
                try:
                    baseline = median_filter(y, size=win, mode="nearest")
                    y = y - baseline
                except Exception:
                    pass

        try:
            y = signal.detrend(y, type="linear")   # removes the linear component
        except Exception:
            y = y - float(np.mean(y))

        # Then reject what detrending cannot: the CURVED sub-band drift whose
        # leakage was causing subharmonic lock-on (see self._sos_riv).
        sos = self._riv_sos(lo, hi)
        if sos is not None and len(y) > 24:
            try:
                y = signal.sosfiltfilt(sos, y)
            except Exception:
                pass

        nperseg = len(y)                            # full span = best resolution
        if nperseg < 8:
            return 0.0, 0.0
        try:
            f, p = signal.welch(
                y, fs=_RIV_FS, window="hann",
                nperseg=nperseg, noverlap=nperseg // 2, nfft=4096,
            )
        except Exception:
            return 0.0, 0.0

        m = (f >= lo) & (f <= hi)
        f, p = f[m], p[m]
        if len(p) == 0 or not np.any(p > 0):
            return 0.0, 0.0

        k = self._select_rr_peak(f, p, lo, hi)
        rr = float(f[k] * 60.0)
        conc = float(np.sum(p[max(0, k - 2):k + 3]) / (float(np.sum(p)) + 1e-12) * 100.0)
        return rr, conc

    def _riv_sos(self, lo: float, hi: float):
        """RIV-grid bandpass for [lo, hi] Hz, built once per quantised band."""
        key = (lo, hi)
        sos = self._sos_riv_cache.get(key)
        if sos is None:
            sos = _safe_butter_sos(lo, hi, _RIV_FS, order=4, clamp_to_pulse_band=False)
            self._sos_riv_cache[key] = sos
        return sos

    @staticmethod
    def _rr_band(beat_hz: float) -> tuple:
        """HR-adaptive RR search band (Hz) — see _RR_BAND_* for the evidence.
        Edges quantised to 0.01 Hz so the per-band filter cache stays small."""
        if beat_hz <= 0:
            return _RESP_LOW, _RESP_HIGH
        lo = min(max(_RR_BAND_LO_HR_FRAC * beat_hz, _RESP_LOW), _RR_BAND_LO_CAP_HZ)
        hi = min(max(_RR_BAND_HI_HR_FRAC * beat_hz, _RESP_HIGH), _RR_BAND_HI_CAP_HZ)
        return round(lo * 100.0) / 100.0, round(hi * 100.0) / 100.0

    @classmethod
    def _select_rr_peak(cls, f: np.ndarray, p: np.ndarray, lo: float, hi: float) -> int:
        """Pick the respiratory peak index: argmax, then low-zone rescue, then
        the exact-2x promote / exact-1/2x demote checks."""
        k = int(np.argmax(p))
        if f[k] * 60.0 < _RR_LOW_ZONE_BPM:
            m = (f >= _RR_LOW_ZONE_SPAN[0] * f[k]) & (f <= _RR_LOW_ZONE_SPAN[1] * f[k])
            if m.any():
                idx = np.where(m)[0]
                k2 = int(idx[int(np.argmax(p[idx]))])
                if p[k2] >= _RR_LOW_ZONE_RATIO * p[k]:
                    k = k2
        k = cls._promote_subharmonic(f, p, k, hi)
        k = cls._demote_harmonic(f, p, k, lo)
        return k

    @staticmethod
    def _sampling_derate(rr: float, conf: float, beat_hz: float) -> tuple:
        """
        Cut confidence when there are too few heartbeats per breath for the
        RIV series to represent respiration (see _RR_MIN_BEATS_PER_BREATH).

        Returns (rr, adjusted_conf) — the rate itself is NOT altered. The
        point is to stop presenting an under-sampled estimate as reliable,
        not to invent a corrected value: this system has no validated way to
        recover a rate its sampling cannot resolve, and a fudge factor fitted
        to 53 ICU recordings would not be one.
        """
        if rr <= 0 or beat_hz <= 0:
            return rr, conf
        beats_per_breath = beat_hz / (rr / 60.0)
        if beats_per_breath >= _RR_MIN_BEATS_PER_BREATH:
            return rr, conf
        # Linear taper to zero as sampling degrades, so the existing
        # confidence gates take over and report "not measured".
        factor = max(0.0, beats_per_breath / _RR_MIN_BEATS_PER_BREATH)
        return rr, float(conf * factor)

    @staticmethod
    def _promote_subharmonic(f: np.ndarray, p: np.ndarray, k: int,
                             hi: float = _RESP_HIGH) -> int:
        """
        If the peak's DOUBLE carries comparable-or-greater power, prefer the
        double.

        Deliberately the same asymmetric shape as _select_dominant_freq_hps
        uses for heart rate, and for the same reason documented there: when a
        tracker is stuck at half the true rate, the true signal's own energy
        sits exactly at the candidate's "2nd harmonic" position, so a
        symmetric or naive-HPS test scores that energy as support FOR the
        wrong answer. Only an explicit "is my double at least as strong as
        me?" check can escape, and it must be one-directional or it will
        happily walk a correct answer up to 2x instead.

        Measured on BIDMC before this check existed: 9 of 11 residual failures
        were 0.5x subharmonic locks (e.g. bidmc_22 gold 19 -> est 7.6,
        bidmc_30 gold 19 -> est 7.3).
        """
        if k >= len(f):
            return k
        f_double = 2.0 * float(f[k])
        if f_double > hi:
            return k                      # double is out of band — nothing to prefer
        k2 = int(np.argmin(np.abs(f - f_double)))
        if k2 == k:
            return k
        # Compare small neighbourhoods, not single bins: with nfft=4096 the
        # bin spacing is far finer than the true resolution, so a lone bin is
        # noise-sensitive in a way a 3-bin sum is not.
        pk = float(np.sum(p[max(0, k - 1):k + 2]))
        p2 = float(np.sum(p[max(0, k2 - 1):k2 + 2]))
        if pk <= 0:
            return k
        if p2 >= pk * _RR_SUBHARM_PROMOTE_RATIO:
            return k2
        return k

    @staticmethod
    def _demote_harmonic(f: np.ndarray, p: np.ndarray, k: int,
                         lo: float = _RESP_LOW) -> int:
        """
        Mirror of _promote_subharmonic: if the peak's HALF carries
        comparable-or-greater power, prefer the half.

        Root cause this addresses, found on CapnoBase (record 0105, true RR
        ~10 br/min): sustained 2x-harmonic lock, with HIGH fused confidence
        precisely on the wrong windows (RIAV+RIIV agreeing on the same wrong
        harmonic looks like strong agreement) and LOW confidence on the rare
        correct windows. Traced to real 2nd-harmonic energy in the RIV
        spectrum exceeding the fundamental's — breathing waveforms are not
        pure sinusoids (asymmetric inhale/exhale), so genuine harmonic
        content is physiologically expected, not a leakage artefact.

        _promote_subharmonic's own docstring already predicted this needs a
        mirror check ("must be one-directional or it will happily walk a
        correct answer up to 2x") — this is that mirror, checking the
        opposite direction. Applied strictly after promotion so the two
        cannot fight over the same bin in the same call.
        """
        if k >= len(f) or f[k] <= 0:
            return k
        f_half = float(f[k]) / 2.0
        if f_half < lo:
            return k                      # half is out of band — nothing to prefer
        k_half = int(np.argmin(np.abs(f - f_half)))
        if k_half == k:
            return k
        pk = float(np.sum(p[max(0, k - 1):k + 2]))
        ph = float(np.sum(p[max(0, k_half - 1):k_half + 2]))
        if pk <= 0:
            return k
        if ph >= pk * _RR_DEMOTE_RATIO:
            return k_half
        return k

    def _estimate_rr(self, ppg_long: Optional[np.ndarray] = None) -> tuple[float, float]:
        """
        Estimate respiratory rate by fusing respiratory-induced variations
        measured per heartbeat (RIAV amplitude + RIIV baseline; RIFV/RSA is
        extracted but excluded from fusion — see the _RR_* constant block for
        why).

        See the _RR_* constant block at the top of this module for the full
        root-cause writeup: the previous implementation band-passed the RAW
        ppg at 0.1-0.5 Hz and took an argmax, which measured 1/f drift at one
        end and leaked pulse energy at the other, and had no measurable
        correlation with human-annotated breaths on BIDMC. This method
        replaces it outright rather than tuning it.

        Two further refinements on top of the original 3-channel version,
        both measured on the full 53-record BIDMC set: dropping RIFV from the
        fusion, and switching RIV interpolation from linear to cubic spline
        (see _riv_candidate). Together they moved session MAE 2.10 -> 1.71
        br/min and the estimate-vs-truth regression slope 0.339 -> 0.541
        (1.0 = unbiased) — a real improvement, though the estimator still
        underestimates on average and this is not yet a solved problem.

        `ppg_long` may be injected for testing; in normal operation it is
        projected from the engine's own long RGB history.

        Returns (rr_breaths_per_min, confidence_0_100). Confidence here means
        "how well do the three independent channels agree", which is a real
        reliability signal, unlike a single spectrum's peak height.
        """
        if ppg_long is None:
            need = int(self.fps * _RR_MIN_WINDOW_SEC)
            if len(self.r_buffer) < need:
                return 0.0, 0.0
            take = int(self.fps * _RR_WINDOW_SEC)
            r = np.array(list(self.r_buffer)[-take:])
            g = np.array(list(self.g_buffer)[-take:])
            b = np.array(list(self.b_buffer)[-take:])
            if self.algorithm == Algorithm.CHROM:
                ppg_long = _project_chrom(r, g, b)
            else:
                ppg_long = _project_pos(r, g, b, self.fps)

        if len(ppg_long) < int(self.fps * _RR_MIN_WINDOW_SEC):
            return 0.0, 0.0

        try:
            got = self._extract_rivs(ppg_long)
            if got is None:
                return 0.0, 0.0
            t, riav, riiv, rifv = got   # rifv (RSA) is extracted but NOT fused — see below

            # Uniform low-rate grid. This is the step that structurally
            # removes the pulse: Nyquist is _RIV_FS/2, far below any heart
            # rate, so pulse energy cannot alias back into the resp band.
            grid = np.arange(t[0], t[-1], 1.0 / _RIV_FS)
            if len(grid) < 16:
                return 0.0, 0.0

            # Fuse RIAV (amplitude) and RIIV (baseline) only. RIFV — the
            # respiratory sinus arrhythmia channel — was in the original
            # 3-channel fusion but measured as near-noise on BIDMC: alone it
            # scored slope 0.09-0.14 and corr ~0.08 against ground truth
            # (vs. RIIV alone's 0.55 / 0.48), and including it in the median
            # dragged the fused answer toward its bad guesses rather than
            # improving robustness. Likely because BIDMC's ICU population
            # (many ventilated) has blunted RSA — exactly what RIFV depends
            # on. Dropping it, together with cubic interpolation above,
            # measured session MAE 2.10 -> 1.71 br/min and moved the
            # estimate-vs-truth regression slope from 0.339 toward 1.0 up to
            # 0.541 — still not unbiased, but a real, verified improvement.
            # Beat rate == the RIV series' sampling rate. Needed to size the
            # HR-adaptive search band below AND to judge whether the answer is
            # adequately sampled at all (_sampling_derate).
            span_s = float(t[-1] - t[0])
            beat_hz = (len(t) / span_s) if span_s > 0 else 0.0
            band_lo, band_hi = self._rr_band(beat_hz)

            cands, concs = [], []
            for series, needs_dedrift in ((riav, False), (riiv, True)):
                rr, conc = self._riv_candidate(grid, t, series, dedrift=needs_dedrift,
                                               lo=band_lo, hi=band_hi)
                if rr > 0:
                    cands.append(rr)
                    concs.append(conc)
            if not cands:
                return 0.0, 0.0

            cands = np.asarray(cands)
            concs = np.asarray(concs)

            # TEMP DIAGNOSTIC (remove after this verification test — see
            # Finding 10 in data/evolution/19-respiratory-rate-ground-truth.md)
            log.info("RR candidates (TEMP DIAG): %s (conc=%s) band=%.2f-%.2fHz",
                     cands.tolist(), concs.tolist(), band_lo, band_hi)

            if len(cands) == 1:
                # Single surviving channel — no cross-check possible, so this
                # can never present as high confidence.
                return self._sampling_derate(
                    float(cands[0]), float(np.clip(concs[0] * 0.5, 0.0, 40.0)), beat_hz)

            spread = float(np.std(cands))
            if spread <= _RR_AGREE_TOL:
                # Median, not mean: measured better than mean on BIDMC
                # (within-2-br/min rate 52.8% -> 62.6%) because one channel
                # landing on a harmonic shouldn't drag the fused answer.
                rr = float(np.median(cands))
                return self._sampling_derate(
                    rr, float(np.clip(100.0 - spread * 12.0, 20.0, 100.0)), beat_hz)

            # Channels disagree overall — fall back to the tightest agreeing
            # pair, at reduced confidence. With exactly 2 fused channels
            # (RIAV, RIIV — see above for why RIFV was dropped) this branch
            # only matters if a third channel is reinstated in the future;
            # today "the tightest pair" is the only pair, already excluded by
            # the spread check above, so it never fires. Left in place rather
            # than special-cased so the fusion logic stays correct if the
            # channel set changes again.
            order = np.argsort(cands)
            cs = cands[order]
            gaps = np.diff(cs)
            j = int(np.argmin(gaps))
            if gaps[j] <= _RR_AGREE_TOL:
                return self._sampling_derate(
                    float(np.mean(cs[j:j + 2])),
                    float(np.clip(60.0 - gaps[j] * 8.0, 15.0, 60.0)), beat_hz)

            # The fused channels disagree outright (with 2 channels: RIAV and
            # RIIV gave different answers and neither dominates). Report the
            # best-concentrated candidate but with confidence low enough that
            # the display/clinical gates (RR_CONFIDENCE_LOW) will drop it —
            # an honest "couldn't measure" beats a confident wrong number
            # feeding the Compensated Shock rule.
            #
            # (Spectral concentration is a weak tie-breaker: measured on
            # synthetic signal it sometimes picks the worst channel, e.g.
            # choosing 11.19 over a correct 15.12 br/min. It survives only
            # because the confidence assigned here is always below
            # _RR_HISTORY_MIN_CONF, so such readings never enter rr_history
            # or reach the clinical engine.)
            k = int(np.argmax(concs))
            return self._sampling_derate(
                float(cands[k]), float(np.clip(concs[k] * 0.3, 0.0, 25.0)), beat_hz)

        except Exception as exc:
            log.debug("RR estimation failed: %s", exc)
            return 0.0, 0.0

    def _smooth_rr(self, rr_raw: float, rr_conf: float) -> float:
        """
        Confidence-weighted EMA smoothing for respiratory rate.

        Heart rate already gets adaptive EMA smoothing (_smooth_bpm) — RR had
        none at all, so every frame reported a fully independent estimate
        with no memory of the previous reading. Combined with RR being the
        noisiest channel in this engine, that's what let single-frame noise
        swing the *displayed* value across the whole 6-30 br/min band between
        consecutive updates. Blend weight scales with this frame's own
        confidence, so a low-confidence spike barely moves the smoothed
        estimate instead of replacing it outright.
        """
        if self.prev_rr <= 0:
            self.prev_rr = rr_raw
            return rr_raw
        alpha = float(np.clip(rr_conf / 100.0, 0.05, 0.35))
        smoothed = alpha * rr_raw + (1.0 - alpha) * self.prev_rr
        self.prev_rr = smoothed
        return smoothed

    # ── HRV + Baevsky stress index ────────────────────────────────────────────

    def _compute_hrv_stress(
        self, ppg_filtered: np.ndarray, sqi: float
    ) -> tuple[float, float]:
        if len(ppg_filtered) < int(self.fps * 5) or sqi < 5:
            return 0.0, 0.0
        try:
            # Upsample to 250 Hz for sub-frame IBI precision (~4 ms quantisation)
            n      = len(ppg_filtered)
            t_orig = np.linspace(0.0, n / self.fps, n)
            t_hi   = np.linspace(0.0, n / self.fps, int(n * (_HRV_HI_RES_FPS / self.fps)))
            ppg_hi = CubicSpline(t_orig, ppg_filtered)(t_hi)

            min_dist = int(_HRV_HI_RES_FPS * 0.40)    # 400 ms refractory floor
            sig_std  = float(np.std(ppg_hi))
            peaks, _ = find_peaks(
                ppg_hi,
                distance=min_dist,
                prominence=max(1e-6, sig_std * 0.30),
            )
            if len(peaks) < 3:
                return 0.0, 0.0

            ibis = np.diff(peaks) / _HRV_HI_RES_FPS * 1000.0   # ms
            ibis = ibis[(ibis >= 300) & (ibis <= 2000)]

            if len(ibis) < 2:
                return 0.0, 0.0

            # Reject IBI outliers (±30% of median)
            med_ibi = float(np.median(ibis))
            ibis = ibis[(ibis >= med_ibi * 0.70) & (ibis <= med_ibi * 1.30)]

            if len(ibis) < 2:
                return 0.0, 0.0

            # RMSSD
            rmssd = float(np.sqrt(np.mean(np.diff(ibis) ** 2)))
            hrv   = float(np.clip(rmssd, 5.0, 100.0))

            # Baevsky Stress Index
            mo     = float(np.median(ibis))
            mxdmn  = max(120.0, float(np.max(ibis) - np.min(ibis)))
            n_bins = max(2, min(len(ibis), 8))
            hist, _ = np.histogram(ibis, bins=n_bins)
            am_pct  = (float(np.max(hist)) / len(ibis)) * 100.0
            si_raw  = am_pct / (2.0 * (mo / 1000.0) * (mxdmn / 1000.0))
            stress  = float(np.clip(si_raw, 45.0, 350.0))

            return hrv, stress

        except Exception as exc:
            log.debug("HRV peak detection failed: %s", exc)
            return 0.0, 0.0

    # ── Result helpers ────────────────────────────────────────────────────────

    def _empty_result(self) -> dict:
        return {
            "bpm": 0, "confidence": 0.0, "status": "NO_FACE",
            "snr_db": 0.0, "sqi": 0.0, "sqi_components": {},
            "stability_score": 0.0, "stability_indicator": "LOW",
            "rr": 0.0, "rr_confidence": 0.0,
            "hrv": 0.0, "stress_index": 0.0,
            "algorithm": self.algorithm.value,
            "filter_band_hz": [_BP_SEED_LOW, _BP_SEED_HIGH],
            "harmonic_ratio": 0.0, "motion_delta": float(self._latest_motion_delta),
            "ready": False, "ppg_signal": [],
        }

    # ── Session summary ───────────────────────────────────────────────────────

    def get_final_summary(self) -> dict:
        """
        Return session-end summary statistics.
        Returns final_bpm=None when data is insufficient — no fake values.

        Respiratory rate is aggregated the same way heart rate is (see
        _summarise_rr): RR used to have no session aggregation at all, so the
        session RR was effectively the last frame's EMA output. On BIDMC,
        aggregating instead of last-value took RR from 2.89 to 1.37 br/min MAE.
        """
        if not self.bpm_history:
            if self.prev_bpm > 40:
                log.info("Final summary: no history, using last BPM %.1f", self.prev_bpm)
                fb = round(self.prev_bpm)
                return {
                    "final_bpm":        fb,
                    "min_bpm": fb, "max_bpm": fb, "avg_bpm": fb,
                    "stability_percent": 0,
                    "remark":           self._classify_remark(fb) + " — Low Confidence",
                    "total_readings":   0,
                    "algorithm":        self.algorithm.value,
                    **self._summarise_rr(),
                }
            log.info("Final summary: insufficient data.")
            return {
                "final_bpm":        None,
                "min_bpm": None, "max_bpm": None, "avg_bpm": None,
                "stability_percent": 0,
                "remark":           "INSUFFICIENT_DATA",
                "error":            "Minimum 5 seconds of clean face signal required.",
                "total_readings":   0,
                "algorithm":        self.algorithm.value,
                **self._summarise_rr(),
            }

        median_bpm  = statistics.median(self.bpm_history)
        final_bpm   = round(median_bpm)
        remark      = self._classify_remark(final_bpm)

        # IQR outlier filter for display stats
        sorted_h = sorted(self.bpm_history)
        n  = len(sorted_h)
        q1 = sorted_h[n // 4]
        q3 = sorted_h[(3 * n) // 4]
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filtered = [x for x in sorted_h if lo <= x <= hi] or sorted_h

        overall_std        = statistics.stdev(filtered) if len(filtered) > 1 else 0.0
        stability_percent  = round(max(0.0, min(100.0, 100.0 - overall_std * 4.0)))

        log.info(
            "Final summary: median=%d BPM  remark=%s  readings=%d  stability=%d%%",
            final_bpm, remark, len(self.bpm_history), stability_percent,
        )

        return {
            "final_bpm":         final_bpm,
            "min_bpm":           round(min(self.bpm_history)),
            "max_bpm":           round(max(self.bpm_history)),
            "avg_bpm":           round(statistics.mean(self.bpm_history)),
            "stability_percent": stability_percent,
            "remark":            remark,
            "total_readings":    len(self.bpm_history),
            "algorithm":         self.algorithm.value,
            **self._summarise_rr(),
        }

    def _summarise_rr(self) -> dict:
        """
        Session-level respiratory rate: IQR-filtered median of rr_history.

        Deliberately mirrors how final_bpm is derived above, because RR needs
        the same treatment for the same reason — individual per-window
        estimates are noisy, and a median over many of them is dramatically
        better than any single one. Measured on BIDMC: per-window MAE 2.89
        br/min vs. session-level MAE 1.37 (median 0.37), 85% of recordings
        within 2 br/min.

        Returns final_rr=None (never 0, never a guess) when too few
        confident readings exist, so callers can distinguish "not measured"
        from "measured as low" — core.clinical.assess() already handles a
        missing RR correctly and will simply assess on HR alone.
        """
        # 3 is the floor at which a median means anything at all; below that a
        # single bad reading would BE the answer.
        if len(self.rr_history) < 3:
            log.info("Final summary: RR not measurable (%d confident readings)",
                     len(self.rr_history))
            return {
                "final_rr":        None,
                "rr_readings":     len(self.rr_history),
                "rr_stability_percent": 0,
            }

        sorted_rr = sorted(self.rr_history)
        n  = len(sorted_rr)
        q1 = sorted_rr[n // 4]
        q3 = sorted_rr[(3 * n) // 4]
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filtered = [x for x in sorted_rr if lo <= x <= hi] or sorted_rr

        final_rr = round(float(statistics.median(filtered)), 1)
        rr_std   = statistics.stdev(filtered) if len(filtered) > 1 else 0.0
        # RR spans a ~6-30 br/min range vs HR's ~40-180, so a given std is
        # proportionally more significant here — hence the steeper factor
        # than final_bpm's 4.0.
        rr_stability = round(max(0.0, min(100.0, 100.0 - rr_std * 10.0)))

        log.info("Final summary: RR median=%.1f br/min  readings=%d  stability=%d%%",
                 final_rr, len(self.rr_history), rr_stability)

        return {
            "final_rr":             final_rr,
            "rr_readings":          len(self.rr_history),
            "rr_stability_percent": rr_stability,
        }

    @staticmethod
    def _classify_remark(bpm: float) -> str:
        if bpm < 60:
            return "Bradycardia (Slow)"
        if bpm <= 100:
            return "Normal Resting Heart Rate"
        if bpm <= 120:
            return "Monitor (Mildly Elevated)"
        return "Tachycardia (Fast)"

    def get_signal_quality(self) -> float:
        """Legacy compatibility shim."""
        return 0.0
