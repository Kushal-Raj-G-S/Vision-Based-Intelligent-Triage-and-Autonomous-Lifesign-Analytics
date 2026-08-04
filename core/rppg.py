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
_SUBHARM_TOL_HZ        = 0.08  # how close to exactly 2x a bin must land to count


def _select_dominant_freq_hps(valid_freqs: np.ndarray, valid_psd: np.ndarray) -> tuple[float, int]:
    """
    Select the dominant frequency, checking for and correcting a specific
    subharmonic-lock signature: the raw top peak has a comparably-strong
    companion at almost exactly double its frequency. Returns (frequency_hz,
    bin_index). Name kept as `_hps` for call-site continuity even though the
    technique is no longer classic HPS — see module comment above for why.
    """
    n = len(valid_psd)
    top_idx = int(np.argmax(valid_psd))
    if n < 3:
        return float(valid_freqs[top_idx]), top_idx

    f0 = valid_freqs[top_idx]
    target = f0 * 2.0
    j = int(np.argmin(np.abs(valid_freqs - target)))
    if abs(valid_freqs[j] - target) <= _SUBHARM_TOL_HZ:
        if valid_psd[j] >= _SUBHARM_PROMOTE_RATIO * valid_psd[top_idx]:
            return float(valid_freqs[j]), j

    return float(f0), top_idx


# ── Utility ───────────────────────────────────────────────────────────────────

def _safe_butter_sos(low: float, high: float, fs: float, order: int = 4) -> Optional[np.ndarray]:
    """Build a Butterworth bandpass SOS; returns None on failure."""
    try:
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

        # RGB ring buffers
        self.r_buffer: deque[float] = deque(maxlen=window_size)
        self.g_buffer: deque[float] = deque(maxlen=window_size)
        self.b_buffer: deque[float] = deque(maxlen=window_size)

        # ── Filter state ──────────────────────────────────────────────
        # Seed filter (wide, used until first stable BPM is known)
        self._sos_seed = _safe_butter_sos(_BP_SEED_LOW, _BP_SEED_HIGH, fps)
        # Adaptive filter (recomputed each cycle once BPM estimate exists)
        self._sos_adaptive: Optional[np.ndarray] = None
        self._adaptive_low  = _BP_SEED_LOW
        self._adaptive_high = _BP_SEED_HIGH

        # Respiration filter (fixed — respiratory rate drift is slow)
        self._sos_resp = _safe_butter_sos(_RESP_LOW, _RESP_HIGH, fps, order=2)

        # ── Temporal smoothing ────────────────────────────────────────
        self.prev_bpm: float = 0.0
        self.prev_snr: float = 0.0
        self.prev_rr: float = 0.0

        # BPM lock/candidate hysteresis state — see _LOCK_* constants above.
        self._locked_bpm: float = 0.0
        self._raw_bpm_window: deque = deque(maxlen=_LOCK_CONFIRM_STREAK_HARMONIC)
        self._prior_defer_count: int = 0
        self.bpm_history: list[float] = []
        self._history_reject_streak: int = 0
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

    def add_frame(self, rgb: Optional[tuple], timestamp=None) -> None:
        """Append one (r, g, b) sample to the ring buffers."""
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

        dominant_freq, peak_idx = _select_dominant_freq_hps(valid_freqs, valid_psd)
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
        rr_bpm_raw, rr_conf = self._estimate_rr(ppg_raw)
        rr_bpm = self._smooth_rr(rr_bpm_raw, rr_conf) if rr_bpm_raw > 0 else 0.0

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

    def _estimate_rr(self, ppg_raw: np.ndarray) -> tuple[float, float]:
        """
        Estimate respiratory rate from the RSA-modulated component of the
        pulse signal.

        Fix vs. the original implementation: `signal.welch(..., nperseg=
        len(resp), noverlap=0)` used the ENTIRE buffer as a single segment —
        that's not actually Welch's method, it's a plain windowed
        periodogram (the whole point of Welch averaging is multiple
        overlapping sub-segments averaged together to reduce spectral
        variance; a single segment gets none of that benefit no matter how
        much `nfft` zero-padding is applied on top). Respiration's RSA
        modulation is much weaker than the pulse signal itself, so a noisy
        single-segment PSD tends to produce a spurious peak sitting right at
        one edge of the 6-30 br/min search band rather than a genuine
        mid-band peak — which is exactly the "bounces between ~6 and ~30"
        symptom this was producing. Using real overlapping sub-segments
        (Hann-windowed, 50% overlap) genuinely averages down that variance.
        """
        if self._sos_seed is None or len(ppg_raw) < int(self.fps * 6):
            return 0.0, 0.0
        try:
            resp = signal.sosfiltfilt(self._sos_resp, ppg_raw)

            # Real multi-segment Welch averaging: sub-windows of ~8s (or the
            # full buffer if shorter), 50% overlap — several respiration
            # cycles per segment while still averaging 2+ segments together.
            nperseg = min(len(resp), int(self.fps * 8))
            nperseg = max(nperseg, int(self.fps * 4))  # floor: >=1 full breath cycle at 6bpm/6s hard to hit lower
            nperseg = min(nperseg, len(resp))
            noverlap = nperseg // 2

            rfreqs, rpsd = signal.welch(
                resp, fs=self.fps,
                window="hann", nperseg=nperseg, noverlap=noverlap, nfft=4096,
            )
            rmask   = (rfreqs >= _RESP_LOW) & (rfreqs <= _RESP_HIGH)
            rfreqs  = rfreqs[rmask]
            rpsd    = rpsd[rmask]
            if len(rpsd) == 0:
                return 0.0, 0.0
            r_peak  = int(np.argmax(rpsd))
            rr_bpm  = float(rfreqs[r_peak] * 60.0)
            r_sp    = float(np.sum(rpsd[max(0, r_peak - 1):r_peak + 2]))
            r_total = float(np.sum(rpsd)) + 1e-9
            rr_conf = float(np.clip((r_sp / r_total) * 100.0, 0.0, 100.0))

            # A peak sitting exactly at the search-band edge is the known
            # failure signature (no genuine mid-band peak found) — penalise
            # its confidence rather than reporting it at face value.
            edge_bins = max(1, len(rpsd) // 20)  # innermost ~5% at each edge
            if r_peak < edge_bins or r_peak >= len(rpsd) - edge_bins:
                rr_conf *= 0.4

            return rr_bpm, rr_conf
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
            "ready": False, "ppg_signal": [],
        }

    # ── Session summary ───────────────────────────────────────────────────────

    def get_final_summary(self) -> dict:
        """
        Return session-end summary statistics.
        Returns final_bpm=None when data is insufficient — no fake values.
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
