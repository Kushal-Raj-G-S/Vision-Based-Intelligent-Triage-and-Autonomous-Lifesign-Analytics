"""
VITAL — Deterministic Clinical Triage Engine
=============================================
Rule-grounded ESI classification and Compensated Shock scoring, computed
directly from extracted vitals in plain Python.

Why this file exists
─────────────────────
Previously, ESI level / shock flag / priority score were whatever the LLM
happened to emit as JSON at the end of a free-text CrewAI run, with a
silent hardcoded fallback (esi_level=3, priority_score=50, "General
Assessment") whenever parsing failed. That means the single most safety-
critical number in the system — the acuity level driving queue order —
was non-deterministic and occasionally just made up.

This module is now the system of record for that number. The LLM agents
in agents/ receive this assessment as authoritative input and are
instructed to *narrate and explain* it, not compute or override it.

Grounded in
───────────
- Emergency Severity Index (ESI), a five-level acuity scale used across
  US emergency departments (AHRQ ESI Implementation Handbook, public
  reference standard — see agents/config/tasks.yaml for the level table
  as specified in this project's own brief).
- The project's Compensated Shock rule: Tachycardia (HR > 100 bpm) +
  Tachypnea (RR > 20 br/min) occurring together.
- Baevsky Stress Index / HRV (RMSSD) thresholds already used elsewhere in
  this codebase (web/routes.py's report labelling) as secondary,
  lower-weight autonomic modulators.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("vital.clinical")

# ── ESI acuity bands ──────────────────────────────────────────────────────────

ESI_LEVELS = {
    1: ("Resuscitation", "Immediate life-saving intervention required."),
    2: ("Emergent", "Should be seen within 10-15 minutes."),
    3: ("Urgent", "Should be seen within 30-60 minutes."),
    4: ("Semi-Urgent", "Should be seen within 1-2 hours."),
    5: ("Non-Urgent", "Routine care; can safely wait."),
}

# ── Clinical thresholds (adult, resting) ──────────────────────────────────────
# From this project's Triage Rule Table (project details.pdf) plus standard
# adult vital-sign reference ranges already used in web/routes.py._classify().

HR_TACHY = 100.0     # bpm — tachycardia threshold
HR_BRADY = 50.0      # bpm — clinically significant bradycardia
RR_TACHY = 20.0      # breaths/min — tachypnea threshold
STRESS_CRITICAL = 150.0   # Baevsky Stress Index
STRESS_ELEVATED = 80.0
HRV_LOW = 20.0             # ms RMSSD — below this = reduced autonomic reserve
SQI_MIN_CONFIDENT = 40.0   # composite SQI below this = low-confidence vitals
RR_CONFIDENCE_LOW = 30.0   # rr_confidence (0-100) below this = respiration reading is unreliable


@dataclass
class ClinicalAssessment:
    esi_level: int
    esi_category: str
    priority_score: int            # 1-100 queue priority, higher = more urgent
    is_shock: bool
    shock_risk_score: float        # 0-100 continuous risk (the interaction term)
    primary_diagnosis: str
    reasoning: list = field(default_factory=list)
    confidence: str = "MODERATE"   # LOW / MODERATE / HIGH — driven by signal quality
    needs_manual_review: bool = False

    def to_dict(self) -> dict:
        return {
            "esi_level": self.esi_level,
            "esi_category": self.esi_category,
            "priority_score": self.priority_score,
            "is_shock": self.is_shock,
            "shock_risk_score": round(self.shock_risk_score, 1),
            "primary_diagnosis": self.primary_diagnosis,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "needs_manual_review": self.needs_manual_review,
        }

    def narrative_context(self) -> str:
        """Render as plain text for injection into an LLM prompt."""
        lines = [
            f"DETERMINISTIC CLINICAL ASSESSMENT (authoritative — do not override):",
            f"  ESI Level: {self.esi_level} ({self.esi_category})",
            f"  Priority Score: {self.priority_score}/100",
            f"  Compensated Shock Flag: {'YES' if self.is_shock else 'NO'} "
            f"(interaction risk score: {self.shock_risk_score:.1f}/100)",
            f"  Primary Diagnosis: {self.primary_diagnosis}",
            f"  Confidence: {self.confidence}"
            + (" — FLAGGED FOR MANUAL REVIEW" if self.needs_manual_review else ""),
            "  Reasoning trace:",
        ]
        lines += [f"    - {r}" for r in self.reasoning] or ["    - (no contributing factors triggered)"]
        return "\n".join(lines)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _tachy_term(value: float, threshold: float, span: float) -> float:
    """0 at/below threshold, ramps linearly to 1 over `span` units above it."""
    if value <= threshold:
        return 0.0
    return _clamp01((value - threshold) / span)


def _brady_term(value: float, threshold: float, span: float) -> float:
    if value <= 0 or value >= threshold:
        return 0.0
    return _clamp01((threshold - value) / span)


def assess(
    hr: float,
    rr: float,
    hrv: float = 0.0,
    stress_index: float = 0.0,
    sqi: float = 100.0,
    rr_confidence: float = 100.0,
) -> ClinicalAssessment:
    """
    Compute a deterministic ESI level + Compensated Shock assessment.

    Shock scoring — a real polynomial cross-correlation, not an AND-gate
    ────────────────────────────────────────────────────────────────────
    Compensated Shock is a *synergistic* condition: tachycardia AND
    tachypnea together indicate the body compensating for poor perfusion
    (faster pump + blowing off CO2). A boolean `HR>100 and RR>20` treats
    "HR 101, RR 21" identically to "HR 160, RR 35" — both trip the same
    flag with no sense of degree. Instead we compute a graded interaction
    term:

        shock_interaction = tachy_hr(HR) * tachy_rr(RR)   ∈ [0, 1]

    where tachy_hr/tachy_rr are each 0 at the clinical threshold and ramp
    toward 1 as the vital moves further into the abnormal range.
    Multiplying — rather than averaging or OR-ing — means risk only
    accumulates when BOTH vitals are elevated together; an isolated
    tachycardia with normal respiration scores near zero on this term,
    matching the physiological definition of *compensated* shock as a
    combined cardiorespiratory response rather than either sign alone.

    HRV (RMSSD) and the Baevsky Stress Index are folded in as secondary,
    lower-weight terms reflecting reduced autonomic reserve — they shift
    the composite risk score but cannot trigger the shock flag alone.
    """
    if hr <= 0 or rr <= 0:
        return ClinicalAssessment(
            esi_level=3,
            esi_category=ESI_LEVELS[3][0],
            priority_score=50,
            is_shock=False,
            shock_risk_score=0.0,
            primary_diagnosis="Insufficient signal for clinical assessment",
            reasoning=[
                "Heart rate and/or respiratory rate could not be reliably "
                "extracted from the available video/session — defaulting to "
                "ESI 3 pending manual clinical assessment."
            ],
            confidence="LOW",
            needs_manual_review=True,
        )

    reasoning: list = []

    tachy_hr = _tachy_term(hr, HR_TACHY, span=40.0)   # saturates ≈140 bpm
    tachy_rr = _tachy_term(rr, RR_TACHY, span=15.0)   # saturates ≈35 br/min
    brady_hr = _brady_term(hr, HR_BRADY, span=20.0)

    # Polynomial (multiplicative) cross-correlation — the actual
    # "Compensated Shock" synergy signal.
    shock_interaction = tachy_hr * tachy_rr

    # Autonomic secondary terms (weak evidence; cannot trigger shock alone)
    stress_term = (
        _clamp01((stress_index - STRESS_ELEVATED) / (STRESS_CRITICAL - STRESS_ELEVATED))
        if stress_index > 0 else 0.0
    )
    hrv_term = _clamp01((HRV_LOW - hrv) / HRV_LOW) if 0 < hrv < HRV_LOW else 0.0

    shock_risk_score = 100.0 * _clamp01(
        0.55 * shock_interaction +
        0.25 * max(tachy_hr, tachy_rr) +   # isolated single-vital elevation still counts, less heavily
        0.12 * stress_term +
        0.08 * hrv_term
    )

    # The project's Compensated Shock rule is a literal threshold AND
    # (HR > 100 bpm and RR > 20 br/min) — that stays the boolean flag, exactly
    # as specified. The continuous `shock_interaction` term above does NOT
    # gate this boolean; it instead grades *severity* once the flag is true
    # (a 105/22 borderline case and a 160/35 crisis case both flag `is_shock`,
    # but only the latter drives ESI to level 1 and priority near 100).
    is_shock = hr > HR_TACHY and rr > RR_TACHY

    if is_shock:
        reasoning.append(
            f"Compensated Shock criteria met: HR {hr:.0f} bpm (>{HR_TACHY:.0f}, tachycardia) "
            f"AND RR {rr:.0f} br/min (>{RR_TACHY:.0f}, tachypnea); interaction severity={shock_interaction:.2f}."
        )
        # Respiratory rate here is a proxy signal — respiratory sinus
        # arrhythmia riding on the pulse waveform, not a direct chest-motion
        # measurement — and is the noisiest channel this system produces.
        # A shock flag driven by a low-confidence RR reading (e.g. lighting,
        # talking, brief motion during capture) should be treated as
        # tentative, not confirmed, until re-measured under cleaner
        # conditions.
        if rr_confidence < RR_CONFIDENCE_LOW:
            reasoning.append(
                f"CAUTION: respiratory-rate confidence is {rr_confidence:.0f}% "
                f"(below the {RR_CONFIDENCE_LOW:.0f}% reliability floor) — this rr reading is a "
                f"proxy signal (respiratory sinus arrhythmia, not direct chest measurement) and "
                f"may be inflated by lighting, talking, or minor motion. Re-measure under still, "
                f"well-lit, silent conditions before treating this shock flag as confirmed."
            )
    elif tachy_hr > 0:
        reasoning.append(f"Isolated tachycardia: HR {hr:.0f} bpm (>{HR_TACHY:.0f}), respiration normal.")
    elif tachy_rr > 0:
        reasoning.append(f"Isolated tachypnea: RR {rr:.0f} br/min (>{RR_TACHY:.0f}), heart rate normal.")

    if brady_hr > 0:
        reasoning.append(f"Bradycardia present: HR {hr:.0f} bpm (<{HR_BRADY:.0f}).")
    if stress_term > 0:
        label = "CRITICAL" if stress_index > STRESS_CRITICAL else "ELEVATED"
        reasoning.append(f"Baevsky Stress Index {stress_index:.0f} — {label} autonomic load.")
    if hrv_term > 0:
        reasoning.append(f"HRV (RMSSD) {hrv:.1f} ms — reduced autonomic variability (<{HRV_LOW:.0f} ms).")

    # ── ESI mapping ────────────────────────────────────────────────
    if is_shock and shock_risk_score >= 70:
        esi_level = 1
    elif is_shock or shock_risk_score >= 55 or brady_hr >= 0.75:
        esi_level = 2
    elif shock_risk_score >= 25 or tachy_hr > 0 or tachy_rr > 0 or stress_term >= 0.5 or brady_hr > 0:
        esi_level = 3
    elif hrv_term > 0 or stress_term > 0:
        esi_level = 4
    else:
        esi_level = 5

    esi_category, _ = ESI_LEVELS[esi_level]

    # Priority score — monotonic with ESI band, but continuous risk breaks
    # ties *within* a band so the queue is genuinely dynamic, not a flat
    # 5-bucket sort.
    priority_score = int(round(
        _clamp01((6 - esi_level) / 5.0 * 0.7 + shock_risk_score / 100.0 * 0.3) * 100
    ))
    priority_score = max(1, min(100, priority_score))

    if is_shock:
        primary_diagnosis = "Suspected Compensated Shock (Tachycardia + Tachypnea correlation)"
    elif tachy_hr > 0 and tachy_rr == 0:
        primary_diagnosis = "Isolated Tachycardia"
    elif tachy_rr > 0 and tachy_hr == 0:
        primary_diagnosis = "Isolated Tachypnea"
    elif brady_hr > 0:
        primary_diagnosis = "Bradycardia"
    elif esi_level >= 4:
        primary_diagnosis = "Vitals within normal physiological range"
    else:
        primary_diagnosis = "Mild autonomic/stress elevation — no acute cardiorespiratory red flags"

    # ── Confidence / manual-review gating on signal quality ─────────
    if sqi < SQI_MIN_CONFIDENT:
        confidence = "LOW"
        needs_manual_review = True
        reasoning.append(
            f"Signal Quality Index {sqi:.0f} is below the {SQI_MIN_CONFIDENT:.0f} confidence "
            f"floor — this classification should be confirmed by manual assessment."
        )
    elif sqi < 70:
        confidence = "MODERATE"
        needs_manual_review = False
    else:
        confidence = "HIGH"
        needs_manual_review = False

    # A Compensated Shock flag is the single highest-stakes output this
    # system produces. It should never be presented as a settled result
    # without a human confirming it — regardless of how "MODERATE" or
    # "HIGH" the overall signal-quality tier looks, since SQI measures
    # waveform cleanliness generically and does not specifically capture
    # respiratory-rate reliability (see the rr_confidence check above).
    if is_shock and not needs_manual_review:
        needs_manual_review = True
        reasoning.append(
            "Compensated Shock flags always require manual confirmation before acting on "
            "them, independent of signal-quality tier — this is a policy, not a confidence gap."
        )

    result = ClinicalAssessment(
        esi_level=esi_level,
        esi_category=esi_category,
        priority_score=priority_score,
        is_shock=is_shock,
        shock_risk_score=shock_risk_score,
        primary_diagnosis=primary_diagnosis,
        reasoning=reasoning,
        confidence=confidence,
        needs_manual_review=needs_manual_review,
    )
    log.info(
        "Clinical assessment: ESI=%d shock=%s risk=%.1f priority=%d confidence=%s",
        result.esi_level, result.is_shock, result.shock_risk_score,
        result.priority_score, result.confidence,
    )
    return result
