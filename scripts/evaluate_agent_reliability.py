#!/usr/bin/env python3
"""
VITAL — Agent Narration Faithfulness Evaluation
==================================================
Everything validated so far (synthetic self-test, 56 real sessions, rPPG-10)
measures the rPPG SIGNAL pipeline. None of it measures the multi-agent LLM
layer at all — whether the Diagnostic/Coordinator agents actually restate
the deterministic clinical assessment faithfully, or drift/hallucinate a
different number, as instructed in agents/config/tasks.yaml ("restate these
exact figures... your role is explanation, not computation").

This script runs the REAL CrewAI pipeline (real NVIDIA NIM calls, not a
mock) against a fixed set of clinical scenarios spanning the vitals space
(the same 8 scenarios core/clinical.py's own reasoning was checked against),
each run multiple times to see run-to-run variance, and measures whether
the LLM's final JSON output exactly matches the deterministic
ClinicalAssessment on: esi_level, priority_score, is_shock, primary_diagnosis.

This produces the "Narration Fidelity Score" concept described in
README.md's Agent Reliability section — the measured answer to "does the
AI layer actually do what it's told, or just usually do what it's told."

Usage
-----
    python scripts/evaluate_agent_reliability.py
    python scripts/evaluate_agent_reliability.py --repeats 3
"""

import argparse
import json
import re
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import clinical, guidelines  # noqa: E402

SCENARIOS = [
    # name, hr, rr, hrv, stress_index, sqi
    ("normal", 72, 16, 45, 60, 90),
    ("isolated_tachycardia", 115, 16, 40, 70, 90),
    ("isolated_tachypnea", 80, 26, 40, 70, 90),
    ("mild_shock", 105, 22, 30, 100, 90),
    ("severe_shock", 150, 34, 15, 220, 90),
    ("bradycardia", 42, 14, 50, 50, 90),
    ("low_signal_quality", 130, 28, 20, 160, 25),
    ("no_signal", 0, 0, 0, 0, 0),
]


def _extract_json_blob(text):
    """Same best-effort JSON extraction used in production (web/routes.py),
    duplicated here rather than imported to avoid triggering Flask app /
    Neon DB initialisation side effects from importing the web package."""
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
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key_anywhere(v, key)
            if found is not None:
                return found
    return None


def _fake_vitals(hr, rr, hrv, stress, sqi):
    return {
        'success': True, 'bpm': float(hr), 'confidence': sqi, 'sqi': sqi,
        'stability': sqi, 'rr': float(rr), 'rr_confidence': 80.0 if rr > 0 else 0.0,
        'hrv': float(hrv), 'stress_index': float(stress),
        'estimated_lux': 140, 'motion_delta': 2, 'mode': 'WEBCAM',
    }


# A REAL, small, already-validated video — not a fake placeholder string.
# A prior version of this script used "synthetic:<name>" as video_path,
# trusting the task instructions to stop the agent from re-calling the
# extraction tool. The 8B model didn't reliably follow that instruction,
# called the tool anyway with the fake path, and Camera's dummy-mode
# fallback (correct for the live MJPEG stream, wrong here) never signalled
# "no more frames" — the process spun for 2+ hours before anyone noticed it
# could never finish. core/camera.py and scripts/analyze_video.py both got
# a real fix for that (fail-fast on dummy_mode + a hard frame cap), but this
# script gets a second, independent layer of safety on top: if the tool
# gets called anyway, it now processes a real, short, bounded clip instead
# of a nonexistent path.
_SAFE_TEST_VIDEO = "test/dataset_2/subject1/vid.avi"  # ~53s, already validated, always resolves


def run_one(name, hr, rr, hrv, stress, sqi, run_idx):
    from agents.crew import build_triage_crew  # imported lazily so --help doesn't need NVIDIA_API_KEY

    assessment = clinical.assess(hr=hr, rr=rr, hrv=hrv, stress_index=stress, sqi=sqi)
    grounding = guidelines.grounding_context(assessment)
    vitals = _fake_vitals(hr, rr, hrv, stress, sqi)

    t0 = time.time()
    try:
        crew_result = build_triage_crew().kickoff(inputs={
            "video_path": _SAFE_TEST_VIDEO,
            "precomputed_vitals": str(vitals),
            "clinical_assessment": assessment.narrative_context(),
            "grounding_context": grounding,
        })
        result_text = str(crew_result)
        crew_error = None
    except Exception as exc:
        result_text = ""
        crew_error = str(exc)
    elapsed = time.time() - t0

    parsed = _extract_json_blob(result_text) if result_text else None

    def _match(field, expected):
        if parsed is None:
            return None, "no_parseable_json"
        got = _find_key_anywhere(parsed, field)
        if got is None:
            return False, "field_missing"
        if isinstance(expected, bool):
            return bool(got) == expected, str(got)
        if isinstance(expected, (int, float)):
            try:
                return int(got) == int(expected), str(got)
            except (ValueError, TypeError):
                return False, str(got)
        return None, str(got)  # primary_diagnosis handled separately (string, not exact-match scored here)

    esi_match, esi_got = _match('esi_level', assessment.esi_level)
    priority_match, priority_got = _match('priority_score', assessment.priority_score)
    shock_match, shock_got = _match('is_shock', assessment.is_shock)
    diagnosis_got = _find_key_anywhere(parsed, 'primary_diagnosis') if parsed else None
    diagnosis_exact = (str(diagnosis_got).strip() == assessment.primary_diagnosis) if diagnosis_got else False
    diagnosis_lenient = (
        diagnosis_got is not None and
        any(word.lower() in str(diagnosis_got).lower() for word in assessment.primary_diagnosis.split() if len(word) > 4)
    )

    return {
        "scenario": name, "run": run_idx,
        "expected": assessment.to_dict(),
        "crew_error": crew_error,
        "parsed_ok": parsed is not None,
        "esi_match": esi_match, "esi_got": esi_got,
        "priority_match": priority_match, "priority_got": priority_got,
        "shock_match": shock_match, "shock_got": shock_got,
        "diagnosis_exact": diagnosis_exact, "diagnosis_lenient": diagnosis_lenient, "diagnosis_got": diagnosis_got,
        "elapsed_s": round(elapsed, 1),
        "raw_output": result_text[:800],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-agent narration faithfulness against the deterministic engine.")
    parser.add_argument("--repeats", type=int, default=2, help="Runs per scenario (LLM output can vary run to run)")
    parser.add_argument("--out", type=str, default="test/agent_reliability_results.jsonl")
    args = parser.parse_args()

    results = []
    print("=" * 90)
    print("AGENT NARRATION FAITHFULNESS EVALUATION")
    print(f"{len(SCENARIOS)} scenarios x {args.repeats} repeats = {len(SCENARIOS) * args.repeats} real CrewAI runs")
    print("=" * 90)

    with open(args.out, "w", encoding="utf-8") as f:
        for name, hr, rr, hrv, stress, sqi in SCENARIOS:
            for run_idx in range(1, args.repeats + 1):
                print(f"Running {name} (run {run_idx}/{args.repeats})...", end=" ", flush=True)
                r = run_one(name, hr, rr, hrv, stress, sqi, run_idx)
                results.append(r)
                f.write(json.dumps(r) + "\n")
                if r["crew_error"]:
                    print(f"CREW ERROR: {r['crew_error']}")
                else:
                    print(f"esi={r['esi_match']} priority={r['priority_match']} shock={r['shock_match']} "
                          f"dx_exact={r['diagnosis_exact']} dx_lenient={r['diagnosis_lenient']} ({r['elapsed_s']}s)")

    # Aggregate
    n = len(results)
    n_parsed = sum(1 for r in results if r["parsed_ok"])
    n_esi_ok = sum(1 for r in results if r["esi_match"] is True)
    n_priority_ok = sum(1 for r in results if r["priority_match"] is True)
    n_shock_ok = sum(1 for r in results if r["shock_match"] is True)
    n_dx_exact = sum(1 for r in results if r["diagnosis_exact"])
    n_dx_lenient = sum(1 for r in results if r["diagnosis_lenient"])

    print("-" * 90)
    print(f"Total runs: {n}")
    print(f"JSON successfully parsed:        {n_parsed}/{n} ({100*n_parsed/n:.1f}%)")
    print(f"esi_level exact match:            {n_esi_ok}/{n} ({100*n_esi_ok/n:.1f}%)")
    print(f"priority_score exact match:       {n_priority_ok}/{n} ({100*n_priority_ok/n:.1f}%)")
    print(f"is_shock exact match:             {n_shock_ok}/{n} ({100*n_shock_ok/n:.1f}%)")
    print(f"primary_diagnosis exact match:    {n_dx_exact}/{n} ({100*n_dx_exact/n:.1f}%)")
    print(f"primary_diagnosis lenient match:  {n_dx_lenient}/{n} ({100*n_dx_lenient/n:.1f}%)")
    print(f"\nFull per-run results written to {args.out}")
    print("=" * 90)


if __name__ == "__main__":
    main()
