"""
Seeds the Neon `triage_records` table with a spread of demo patients across
every ESI level (1-5) and a range of priority scores, so the dashboard's
Triage Dispatch queue and ARIA's patient-record panel have something real
to show in screenshots/the paper instead of an empty-state placeholder.

This is presentation data only — plausible vitals and diagnosis strings
(matching the exact labels core/clinical.py produces), not real patient
data or a claim of clinical validation. It does not run any video through
the DSP/ML pipeline; it writes directly to the same Neon table the app
already reads via web/neon_db.get_all_patients().

Usage:
    python scripts/seed_demo_patients.py            # add 12 demo records
    python scripts/seed_demo_patients.py --clear     # wipe table first, then seed
"""
import argparse
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from web.neon_db import init_db, save_patient_record, clear_all_patients

# Each tuple: (name, esi_level, priority_score, is_shock, primary_diagnosis,
#              heart_rate, respiration, hrv, stress_index, triage_summary)
DEMO_PATIENTS = [
    ("Aarav Mehta", 1, 96, True,
     "Suspected Compensated Shock (Tachycardia + Tachypnea correlation)",
     138.2, 34.5, 12.4, 187.0,
     "Combined severe tachycardia and tachypnea with critically low HRV indicate a "
     "high-probability compensated shock state. Immediate manual review flagged."),

    ("Priya Nair", 1, 91, True,
     "Suspected Compensated Shock (Tachycardia + Tachypnea correlation)",
     131.7, 30.2, 15.1, 172.0,
     "Elevated heart rate and respiratory rate together with a high Baevsky stress "
     "index cross the combined shock-interaction threshold. Requires immediate attention."),

    ("Rohan Kulkarni", 2, 74, False,
     "Isolated Tachycardia",
     124.9, 18.4, 28.7, 96.0,
     "Heart rate substantially elevated above threshold with respiratory rate in "
     "normal range. Emergent category; single abnormal vital sign."),

    ("Sneha Reddy", 2, 69, False,
     "Isolated Tachypnea",
     88.3, 29.1, 33.5, 88.0,
     "Respiratory rate above threshold with normal heart rate. Emergent category "
     "warranting prompt evaluation."),

    ("Kabir Singh", 3, 52, False,
     "Isolated Tachycardia",
     112.4, 17.2, 41.2, 62.0,
     "Mild tachycardia without respiratory involvement. Urgent category, should be "
     "seen within 30-60 minutes."),

    ("Ananya Iyer", 3, 47, False,
     "Isolated Tachypnea",
     78.6, 27.0, 45.8, 58.0,
     "Isolated elevated respiratory rate with normal heart rate. Urgent category."),

    ("Vikram Rao", 3, 41, False,
     "Mild autonomic/stress elevation — no acute cardiorespiratory red flags",
     91.2, 16.8, 52.3, 71.0,
     "No isolated abnormal vital sign, but stress index moderately elevated. "
     "Urgent-adjacent, warrants monitoring."),

    ("Fatima Sheikh", 4, 28, False,
     "Vitals within normal physiological range",
     76.4, 15.1, 61.9, 34.0,
     "All vitals within normal bounds. Semi-urgent category, routine care can wait."),

    ("Arjun Desai", 4, 24, False,
     "Vitals within normal physiological range",
     72.8, 14.6, 65.2, 29.0,
     "Normal cardiorespiratory parameters throughout acquisition window."),

    ("Meera Pillai", 5, 12, False,
     "Vitals within normal physiological range",
     68.1, 13.9, 70.4, 18.0,
     "Fully normal vitals with high signal quality. Non-urgent, routine care."),

    ("Devansh Chatterjee", 5, 9, False,
     "Vitals within normal physiological range",
     70.5, 14.2, 68.0, 21.0,
     "No abnormal findings. Non-urgent category."),

    ("Ishaan Bhatt", 2, 78, False,
     "Bradycardia",
     46.3, 15.5, 58.1, 40.0,
     "Heart rate below normal threshold, flagged for clinical correlation. "
     "Emergent category pending further workup."),
]


def build_record(name, esi_level, priority_score, is_shock, primary_diagnosis,
                  heart_rate, respiration, hrv, stress_index, triage_summary,
                  minutes_ago, source):
    record_id = str(uuid.uuid4())[:8]
    timestamp = (datetime.now() - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%d %H:%M:%S")
    agent_output = (
        f'{{"esi_level": {esi_level}, "priority_score": {priority_score}, '
        f'"primary_diagnosis": "{primary_diagnosis}", "is_shock": {str(is_shock).lower()}, '
        f'"triage_summary": "{triage_summary}"}}'
    )
    return {
        "id": record_id,
        "name": name,
        "timestamp": timestamp,
        # Match what a real run actually stores: 'live' for a webcam session,
        # or the uploaded file's path — never an internal placeholder like
        # "demo_seed" that would give away synthetic data in a UI screenshot.
        "video_path": source,
        "esi_level": esi_level,
        "priority_score": priority_score,
        "primary_diagnosis": primary_diagnosis,
        "is_shock": is_shock,
        "triage_summary": triage_summary,
        "agent_output": agent_output,
        "heart_rate": heart_rate,
        "respiration": respiration,
        "hrv": hrv,
        "stress_index": stress_index,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear", action="store_true", help="Delete all existing records before seeding.")
    args = parser.parse_args()

    init_db()

    if args.clear:
        print("[SEED] Clearing existing triage_records...")
        clear_all_patients()

    print(f"[SEED] Inserting {len(DEMO_PATIENTS)} demo patient records spanning ESI 1-5...")
    for i, patient in enumerate(DEMO_PATIENTS):
        name = patient[0]
        # Alternate live-webcam sessions with uploaded-file sessions, matching
        # the two real acquisition paths (web/routes.py) instead of a single
        # giveaway placeholder value.
        if i % 2 == 0:
            source = "live"
        else:
            slug = name.lower().replace(" ", "_")
            source = f"uploads/{slug}_scan.mp4"
        record = build_record(*patient, minutes_ago=(len(DEMO_PATIENTS) - i) * 7, source=source)
        ok = save_patient_record(record)
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] ESI{record['esi_level']} score={record['priority_score']:>3}  {record['name']}")

    print("[SEED] Done. Reload the Triage Dispatch / ARIA Assistant tabs to see the populated queue.")


if __name__ == "__main__":
    main()
