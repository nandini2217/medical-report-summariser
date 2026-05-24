"""
scripts/predict.py
------------------
Test the model on any clinical text.

Usage:
    python scripts/predict.py
    python scripts/predict.py --text "Patient admitted with chest pain..."
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model.trainer import generate_summary

DEMO_TEXT = """
Patient is a 72-year-old female with a history of atrial fibrillation,
hypertension, and type 2 diabetes mellitus presenting with a 2-day history
of palpitations and mild dyspnea on exertion. EKG reveals rapid ventricular
response AFib at 138 bpm. She was rate controlled with IV metoprolol 5mg x2
with heart rate reducing to 84 bpm. Warfarin continued with INR therapeutic
at 2.3. Echo showed preserved EF at 58%. Discharged on metoprolol succinate
100mg daily, warfarin 5mg daily, lisinopril 10mg daily, metformin 1000mg BID.
Follow up with cardiology in 1 week.
"""

def main(args):
    text = args.text or DEMO_TEXT.strip()
    print("=" * 55)
    print("  Medical Report Summariser — Prediction")
    print("=" * 55)
    print(f"\nInput ({len(text.split())} words):")
    print("-" * 40)
    print(text[:300] + ("..." if len(text) > 300 else ""))
    print("-" * 40)
    print("\nGenerating summary...")
    summary = generate_summary(text)
    print("\nOutput:")
    print("-" * 40)
    print(summary)
    print("-" * 40)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, default=None,
                   help="Clinical text to summarise (uses demo text if not provided)")
    main(p.parse_args())
