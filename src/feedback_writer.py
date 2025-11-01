# feedback_writer.py (updated)
#!/usr/bin/env python3
"""
Feedback Writer - Creates feedback.csv and writes user classifications
This is called BEFORE merge_feedback_dnn.py
"""
import os
import csv
from datetime import datetime
from pathlib import Path

FEEDBACK_FILE = "feedback.csv"

def ensure_feedback_csv_exists():
    """Create feedback.csv with headers if it doesn't exist"""
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['question', 'predicted', 'actual', 'confidence', 'correct', 'timestamp'])
        print(f"✅ Created {FEEDBACK_FILE}")
    return FEEDBACK_FILE

def write_feedback(question, predicted_label, actual_label, confidence=0.0, confirmed=True):
    """
    Write user feedback to feedback.csv

    Args:
        question: The physics question
        predicted_label: What DNN predicted ('numerical' or 'conceptual')
        actual_label: What user confirmed ('numerical' or 'conceptual')
        confidence: Confidence score from DNN
        confirmed: Whether user confirmed (True) or overrode (False)
    """
    try:
        ensure_feedback_csv_exists()

        with open(FEEDBACK_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                question,
                predicted_label,
                actual_label,
                confidence,
                'YES' if confirmed else 'NO',
                datetime.now().isoformat()
            ])

        print(f"✅ Feedback written: {question[:50]}")
        return True
    except Exception as e:
        print(f"❌ Error writing feedback: {e}")
        return False

if __name__ == "__main__":
    # Test
    write_feedback("What is Newton's first law?", "numerical", "conceptual", confidence=0.85, confirmed=False)