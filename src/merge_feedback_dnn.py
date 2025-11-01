import csv
import os
import pandas as pd
from datetime import datetime

FEEDBACK_FILE = "feedback.csv"
NUM_FILE = "numerical_questions.csv"
CONC_FILE = "conceptual_questions.csv"


def merge_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        print("❌ No feedback file found. Nothing to merge.")
        return

    # Read feedback file
    feedback_df = pd.read_csv(FEEDBACK_FILE)
    if feedback_df.empty:
        print("📭 Feedback file is empty. Nothing to merge.")
        return

    print(f"📥 Found {len(feedback_df)} feedback samples to merge")

    # Read existing datasets
    numerical_data = []
    conceptual_data = []
    
    if os.path.exists(NUM_FILE):
        num_df = pd.read_csv(NUM_FILE)
        numerical_data = num_df.to_dict('records')
        print(f"📊 Existing numerical questions: {len(numerical_data)}")
    else:
        numerical_data = []
        print("📁 Creating new numerical questions file")
    
    if os.path.exists(CONC_FILE):
        conc_df = pd.read_csv(CONC_FILE)
        conceptual_data = conc_df.to_dict('records')
        print(f"📊 Existing conceptual questions: {len(conceptual_data)}")
    else:
        conceptual_data = []
        print("📁 Creating new conceptual questions file")

    # Process feedback
    merged_numerical = numerical_data.copy()
    merged_conceptual = conceptual_data.copy()
    
    new_numerical = 0
    new_conceptual = 0
    skipped = 0

    for _, feedback in feedback_df.iterrows():
        question = feedback['question']
        actual_label = feedback['actual']
        
        # Check if question already exists in either dataset
        question_exists = any(
            item['question'] == question 
            for item in merged_numerical + merged_conceptual
        )
        
        if question_exists:
            skipped += 1
            continue
        
        new_entry = {'question': question, 'label': actual_label}
        
        if actual_label == 'numerical':
            merged_numerical.append(new_entry)
            new_numerical += 1
        elif actual_label == 'conceptual':
            merged_conceptual.append(new_entry)
            new_conceptual += 1

    # Save merged datasets
    pd.DataFrame(merged_numerical).to_csv(NUM_FILE, index=False)
    pd.DataFrame(merged_conceptual).to_csv(CONC_FILE, index=False)
    
    # Clear feedback file
    os.remove(FEEDBACK_FILE)
    
    print(f"\n✅ Merge completed!")
    print(f"📈 New numerical questions added: {new_numerical}")
    print(f"📈 New conceptual questions added: {new_conceptual}")
    print(f"📊 Skipped duplicates: {skipped}")
    print(f"📁 Updated numerical questions: {len(merged_numerical)}")
    print(f"📁 Updated conceptual questions: {len(merged_conceptual)}")
    print(f"🗑️  Feedback file cleared")


if __name__ == "__main__":
    merge_feedback()