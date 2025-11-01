import os
import csv
from typing import List, Tuple
import torch
import torch.nn as nn
import joblib
import pandas as pd
from datetime import datetime

MODEL_FILE = "question_classifier_dnn.pt"
VECTORIZER_FILE = "tfidf_vectorizer.joblib"
FEEDBACK_FILE = "feedback.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuestionClassifierDNN(nn.Module):
    def __init__(self, input_size):
        super(QuestionClassifierDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu3(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


class InteractiveQuestionClassifier:
    def __init__(self):
        if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
            raise FileNotFoundError("Model or vectorizer not found. Please run training first.")
        
        self.vectorizer = joblib.load(VECTORIZER_FILE)
        input_size = len(self.vectorizer.get_feature_names_out())
        self.model = QuestionClassifierDNN(input_size=input_size).to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
        self.model.eval()
        
        self.label_map = {0: "numerical", 1: "conceptual"}
        print("✅ Model loaded successfully!")

    def predict(self, question: str) -> str:
        X = self.vectorizer.transform([question]).toarray()
        X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)
        
        predicted_label = self.label_map[pred_idx.item()]
        confidence_score = confidence.item()
        
        return predicted_label, confidence_score

    def get_user_feedback(self, question: str, predicted_label: str, confidence: float):
        print(f"🧠 Predicted label: {predicted_label} (confidence: {confidence:.2f})")
        response = input("Is this correct? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            # Save positive feedback
            self._save_feedback(question, predicted_label, predicted_label, confidence, correct=True)
            print("✅ Great! Thank you for confirming. Feedback saved!")
        elif response in ['no', 'n']:
            correct_label = input("Please provide the correct label (numerical/conceptual): ").strip().lower()
            if correct_label in ['numerical', 'conceptual']:
                self._save_feedback(question, predicted_label, correct_label, confidence, correct=False)
                print("✅ Thanks for the correction! Feedback saved for retraining.")
            else:
                print("❌ Invalid label. Please enter 'numerical' or 'conceptual'.")
        else:
            print("❌ Please answer with 'yes' or 'no'.")

    def _save_feedback(self, question: str, predicted: str, actual: str, confidence: float, correct: bool):
        """Save feedback to CSV file with proper formatting"""
        feedback_data = {
            'question': question,
            'predicted': predicted,
            'actual': actual,
            'confidence': confidence,
            'correct': correct,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create DataFrame for this feedback
        new_feedback = pd.DataFrame([feedback_data])
        
        # Append to existing feedback file or create new one
        if os.path.exists(FEEDBACK_FILE):
            existing_feedback = pd.read_csv(FEEDBACK_FILE)
            updated_feedback = pd.concat([existing_feedback, new_feedback], ignore_index=True)
        else:
            updated_feedback = new_feedback
        
        # Save with proper headers
        updated_feedback.to_csv(FEEDBACK_FILE, index=False)
        
        print(f"💾 Feedback saved. Total feedback samples: {len(updated_feedback)}")


def main():
    try:
        classifier = InteractiveQuestionClassifier()
        print("🔍 Interactive Question Classifier (DNN)")
        print("   Type 'exit' to quit, 'stats' to see feedback count")
        print("=" * 50)
        
        feedback_count = 0
        if os.path.exists(FEEDBACK_FILE):
            feedback_df = pd.read_csv(FEEDBACK_FILE)
            feedback_count = len(feedback_df)
            print(f"📊 Starting with {feedback_count} feedback samples")
        
        while True:
            question = input("\n❓ Enter your question: ").strip()
            
            if question.lower() == 'exit':
                break
            elif question.lower() == 'stats':
                if os.path.exists(FEEDBACK_FILE):
                    feedback_df = pd.read_csv(FEEDBACK_FILE)
                    print(f"📈 Current feedback samples: {len(feedback_df)}")
                else:
                    print("📈 No feedback collected yet.")
                continue
            elif not question:
                continue
            
            try:
                predicted_label, confidence = classifier.predict(question)
                classifier.get_user_feedback(question, predicted_label, confidence)
            except Exception as e:
                print(f"❌ Error during prediction: {e}")
        
        print("\n👋 Exiting...")
        if os.path.exists(FEEDBACK_FILE):
            feedback_df = pd.read_csv(FEEDBACK_FILE)
            print(f"📊 Total feedback collected in this session: {len(feedback_df) - feedback_count}")
            
    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()