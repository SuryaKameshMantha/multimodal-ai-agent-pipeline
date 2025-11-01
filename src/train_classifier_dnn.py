import os
import csv
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np

NUM_FILE = "numerical_questions.csv"
CONC_FILE = "conceptual_questions.csv"
FEEDBACK_FILE = "feedback.csv"
MODEL_FILE = "question_classifier_dnn.pt"
VECTORIZER_FILE = "tfidf_vectorizer.joblib"

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
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
        
        self.fc4 = nn.Linear(64, 2)  # 2 classes: numerical, conceptual

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.relu3(self.fc3(x))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x


def load_dataset(files: List[str]) -> List[Tuple[str, str]]:
    data = []
    for file in files:
        if os.path.exists(file):
            print(f"Loading data from {file}")
            # Use pandas to properly handle CSV with headers
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                data.append((row['question'], row['label']))
            print(f"  Loaded {len(df)} examples from {file}")
    return data


def train():
    # Only use the main dataset files, NOT feedback during initial training
    files = [NUM_FILE, CONC_FILE]
    dataset = load_dataset(files)
    
    if not dataset:
        raise ValueError("No training data found. Please provide datasets.")

    texts, labels = zip(*dataset)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"Total examples: {len(texts)}")
    label_counts = pd.Series(labels).value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count} examples")
    
    print("\nSample questions from each category:")
    numerical_samples = [text for text, label in dataset if label == 'numerical'][:2]
    conceptual_samples = [text for text, label in dataset if label == 'conceptual'][:2]
    
    print("Numerical samples:")
    for sample in numerical_samples:
        print(f"  - {sample[:80]}...")
    
    print("Conceptual samples:")
    for sample in conceptual_samples:
        print(f"  - {sample[:80]}...")

    # Vectorize texts using TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=1000, 
        stop_words='english', 
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    X = vectorizer.fit_transform(texts).toarray()
    
    print(f"\nVectorizer features: {X.shape[1]}")

    # Convert labels to numeric
    label_map = {"numerical": 0, "conceptual": 1}
    y = torch.tensor([label_map[label] for label in labels], dtype=torch.long)
    X = torch.tensor(X, dtype=torch.float32)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} examples")
    print(f"Validation set: {len(X_val)} examples")

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = QuestionClassifierDNN(X.shape[1]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nTraining DNN classifier...")
    best_accuracy = 0
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), MODEL_FILE)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, Val Accuracy: {accuracy:.2f}%")
    
    print(f"\nTraining complete! Best validation accuracy: {best_accuracy:.2f}%")
    
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print(f"Model saved to {MODEL_FILE}")
    print(f"Vectorizer saved to {VECTORIZER_FILE}")


if __name__ == "__main__":
    train()