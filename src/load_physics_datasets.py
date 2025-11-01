"""
Load physics datasets efficiently
- Only uses working datasets
- Avoids gated/problematic ones
- Fast and reliable
"""

import os
from typing import List, Tuple, Dict
import pandas as pd
from datasets import load_dataset
import json
from tqdm import tqdm


class PhysicsDatasetLoader:
    """Load and process real physics Q&A datasets"""
    
    def __init__(self):
        self.datasets = {}
        self.combined_qa = []
    
    def load_physics_scienceqa(self) -> List[Tuple[str, str]]:
        """Load physics-scienceqa dataset (WORKING) ✅"""
        print("📥 Loading physics-scienceqa dataset...")
        
        try:
            dataset = load_dataset("veggiebird/physics-scienceqa")
            qa_pairs = []
            
            for item in tqdm(dataset['train'], desc="Processing ScienceQA"):
                question = item['input']
                answer = item['output']
                if question and answer:
                    qa_pairs.append((question, answer))
            
            print(f"✅ Loaded {len(qa_pairs)} physics-scienceqa questions\n")
            return qa_pairs
            
        except Exception as e:
            print(f"⚠️  Could not load physics-scienceqa: {e}\n")
            return []
    
    def load_sciq_dataset(self) -> List[Tuple[str, str]]:
        """Load SciQ dataset (Science Q&A) (WORKING) ✅"""
        print("📥 Loading SciQ dataset...")
        
        try:
            dataset = load_dataset("sciq")
            qa_pairs = []
            
            # Filter for physics questions
            for item in tqdm(dataset['train'], desc="Processing SciQ"):
                question = item['question']
                correct_answer = item['correct_answer']
                if question and correct_answer:
                    qa_pairs.append((question, correct_answer))
            
            print(f"✅ Loaded {len(qa_pairs)} SciQ questions\n")
            return qa_pairs
            
        except Exception as e:
            print(f"⚠️  Could not load SciQ: {e}\n")
            return []
    
    def load_arc_dataset(self) -> List[Tuple[str, str]]:
        """Load ARC (AI2 Reasoning Challenge) dataset (WORKING) ✅"""
        print("📥 Loading ARC dataset...")
        
        try:
            # ARC has multiple splits
            dataset = load_dataset("ai2_arc", "ARC-Easy")
            qa_pairs = []
            
            for item in tqdm(dataset['train'], desc="Processing ARC"):
                question = item['question']
                choices = item['choices']['text']
                label = item['answerKey']
                
                # Convert label to index
                label_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
                if label in label_to_idx and label_to_idx[label] < len(choices):
                    correct_answer = choices[label_to_idx[label]]
                    if question and correct_answer:
                        qa_pairs.append((question, correct_answer))
            
            print(f"✅ Loaded {len(qa_pairs)} ARC questions\n")
            return qa_pairs
            
        except Exception as e:
            print(f"⚠️  Could not load ARC: {e}\n")
            return []
    
    def load_mmlu_physics(self) -> List[Tuple[str, str]]:
        """Load MMLU physics subset (WORKING) ✅"""
        print("📥 Loading MMLU physics dataset...")
        
        try:
            dataset = load_dataset("cais/mmlu", "physics")
            qa_pairs = []
            
            for item in tqdm(dataset['test'], desc="Processing MMLU Physics"):
                question = item['question']
                choices = item['choices']
                answer_idx = item['answer']
                
                if answer_idx < len(choices):
                    correct_answer = choices[answer_idx]
                    if question and correct_answer:
                        qa_pairs.append((question, correct_answer))
            
            print(f"✅ Loaded {len(qa_pairs)} MMLU physics questions\n")
            return qa_pairs
            
        except Exception as e:
            print(f"⚠️  Could not load MMLU physics: {e}\n")
            return []
    
    def load_all_datasets(self) -> List[Tuple[str, str, str]]:
        """
        Load all available physics datasets
        Returns: List of (question, answer, source)
        """
        print(f"\n{'='*60}")
        print("🚀 LOADING REAL PHYSICS DATASETS")
        print(f"{'='*60}\n")
        
        all_qa = []
        
        # Load datasets in order of priority
        datasets_to_load = [
            ("PhysicsScienceQA", self.load_physics_scienceqa),
            ("SciQ", self.load_sciq_dataset),
            ("ARC", self.load_arc_dataset),
            ("MMLU Physics", self.load_mmlu_physics)
        ]
        
        for name, loader_func in datasets_to_load:
            qa_pairs = loader_func()
            for q, a in qa_pairs:
                all_qa.append((q, a, name))
        
        print(f"\n{'='*60}")
        print(f"📊 TOTAL LOADED: {len(all_qa)} Q&A pairs")
        print(f"{'='*60}\n")
        
        return all_qa
    
    def save_combined_dataset(self, qa_data: List[Tuple[str, str, str]], output_file: str = "real_physics_qa.csv"):
        """Save combined dataset to CSV"""
        
        df = pd.DataFrame(qa_data, columns=['question', 'answer', 'source'])
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"💾 Saved {len(qa_data)} Q&A pairs to {output_file}\n")
        
        # Print statistics
        print(f"{'='*60}")
        print("📊 DATASET STATISTICS")
        print(f"{'='*60}")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"   {source}: {count} questions")
        print(f"{'='*60}\n")
        
        # Show sample
        print("🔍 Sample Questions:")
        for i, (q, a, source) in enumerate(qa_data[:3]):
            print(f"\n{i+1}. [{source}]")
            print(f"   Q: {q[:80]}...")
            print(f"   A: {a[:80]}...")


def main():
    """Load all physics datasets"""
    
    loader = PhysicsDatasetLoader()
    
    # Load all datasets
    all_qa = loader.load_all_datasets()
    
    if all_qa:
        # Save combined dataset
        loader.save_combined_dataset(all_qa)
        print("✅ Ready to use for training!")
    else:
        print("❌ No datasets loaded successfully")


if __name__ == "__main__":
    main()
