#!/usr/bin/env python3
"""
Main Automation Script for Physics Q&A System
Orchestrates the complete workflow with 3 optional parameters + automatic DNN training
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict

# Import core modules (always needed)
from textbook_kb_builder import TextbookKnowledgeBase
from train_classifier_dnn import train as train_classifier
from interactive_classifier_dnn import InteractiveQuestionClassifier
from merge_feedback_dnn import merge_feedback
from numerical_solver import NumericalSolver
from complete_rag_lora_comparison import CompleteRAGLoRAComparison

# Lazy imports for optional features (imported only when needed)
def import_lora_modules():
    """Import LoRA-related modules only when needed"""
    global PhysicsDatasetLoader, train_lora_small
    from load_physics_datasets import PhysicsDatasetLoader
    from train_lora_physics import train_lora_small
    return PhysicsDatasetLoader, train_lora_small


class PhysicsQASystem:
    """Main orchestrator for the complete Physics Q&A system"""
    
    def __init__(self, 
                 pdf_path: str = "Knowledge_Base/physics.pdf",
                 vector_db_dir: str = "./vector_db",
                 lora_model_dir: str = "./lora_finetuned_physics"):
        """
        Initialize the Physics Q&A System
        
        Args:
            pdf_path: Path to the physics textbook PDF
            vector_db_dir: Directory for vector database
            lora_model_dir: Directory for LoRA fine-tuned model
        """
        self.pdf_path = pdf_path
        self.vector_db_dir = vector_db_dir
        self.lora_model_dir = lora_model_dir
        
        print("=" * 70)
        print("🚀 PHYSICS Q&A SYSTEM INITIALIZING")
        print("=" * 70)
    
    def build_knowledge_base(self):
        """OPTIONAL STEP: Build knowledge base from PDF"""
        print("\n" + "=" * 70)
        print("📚 BUILDING KNOWLEDGE BASE (OPTIONAL)")
        print("=" * 70)
        
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"❌ PDF not found: {self.pdf_path}")
        
        # Build KB
        kb = TextbookKnowledgeBase(persist_dir=self.vector_db_dir)
        kb.build_knowledge_base(self.pdf_path)
        print("✅ Knowledge base built successfully!\n")
    
    def generate_physics_dataset(self):
        """OPTIONAL STEP: Generate physics dataset for LoRA training"""
        print("\n" + "=" * 70)
        print("📥 GENERATING PHYSICS DATASET (OPTIONAL)")
        print("=" * 70)
        
        try:
            PhysicsDatasetLoader, _ = import_lora_modules()
            loader = PhysicsDatasetLoader()
            all_qa = loader.load_all_datasets()
            # Saves to real_physics_qa.csv
            loader.save_combined_dataset(all_qa)
            print("✅ Physics dataset generated successfully!\n")
        except ImportError as e:
            print(f"❌ Missing dependency for dataset generation: {e}")
            print("   Install with: pip install datasets")
            sys.exit(1)
    
    def train_lora(self):
        """OPTIONAL STEP: Train LoRA model"""
        print("\n" + "=" * 70)
        print("🎯 TRAINING LoRA MODEL (OPTIONAL)")
        print("=" * 70)
        
        try:
            _, train_lora_small_fn = import_lora_modules()
            
            if not os.path.exists("real_physics_qa.csv"):
                print("⚠️  Dataset (real_physics_qa.csv) not found. Generating first...")
                self.generate_physics_dataset()
            
            train_lora_small_fn()
            print("✅ LoRA model trained successfully!\n")
        except ImportError as e:
            print(f"❌ Missing dependency for LoRA training: {e}")
            print("   Install with: pip install datasets")
            sys.exit(1)
    
    def train_dnn_classifier(self):
        """ALWAYS RUN: Train the DNN classifier"""
        print("\n" + "=" * 70)
        print("🧠 TRAINING QUESTION CLASSIFIER (ALWAYS RUN)")
        print("=" * 70)
        
        train_classifier()
        print("✅ Classifier trained successfully!\n")
    
    def classify_question_with_feedback(self, question: str):
        """
        Classify question and get user feedback, returns corrected label
        
        Args:
            question: The physics question to classify
            
        Returns:
            Tuple of (final_label, confidence) - final_label is corrected if user provided feedback
        """
        classifier = InteractiveQuestionClassifier()
        predicted_label, confidence = classifier.predict(question)
        
        print(f"🧠 Predicted label: {predicted_label} (confidence: {confidence:.2f})")
        response = input("Is this correct? (yes/no): ").strip().lower()
        
        final_label = predicted_label  # Default: use predicted label
        
        if response in ['yes', 'y']:
            # Save positive feedback
            classifier._save_feedback(question, predicted_label, predicted_label, confidence, correct=True)
            print("✅ Great! Thank you for confirming. Feedback saved!\n")
        
        elif response in ['no', 'n']:
            correct_label = input("Please provide the correct label (numerical/conceptual): ").strip().lower()
            if correct_label in ['numerical', 'conceptual']:
                # Save incorrect feedback and use corrected label
                classifier._save_feedback(question, predicted_label, correct_label, confidence, correct=False)
                final_label = correct_label  # USE CORRECTED LABEL FOR SOLVING
                print("✅ Thanks for the correction! Feedback saved for retraining.\n")
            else:
                print("❌ Invalid label. Using original prediction.\n")
        else:
            print("❌ Please answer with 'yes' or 'no'. Using original prediction.\n")
        
        return final_label, confidence
    
    def process_question(self, question: str) -> Dict:
        """
        Complete pipeline: Classify and process a question
        
        Args:
            question: The physics question to answer
            
        Returns:
            Dictionary containing the complete result
        """
        print("\n" + "=" * 70)
        print("🔍 CLASSIFYING QUESTION")
        print("=" * 70)
        print(f"Question: {question}\n")
        
        # Step 1: Classify question and get feedback (returns corrected label if user corrects)
        final_label, confidence = self.classify_question_with_feedback(question)
        
        print(f"Final label to use: {final_label} (confidence: {confidence:.2f})\n")
        
        # Step 2: Merge feedback
        print("=" * 70)
        print("📊 MERGING FEEDBACK")
        print("=" * 70)
        if os.path.exists("feedback.csv"):
            merge_feedback()
        else:
            print("ℹ️  No feedback to merge\n")
        
        # Step 3: Route to appropriate solver using CORRECTED label
        if final_label == "numerical":
            result = self._handle_numerical(question)
        else:  # conceptual
            result = self._handle_conceptual(question)
        
        # Add classification info (use corrected label)
        result["classification"] = {
            "type": final_label,
            "confidence": float(confidence)
        }
        
        return result
    
    def _handle_numerical(self, question: str) -> Dict:
        """Handle numerical question"""
        print("\n" + "=" * 70)
        print("🔢 SOLVING NUMERICAL QUESTION")
        print("=" * 70)
        
        solver = NumericalSolver()
        result = solver.solve(question)
        
        # Format output as JSON
        output = {
            "question": question,
            "type": "numerical",
            "timestamp": datetime.now().isoformat(),
            "solution": result.get("solution", ""),
            "model": result.get("model", ""),
            "success": result.get("success", False),
            "context_used": result.get("context_used", "")
        }
        
        return output
    
    def _handle_conceptual(self, question: str) -> Dict:
        """Handle conceptual question"""
        print("\n" + "=" * 70)
        print("💡 HANDLING CONCEPTUAL QUESTION")
        print("=" * 70)
        
        # Check if LoRA model exists
        lora_exists = os.path.exists(self.lora_model_dir) and \
                     os.path.exists(os.path.join(self.lora_model_dir, "adapter_config.json"))
        
        if not lora_exists:
            print("⚠️  LoRA model not found. Using base model only.")
            print("   Run with --train-lora-dataset and --train-lora to use LoRA.\n")
        
        # Run RAG + LoRA comparison
        system = CompleteRAGLoRAComparison(
            kb_path=self.vector_db_dir,
            lora_model_path=self.lora_model_dir if lora_exists else None
        )
        
        result = system.process_question(question)
        
        # Format output as JSON
        output = {
            "question": question,
            "type": "conceptual",
            "timestamp": result.get("timestamp", datetime.now().isoformat()),
            "base_response": result.get("base_response", ""),
            "lora_response": result.get("lora_response", "N/A - LoRA not trained"),
            "best_response": result.get("best_response", ""),
            "selected_model": result.get("selected_model", "Base Model"),
            "comparison": result.get("comparison", {}),
            "context": result.get("context", ""),
            "success": "error" not in result,
            "lora_available": lora_exists
        }
        
        return output
    
    def save_result(self, result: Dict, output_dir: str = "./results"):
        """Save result to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/result_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n💾 Result saved to: {filename}")
        return filename
    
    def interactive_mode(self):
        """Run in interactive mode for multiple questions"""
        print("\n" + "=" * 70)
        print("💬 INTERACTIVE MODE")
        print("=" * 70)
        print("Type your physics questions (or 'exit' to quit)")
        print("=" * 70)
        
        while True:
            question = input("\n❓ Enter your question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("👋 Exiting interactive mode...")
                break
            
            if not question:
                continue
            
            # Process question
            try:
                result = self.process_question(question)
                
                # Display result
                print("\n" + "=" * 70)
                print("📊 RESULT")
                print("=" * 70)
                print(f"Type: {result['type']}")
                
                if result['type'] == 'numerical':
                    print(f"\n{result['solution']}")
                else:
                    print(f"\nBest Answer ({result['selected_model']}):")
                    print(f"{result['best_response']}")
                
                # Save result
                self.save_result(result)
                
            except Exception as e:
                print(f"❌ Error processing question: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Physics Q&A System - Complete Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ask a question (minimal - just runs DNN training + classification + feedback)
  python main.py --question "What is Newton's first law?"
  
  # First time setup with everything
  python main.py --build-kb --train-lora-dataset --train-lora
  
  # Ask question in interactive mode with feedback loop
  python main.py --interactive
  
  # Rebuild KB only (when PDF changes)
  python main.py --build-kb
  
  # Generate dataset and train LoRA only
  python main.py --train-lora-dataset --train-lora
  
  # Ask question with all options
  python main.py --build-kb --train-lora-dataset --train-lora --question "Explain momentum"
        """
    )
    
    # Configuration
    parser.add_argument('--pdf', type=str, default='Knowledge_Base/physics.pdf',
                       help='Path to physics textbook PDF (default: Knowledge_Base/physics.pdf)')
    parser.add_argument('--vector-db', type=str, default='./vector_db',
                       help='Vector database directory')
    parser.add_argument('--lora-model', type=str, default='./lora_finetuned_physics',
                       help='LoRA model directory')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results')
    
    # Three optional parameters
    parser.add_argument('--build-kb', action='store_true',
                       help='OPTIONAL: Build knowledge base from PDF')
    parser.add_argument('--train-lora-dataset', action='store_true',
                       help='OPTIONAL: Generate physics dataset for LoRA training (creates real_physics_qa.csv)')
    parser.add_argument('--train-lora', action='store_true',
                       help='OPTIONAL: Train LoRA model (runs dataset generation if real_physics_qa.csv not found)')
    
    # Actions (always run DNN + classification)
    parser.add_argument('--question', type=str,
                       help='Single question to process')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = PhysicsQASystem(
            pdf_path=args.pdf,
            vector_db_dir=args.vector_db,
            lora_model_dir=args.lora_model
        )
        
        # OPTIONAL STEP 1: Build Knowledge Base
        if args.build_kb:
            system.build_knowledge_base()
        
        # OPTIONAL STEP 2: Generate Physics Dataset for LoRA
        if args.train_lora_dataset:
            system.generate_physics_dataset()
        
        # OPTIONAL STEP 3: Train LoRA
        if args.train_lora:
            # Auto-generate dataset if not exists
            if not os.path.exists("real_physics_qa.csv"):
                print("⚠️  Dataset (real_physics_qa.csv) not found. Auto-generating...")
                system.generate_physics_dataset()
            system.train_lora()
        
        # Check if user wants to process questions
        if args.question or args.interactive:
            # ALWAYS RUN: Train DNN Classifier
            system.train_dnn_classifier()
            
            # Process single question
            if args.question:
                # ALWAYS RUN: Interactive classifier + feedback + merge feedback
                result = system.process_question(args.question)
                
                # Display result
                print("\n" + "=" * 70)
                print("📊 FINAL RESULT")
                print("=" * 70)
                print(json.dumps(result, indent=2))
                
                # Save result
                system.save_result(result, output_dir=args.output_dir)
            
            # Interactive mode
            elif args.interactive:
                system.interactive_mode()
        
        # If only optional steps were run
        elif args.build_kb or args.train_lora_dataset or args.train_lora:
            print("\n" + "=" * 70)
            print("✅ OPTIONAL STEPS COMPLETED")
            print("=" * 70)
            print("Now you can ask questions with:")
            print("  python main.py --question \"Your question here\"")
            print("  python main.py --interactive")
        
        # No action specified
        else:
            print("\n⚠️  No action specified!")
            print("\nTo ask a question:")
            print("  python main.py --question \"What is velocity?\"")
            print("  python main.py --interactive")
            print("\nOptional setup (run once or when files change):")
            print("  python main.py --build-kb                    # Build KB from PDF (Knowledge_Base/physics.pdf)")
            print("  python main.py --train-lora-dataset          # Generate LoRA dataset (real_physics_qa.csv)")
            print("  python main.py --train-lora                  # Train LoRA model")
            print("\nRun 'python main.py --help' for more information")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
