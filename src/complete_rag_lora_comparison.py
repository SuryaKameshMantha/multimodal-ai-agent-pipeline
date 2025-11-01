# complete_rag_lora_comparison_enhanced.py
import torch
import pandas as pd
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer, util
import numpy as np
from datetime import datetime
import json
import os
import textwrap

class CompleteRAGLoRAComparison:
    def __init__(self, kb_path=None, lora_model_path="./lora_finetuned_physics"):
        self.device = torch.device("cpu")
        print(f"🖥️  Using device: {self.device}")
        
        # Initialize components
        self.knowledge_base = self.setup_knowledge_base(kb_path)
        self.base_model = self.setup_base_model()
        self.lora_model = self.setup_lora_model(lora_model_path)
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("🚀 Complete RAG + LoRA Comparison System Initialized!")
    
    def setup_knowledge_base(self, kb_path):
        """Setup the knowledge base for retrieval"""
        from textbook_kb_builder import TextbookKnowledgeBase
        return TextbookKnowledgeBase(persist_dir=kb_path)
    
    def setup_base_model(self):
        """Setup base FLAN-T5 model without fine-tuning"""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        model_name = "google/flan-t5-base"
        print(f"📥 Loading base model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None
        )
        
        model = model.to(self.device)
        model.eval()
        
        return {"model": model, "tokenizer": tokenizer, "name": "Base Model"}
    
    def setup_lora_model(self, lora_path):
        """Setup LoRA fine-tuned FLAN-T5 model"""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from peft import PeftModel
        
        if not os.path.exists(lora_path):
            print(f"❌ LoRA model directory not found: {lora_path}")
            return None
            
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        existing_files = os.listdir(lora_path)
        missing_files = [f for f in required_files if f not in existing_files]
        
        if missing_files:
            print(f"❌ LoRA model missing files: {missing_files}")
            return None
        
        base_model_name = "google/flan-t5-base"
        print(f"📥 Loading LoRA model from: {lora_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map=None
        )
        
        try:
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.to(self.device)
            model.eval()
            
            print("✅ LoRA model loaded successfully!")
            return {"model": model, "tokenizer": tokenizer, "name": "LoRA Fine-tuned"}
            
        except Exception as e:
            print(f"❌ Error loading LoRA model: {e}")
            return None
    
    def retrieve_context(self, question: str, top_k: int = 3) -> str:
        """Retrieve relevant context from knowledge base"""
        try:
            retrieved = self.knowledge_base.retrieve(question, top_k=top_k)
            
            if not retrieved:
                return ""
            
            context_parts = []
            for i, doc in enumerate(retrieved, 1):
                page = doc['metadata']['page']
                text = doc['text']
                context_parts.append(f"[Source {i}, Page {page}]: {text}")
            
            return "\n\n".join(context_parts)
        except Exception as e:
            print(f"❌ Error retrieving context: {e}")
            return ""
    
    def generate_response(self, model_info: Dict, question: str, context: str, max_length: int = 500) -> str:
        """Generate response using specified model with enhanced parameters"""
        try:
            prompt = self.create_enhanced_prompt(question, context)
            
            inputs = model_info["tokenizer"](
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model_info["model"].generate(
                    **inputs,
                    max_new_tokens=max_length,  # Increased for longer responses
                    temperature=0.8,  # Slightly higher for more creativity
                    do_sample=True,
                    pad_token_id=model_info["tokenizer"].pad_token_id,
                    repetition_penalty=1.2,  # Increased to reduce repetition
                    num_beams=4,  # Use beam search for better quality
                    early_stopping=True,
                    no_repeat_ngram_size=3  # Prevent repeating 3-grams
                )
            
            full_response = model_info["tokenizer"].decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer from response
            if "ANSWER:" in full_response:
                answer = full_response.split("ANSWER:")[-1].strip()
            elif "Answer:" in full_response:
                answer = full_response.split("Answer:")[-1].strip()
            else:
                answer = full_response
            
            # Clean up the response
            answer = self.clean_response(answer)
            
            return answer
            
        except Exception as e:
            print(f"❌ Error generating response with {model_info['name']}: {e}")
            return f"Error: Could not generate response with {model_info['name']}"
    
    def create_enhanced_prompt(self, question: str, context: str) -> str:
        """Create enhanced prompt for longer, more detailed responses"""
        if context and len(context) > 50:
            return f"""You are a physics expert. Based on the textbook content below, provide a comprehensive and detailed answer to the question.

TEXTBOOK CONTENT:
{context}

QUESTION: {question}

Please provide a thorough explanation that covers:
1. The key concepts involved
2. Relevant principles or laws
3. Clear reasoning
4. Practical implications if applicable

ANSWER:"""
        else:
            return f"""You are a physics expert. Provide a comprehensive and detailed answer to this question.

QUESTION: {question}

Please provide a thorough explanation that covers the key concepts, relevant principles, and clear reasoning.

ANSWER:"""
    
    def clean_response(self, response: str) -> str:
        """Clean and format the response for better readability"""
        # Remove any trailing incomplete sentences
        response = response.strip()
        
        # Ensure the response ends with proper punctuation
        if response and not response[-1] in ['.', '!', '?']:
            response += '.'
            
        # Remove any repetitive patterns
        sentences = response.split('. ')
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if clean_sentence and clean_sentence not in seen_sentences:
                seen_sentences.add(clean_sentence)
                unique_sentences.append(clean_sentence)
        
        return '. '.join(unique_sentences)
    
    def calculate_metrics(self, question: str, response: str, context: str) -> Dict:
        """Calculate quality metrics for a response"""
        metrics = {}
        
        try:
            # 1. Response Length
            metrics["response_length"] = len(response)
            metrics["word_count"] = len(response.split())
            metrics["sentence_count"] = len([s for s in response.split('.') if s.strip()])
            
            # 2. Semantic Relevance to Question
            question_embedding = self.similarity_model.encode([question], convert_to_tensor=True)
            response_embedding = self.similarity_model.encode([response], convert_to_tensor=True)
            metrics["question_relevance"] = float(util.pytorch_cos_sim(question_embedding, response_embedding)[0][0])
            
            # 3. Context Utilization (if context available)
            if context and len(context) > 50:
                context_embedding = self.similarity_model.encode([context], convert_to_tensor=True)
                metrics["context_alignment"] = float(util.pytorch_cos_sim(response_embedding, context_embedding)[0][0])
            else:
                metrics["context_alignment"] = 0.0
            
            # 4. Response Quality Score
            length_score = min(metrics["word_count"] / 100, 1.0)  # Bonus for responses > 100 words
            sentence_score = min(metrics["sentence_count"] / 3, 1.0)  # Bonus for multiple sentences
            
            metrics["overall_quality"] = (
                metrics["question_relevance"] * 0.4 +
                metrics["context_alignment"] * 0.3 +
                length_score * 0.2 +
                sentence_score * 0.1
            )
            
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            metrics = {
                "response_length": len(response),
                "word_count": len(response.split()),
                "sentence_count": len([s for s in response.split('.') if s.strip()]),
                "question_relevance": 0.5,
                "context_alignment": 0.5,
                "overall_quality": 0.5
            }
        
        return metrics
    
    def compare_responses(self, question: str, base_response: str, lora_response: str, context: str) -> Dict:
        """Compare base and LoRA model responses"""
        print("\n🔍 COMPARING RESPONSES")
        print("=" * 50)
        
        # Calculate metrics for both responses
        base_metrics = self.calculate_metrics(question, base_response, context)
        lora_metrics = self.calculate_metrics(question, lora_response, context)
        
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "context_length": len(context),
            "base_model": {
                "response": base_response,
                "metrics": base_metrics
            },
            "lora_model": {
                "response": lora_response, 
                "metrics": lora_metrics
            },
            "winner": None,
            "decision_reason": ""
        }
        
        # Determine winner
        base_quality = base_metrics["overall_quality"]
        lora_quality = lora_metrics["overall_quality"]
        
        # Skip comparison if either response has error
        if "Error:" in base_response or "Error:" in lora_response:
            if "Error:" in base_response and "Error:" not in lora_response:
                comparison["winner"] = "LoRA Model"
                comparison["decision_reason"] = "Base model failed, LoRA model succeeded"
            elif "Error:" not in base_response and "Error:" in lora_response:
                comparison["winner"] = "Base Model"
                comparison["decision_reason"] = "LoRA model failed, Base model succeeded"
            else:
                comparison["winner"] = "Base Model"
                comparison["decision_reason"] = "Both models had errors, defaulting to Base"
        else:
            quality_difference = lora_quality - base_quality
            improvement_percentage = (quality_difference / base_quality * 100) if base_quality > 0 else 0
            
            if lora_quality > base_quality + 0.05:
                comparison["winner"] = "LoRA Model"
                comparison["decision_reason"] = f"LoRA shows {improvement_percentage:+.1f}% quality improvement"
            elif base_quality > lora_quality + 0.05:
                comparison["winner"] = "Base Model" 
                comparison["decision_reason"] = f"Base model performs better by {abs(improvement_percentage):.1f}%"
            else:
                comparison["winner"] = "Tie"
                comparison["decision_reason"] = "Models perform similarly (within 5% difference)"
        
        return comparison
    
    def display_comparison(self, comparison: Dict):
        """Display detailed comparison results with better formatting"""
        print(f"\n📊 COMPARISON RESULTS")
        print("=" * 60)
        
        base_metrics = comparison["base_model"]["metrics"]
        lora_metrics = comparison["lora_model"]["metrics"]
        
        print(f"🏆 WINNER: {comparison['winner']}")
        print(f"📝 Reason: {comparison['decision_reason']}")
        
        print(f"\n📈 METRICS COMPARISON:")
        print(f"{'Metric':<25} {'Base Model':<12} {'LoRA Model':<12} {'Difference':<12}")
        print("-" * 65)
        
        metrics_to_show = [
            ("Overall Quality", "overall_quality"),
            ("Question Relevance", "question_relevance"), 
            ("Context Alignment", "context_alignment"),
            ("Word Count", "word_count"),
            ("Sentence Count", "sentence_count")
        ]
        
        for metric_name, metric_key in metrics_to_show:
            base_val = base_metrics[metric_key]
            lora_val = lora_metrics[metric_key]
            diff = lora_val - base_val
            
            if metric_key in ["word_count", "sentence_count"]:
                print(f"{metric_name:<25} {base_val:<12} {lora_val:<12} {diff:+.1f}")
            else:
                print(f"{metric_name:<25} {base_val:<12.3f} {lora_val:<12.3f} {diff:+.3f}")
    
    def format_response_display(self, response: str, title: str) -> str:
        """Format response for clean display with proper wrapping"""
        print(f"\n{title}")
        print("=" * 60)
        
        # Wrap text for better readability
        wrapped_text = textwrap.fill(response, width=80)
        print(wrapped_text)
        
        return wrapped_text
    
    def process_question(self, question: str) -> Dict:
        """Complete processing pipeline for a single question"""
        print(f"\n{'='*70}")
        print(f"🎯 PROCESSING QUESTION: {question}")
        print(f"{'='*70}")
        
        result = {
            "question": question,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Retrieve context from knowledge base
            print("\n1️⃣ STEP 1: RETRIEVING CONTEXT FROM KNOWLEDGE BASE")
            context = self.retrieve_context(question, top_k=3)
            result["context"] = context
            print(f"✅ Retrieved context ({len(context)} characters)")
            
            # Step 2: Generate response with BASE MODEL
            print("\n2️⃣ STEP 2: BASE MODEL GENERATION")
            base_response = self.generate_response(self.base_model, question, context)
            result["base_response"] = base_response
            self.format_response_display(base_response, "🔷 BASE MODEL RESPONSE")
            
            # Step 3: Generate response with LoRA MODEL  
            lora_response = None
            if self.lora_model is not None:
                print("\n3️⃣ STEP 3: LoRA MODEL GENERATION")
                lora_response = self.generate_response(self.lora_model, question, context)
                result["lora_response"] = lora_response
                self.format_response_display(lora_response, "🔹 LoRA MODEL RESPONSE")
            else:
                print("\n3️⃣ STEP 3: LoRA MODEL - SKIPPED (Model not available)")
                lora_response = "LoRA model not available"
                result["lora_response"] = lora_response
            
            # Step 4: Compare responses
            print("\n4️⃣ STEP 4: COMPARING RESPONSES")
            comparison = self.compare_responses(question, base_response, lora_response, context)
            result["comparison"] = comparison
            
            # Step 5: Display results
            self.display_comparison(comparison)
            
            # Step 6: Return best response
            if comparison["winner"] == "LoRA Model" and "Error:" not in lora_response and "not available" not in lora_response:
                best_response = lora_response
            else:
                best_response = base_response
            
            result["best_response"] = best_response
            result["selected_model"] = comparison["winner"]
            
            print(f"\n{'='*70}")
            print(f"✅ FINAL ANSWER (Selected from {comparison['winner']})")
            print(f"{'='*70}")
            self.format_response_display(best_response, "")
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)
            return result

    def simple_test(self):
        """Simple test to verify the system works"""
        print("🧪 RUNNING ENHANCED TEST...")
        
        test_question = "What are scalars and vectors in physics?"
        print(f"Test question: {test_question}")
        
        result = self.process_question(test_question)
        
        print(f"\n🎯 ENHANCED TEST COMPLETED!")
        print(f"Base Model Word Count: {len(result.get('base_response', '').split())}")
        print(f"LoRA Model Word Count: {len(result.get('lora_response', '').split())}")
        print(f"Winner: {result.get('selected_model', 'N/A')}")
        
        return result


def main():
    """Main function to run the enhanced system"""
    
    print("🚀 INITIALIZING ENHANCED RAG + LoRA COMPARISON SYSTEM")
    
    try:
        system = CompleteRAGLoRAComparison()
        
        # Run a simple test first
        print("\n" + "="*70)
        print("🧪 RUNNING ENHANCED INITIAL TEST")
        print("="*70)
        test_result = system.simple_test()
        
        if "error" in test_result:
            print("❌ Initial test failed. Please check the error above.")
            return
        
        # Main interaction loop
        test_questions = [
            "What is Newton's first law of motion and provide examples?",
            "Explain the concept of conservation of energy with real-world applications.",
            "How does electromagnetic induction work in generators and transformers?",
        ]
        
        while True:
            print(f"\n{'='*70}")
            print("🎯 CHOOSE MODE:")
            print("1. Single question")
            print("2. Batch process test questions")
            print("3. Exit")
            
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == "1":
                question = input("\nEnter your physics question: ").strip()
                if question:
                    result = system.process_question(question)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    with open(f"result_{timestamp}.json", 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"\n💾 Result saved to: result_{timestamp}.json")
            
            elif choice == "2":
                print(f"\n🔄 Processing {len(test_questions)} enhanced test questions...")
                results = []
                for question in test_questions:
                    result = system.process_question(question)
                    results.append(result)
                
                with open("batch_results_enhanced.json", 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"💾 Enhanced batch results saved to: batch_results_enhanced.json")
            
            elif choice == "3":
                print("👋 Exiting...")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
    except Exception as e:
        print(f"❌ Fatal error initializing system: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()