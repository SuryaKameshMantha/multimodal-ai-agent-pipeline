# streamlit_app.py (completely updated)
import streamlit as st
import os
import sys
import logging
import json
import io
import subprocess
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import feedback writer
from feedback_writer import write_feedback, ensure_feedback_csv_exists

RESULTS_FOLDER = "./results"
Path(RESULTS_FOLDER).mkdir(exist_ok=True)

st.set_page_config(page_title="Physics Q&A System", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .thinking-box { border: 3px solid #1f77b4; border-radius: 15px; padding: 20px; 
        background: linear-gradient(135deg, #f0f7ff 0%, #e6f2ff 100%); margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .score-box { border-radius: 10px; padding: 15px; margin: 10px 0; font-size: 16px; font-weight: bold; }
    .numerical-score { background: linear-gradient(90deg, #d4edda 0%, #c3e6cb 100%); color: #155724; border-left: 5px solid #28a745; }
    .conceptual-score { background: linear-gradient(90deg, #cfe2ff 0%, #b6d4fe 100%); color: #084298; border-left: 5px solid #0d6efd; }
    .prediction-box { background: #fff3cd; border: 2px solid #ffc107; border-radius: 10px; padding: 15px; margin: 10px 0; font-weight: bold; }
    .comparison-box { border: 2px solid #0066cc; border-radius: 10px; padding: 15px; margin: 10px 0; background: #f0f9ff; }
    .quality-metrics { background: #f0f8ff; border: 2px solid #0066cc; border-radius: 10px; padding: 15px; margin: 10px 0; }
    .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 15px 0; }
    .metric-item { background: white; padding: 10px; border-radius: 8px; border-left: 4px solid #0066cc; }
    </style>
    """, unsafe_allow_html=True)

def save_comparison_json(question, dnn_scores, comparison_data, selected_model):
    """Save comparison results as JSON"""
    try:
        timestamp = datetime.now().isoformat().replace(":", "-").split(".")[0]
        filename = f"{RESULTS_FOLDER}/comparison_{timestamp}.json"

        full_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "dnn_classification": {
                "numerical_score": dnn_scores.get('numerical_score', 0),
                "conceptual_score": dnn_scores.get('conceptual_score', 0),
                "predicted_label": dnn_scores.get('predicted_label', ''),
                "confidence": dnn_scores.get('confidence', 0)
            },
            "comparison": comparison_data,
            "user_selection": {
                "selected_model": selected_model,
                "timestamp_selected": datetime.now().isoformat()
            }
        }

        with open(filename, 'w') as f:
            json.dump(full_data, f, indent=2)

        logger.info(f"✅ JSON saved: {filename}")
        return filename
    except Exception as e:
        logger.error(f"❌ Error saving JSON: {e}")
        return None

def run_merge_feedback():
    """Run merge_feedback_dnn to update CSVs"""
    try:
        result = subprocess.run([sys.executable, "merge_feedback_dnn.py"],
            capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            logger.info("✅ merge_feedback_dnn completed successfully")
            if result.stdout:
                logger.info(result.stdout)
            return True
        else:
            logger.warning(f"merge_feedback_dnn status: {result.returncode}")
            if result.stderr:
                logger.warning(result.stderr)
            return False
    except Exception as e:
        logger.warning(f"Could not run merge_feedback: {e}")
        return False

@st.cache_resource
def load_modules():
    try:
        logger.info("Loading modules...")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            from interactive_classifier_dnn import InteractiveQuestionClassifier
            from numerical_solver import NumericalSolver
            from complete_rag_lora_comparison import CompleteRAGLoRAComparison
            import torch

        logger.info("✅ All modules loaded")
        return {
            'InteractiveQuestionClassifier': InteractiveQuestionClassifier,
            'NumericalSolver': NumericalSolver,
            'CompleteRAGLoRAComparison': CompleteRAGLoRAComparison,
            'torch': torch
        }
    except Exception as e:
        logger.error(f"❌ Failed to load modules: {str(e)}")
        st.error(f"Failed to load modules: {str(e)}")
        return None

def get_dnn_scores(question, modules):
    try:
        logger.info(f"📊 Getting DNN scores for: {question[:50]}...")

        classifier = modules['InteractiveQuestionClassifier']()
        torch = modules['torch']

        X = classifier.vectorizer.transform([question]).toarray()
        X_tensor = torch.tensor(X, dtype=torch.float32).to('cpu')

        with torch.no_grad():
            outputs = classifier.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        numerical_score = float(probabilities[0][0].item()) * 100
        conceptual_score = float(probabilities[0][1].item()) * 100
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_label = classifier.label_map.get(predicted_idx, "UNKNOWN")
        confidence = max(numerical_score, conceptual_score)

        data = {
            "question": question,
            "numerical_score": round(numerical_score, 2),
            "conceptual_score": round(conceptual_score, 2),
            "predicted_label": predicted_label,
            "confidence": round(confidence, 2),
            "success": True
        }

        logger.info(f"✅ Scores: Numerical={data['numerical_score']}%, Conceptual={data['conceptual_score']}%")
        return data, None
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return None, str(e)

def get_comparison(question, label, modules):
    try:
        logger.info(f"🔍 Getting comparison for {label}: {question[:50]}...")

        if label.lower() == "numerical":
            solver = modules['NumericalSolver']()
            result = solver.solve(question)

            data = {
                "type": "numerical",
                "solution": result.get("solution", "No solution"),
                "model": result.get("model", "GROQ"),
                "success": True,
                "is_comparison": False
            }
        else:  # conceptual
            RAGSystem = modules['CompleteRAGLoRAComparison']

            lora_path = "./lora_finetuned_physics"
            lora_exists = os.path.exists(lora_path) and os.path.exists(os.path.join(lora_path, "adapter_config.json"))

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                system = RAGSystem(kb_path="./vector_db", lora_model_path=lora_path if lora_exists else None)
                result = system.process_question(question)

            # Extract metrics from comparison
            comparison_metrics = result.get("comparison", {})
            base_metrics = comparison_metrics.get("base_model", {}).get("metrics", {})
            lora_metrics = comparison_metrics.get("lora_model", {}).get("metrics", {})

            data = {
                "type": "conceptual",
                "is_comparison": True,
                "base_response": result.get("base_response", "No response"),
                "lora_response": result.get("lora_response", "N/A"),
                "best_response": result.get("best_response", "No response"),
                "selected_model": result.get("selected_model", "Base Model"),
                "comparison_metrics": comparison_metrics,
                "base_metrics": base_metrics,
                "lora_metrics": lora_metrics,
                "lora_available": lora_exists,
                "success": True
            }

        logger.info(f"✅ Comparison received for {label}")
        return data, None
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return None, str(e)

def display_conceptual_metrics(comparison):
    """Display detailed metrics for conceptual questions"""
    st.markdown("### 📊 Detailed Comparison Metrics")
    
    base_metrics = comparison.get('base_metrics', {})
    lora_metrics = comparison.get('lora_metrics', {})
    lora_available = comparison.get('lora_available', False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Base Model Metrics")
        st.metric("Overall Quality", f"{base_metrics.get('overall_quality', 0):.3f}")
        st.metric("Question Relevance", f"{base_metrics.get('question_relevance', 0):.3f}")
        st.metric("Context Alignment", f"{base_metrics.get('context_alignment', 0):.3f}")
        st.metric("Word Count", base_metrics.get('word_count', 0))
        st.metric("Sentence Count", base_metrics.get('sentence_count', 0))
    
    with col2:
        if lora_available:
            st.subheader("🧠 LoRA Model Metrics")
            st.metric("Overall Quality", f"{lora_metrics.get('overall_quality', 0):.3f}")
            st.metric("Question Relevance", f"{lora_metrics.get('question_relevance', 0):.3f}")
            st.metric("Context Alignment", f"{lora_metrics.get('context_alignment', 0):.3f}")
            st.metric("Word Count", lora_metrics.get('word_count', 0))
            st.metric("Sentence Count", lora_metrics.get('sentence_count', 0))
        else:
            st.subheader("🧠 LoRA Model Metrics")
            st.warning("LoRA Model Not Available")
    
    # Show winner and decision
    comparison_data = comparison.get('comparison_metrics', {})
    winner = comparison_data.get('winner', 'Base Model')
    reason = comparison_data.get('decision_reason', '')
    
    st.markdown(f"### 🏆 Selected Model: **{winner}**")
    if reason:
        st.info(f"**Reason:** {reason}")

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'current_scores' not in st.session_state:
    st.session_state.current_scores = None
if 'current_comparison' not in st.session_state:
    st.session_state.current_comparison = None
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = None
if 'confirmed_label' not in st.session_state:
    st.session_state.confirmed_label = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

modules = load_modules()
if modules is None:
    st.error("Failed to load modules")
    st.stop()

# Ensure feedback.csv exists
ensure_feedback_csv_exists()

st.title("🔬 Physics Q&A System")
st.markdown("### Interactive AI Agent with DNN Classification & Model Comparison")

with st.sidebar:
    st.header("⚙️ System Status")
    
    # Check if DNN training files exist
    dnn_trained = os.path.exists("question_classifier_dnn.pt") and os.path.exists("tfidf_vectorizer.joblib")
    kb_exists = os.path.exists("vector_db")
    lora_exists = os.path.exists("lora_finetuned_physics") and os.path.exists(os.path.join("lora_finetuned_physics", "adapter_config.json"))
    dataset_exists = os.path.exists("real_physics_qa.csv")
    
    status_items = {
        "DNN Classifier": dnn_trained,
        "Knowledge Base": kb_exists, 
        "LoRA Model": lora_exists,
        "Physics Dataset": dataset_exists
    }
    
    for item, status in status_items.items():
        st.write(f"{'✅' if status else '❌'} {item}")

    st.divider()
    st.subheader("📁 Results Storage")
    st.write(f"✅ JSON: `{RESULTS_FOLDER}/`")
    st.write("✅ Feedback: `feedback.csv`")

    # Count files
    json_count = len([f for f in os.listdir(RESULTS_FOLDER) if f.endswith('.json')]) if os.path.exists(RESULTS_FOLDER) else 0
    st.write(f"📊 Comparisons saved: {json_count}")
    
    st.divider()
    st.subheader("🔄 Actions")
    if st.button("🔄 Run DNN Training", use_container_width=True):
        with st.spinner("Training DNN classifier..."):
            try:
                from train_classifier_dnn import train
                train()
                st.success("DNN training completed!")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")
    
    if st.button("🏗️ Build Knowledge Base", use_container_width=True):
        with st.spinner("Building knowledge base..."):
            try:
                from textbook_kb_builder import TextbookKnowledgeBase
                kb = TextbookKnowledgeBase(persist_dir="./vector_db")
                kb.build_knowledge_base("Knowledge_Base/physics.pdf")
                st.success("Knowledge base built!")
                st.rerun()
            except Exception as e:
                st.error(f"KB build failed: {e}")
    
    if st.button("🚀 Train LoRA Model", use_container_width=True):
        with st.spinner("Training LoRA model..."):
            try:
                from train_lora_physics import train_lora_small
                train_lora_small()
                st.success("LoRA training completed!")
                st.rerun()
            except Exception as e:
                st.error(f"LoRA training failed: {e}")

tab1, tab2, tab3, tab4 = st.tabs(["❓ Ask Questions", "📤 Upload & Train", "📚 Examples", "ℹ️ Help"])

# TAB 1: Ask Questions
with tab1:
    st.header("Step 1: Ask Your Question")
    question = st.text_area("Enter your physics question:", 
        placeholder="e.g., 'What is Newton's first law?'", height=100, value=st.session_state.current_question)
    if question:
        st.session_state.current_question = question

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Analyze Question", type="primary", use_container_width=True):
            if question.strip():
                # First ensure DNN is trained
                if not os.path.exists("question_classifier_dnn.pt"):
                    with st.spinner("🤖 Training DNN classifier first time..."):
                        from train_classifier_dnn import train
                        train()
                
                with st.spinner("🤔 Running DNN classifier..."):
                    scores, error = get_dnn_scores(question, modules)
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.session_state.current_scores = scores
                    st.session_state.step = 1
                    st.rerun()
            else:
                st.warning("⚠️ Please enter a question")

    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.step = 0
            st.session_state.current_question = ""
            st.session_state.current_scores = None
            st.session_state.current_comparison = None
            st.session_state.current_answer = None
            st.session_state.confirmed_label = None
            st.session_state.selected_model = None
            st.rerun()

    # STEP 2: Show DNN Scores
    if st.session_state.step >= 1 and st.session_state.current_scores:
        scores = st.session_state.current_scores
        st.divider()
        st.header("Step 2: Review DNN Classification")
        st.markdown('<div class="thinking-box"><h3>🧠 DNN Classification Analysis</h3></div>', unsafe_allow_html=True)

        st.markdown("**Question being analyzed:**")
        st.info(f'"{question}"')
        st.markdown("---")
        st.markdown("### DNN Confidence Scores:")

        col1, col2 = st.columns(2)
        with col1:
            num_score = scores.get('numerical_score', 0)
            st.markdown(f'<div class="score-box numerical-score">📊 Numerical: <span style="font-size: 24px;">{num_score}%</span></div>',
                unsafe_allow_html=True)
            st.progress(min(100, max(0, num_score)) / 100)
        with col2:
            con_score = scores.get('conceptual_score', 0)
            st.markdown(f'<div class="score-box conceptual-score">💡 Conceptual: <span style="font-size: 24px;">{con_score}%</span></div>',
                unsafe_allow_html=True)
            st.progress(min(100, max(0, con_score)) / 100)

        st.markdown("---")
        prediction = scores.get('predicted_label', 'UNKNOWN')
        confidence = scores.get('confidence', 0)
        st.markdown(f'<div class="prediction-box">🎯 DNN Prediction: {prediction.upper()}<br>Confidence: {confidence}%</div>',
            unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ✅ Is this classification CORRECT?")
        col1, col2 = st.columns(2)

        with col1:
            if st.button(f"✅ YES - {prediction}", type="primary", use_container_width=True):
                st.session_state.confirmed_label = prediction
                with st.spinner("⏳ Getting answer..."):
                    comparison, error = get_comparison(question, prediction, modules)
                if error:
                    st.error(f"❌ {error}")
                else:
                    # Log feedback - user confirmed DNN prediction
                    write_feedback(question, scores.get('predicted_label'), prediction, 
                                 confidence=scores.get('confidence', 0), confirmed=True)
                    st.session_state.current_comparison = comparison
                    st.session_state.step = 3  # Skip to final answer directly
                    st.rerun()

        with col2:
            if st.button("❌ NO - Override", use_container_width=True):
                st.session_state.show_override = True
                st.rerun()

        if st.session_state.get('show_override', False):
            st.markdown("---")
            st.markdown("### 🔄 What should it be?")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔢 NUMERICAL", use_container_width=True):
                    st.session_state.confirmed_label = "numerical"
                    with st.spinner("⏳ Getting answer..."):
                        comparison, error = get_comparison(question, "numerical", modules)
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        # Log feedback - user overrode to numerical
                        write_feedback(question, scores.get('predicted_label'), "numerical", 
                                     confidence=scores.get('confidence', 0), confirmed=False)
                        st.session_state.current_comparison = comparison
                        st.session_state.show_override = False
                        st.session_state.step = 3  # Skip to final answer directly
                        st.rerun()

            with col2:
                if st.button("💡 CONCEPTUAL", use_container_width=True):
                    st.session_state.confirmed_label = "conceptual"
                    with st.spinner("⏳ Getting answer..."):
                        comparison, error = get_comparison(question, "conceptual", modules)
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        # Log feedback - user overrode to conceptual
                        write_feedback(question, scores.get('predicted_label'), "conceptual", 
                                     confidence=scores.get('confidence', 0), confirmed=False)
                        st.session_state.current_comparison = comparison
                        st.session_state.show_override = False
                        st.session_state.step = 3  # Skip to final answer directly
                        st.rerun()

    # STEP 3: Final Answer (Combined step for both question types)
    if st.session_state.step >= 3 and st.session_state.current_comparison:
        comparison = st.session_state.current_comparison
        label = st.session_state.confirmed_label

        st.divider()
        st.header("🎯 Final Answer")

        # Show question type and model info
        col1, col2, col3 = st.columns(3)
        with col1:
            q_type = comparison.get('type', 'Unknown').upper()
            icon = '📊' if q_type == 'NUMERICAL' else '💡'
            st.metric("Question Type", f"{icon} {q_type}")
        with col2:
            if comparison.get('type') == 'numerical':
                model = comparison.get('model', 'GROQ')
            else:
                model = comparison.get('selected_model', 'Base Model')
            st.metric("Model Used", model)
        with col3:
            st.metric("Status", "✅ Complete")

        st.markdown("---")

        if comparison.get('type') == 'numerical':
            # Numerical response
            st.subheader("📊 Solution")
            st.text_area("Solution", value=comparison.get('solution', ''), height=300, 
                        disabled=True, label_visibility="collapsed", key="numerical_solution")
            
        else:
            # Conceptual response - show metrics and best response
            display_conceptual_metrics(comparison)
            
            st.markdown("---")
            st.subheader("💡 Best Answer")
            st.text_area("Answer", value=comparison.get('best_response', ''), height=300, 
                        disabled=True, label_visibility="collapsed", key="conceptual_answer")

        # Save results and run merge feedback
        if st.button("💾 Save Results & Continue", type="primary", use_container_width=True):
            # Save JSON
            save_comparison_json(question, st.session_state.current_scores or {}, comparison, 
                               comparison.get('selected_model', comparison.get('model', 'Unknown')))
            
            # Run merge feedback
            with st.spinner("📊 Updating training data..."):
                run_merge_feedback()
            
            st.success("✅ Results saved and feedback merged!")
            
            # Reset for next question but keep current question
            st.session_state.current_scores = None
            st.session_state.current_comparison = None
            st.session_state.current_answer = None
            st.session_state.confirmed_label = None
            st.session_state.selected_model = None
            st.session_state.step = 1  # Go back to classification step
            st.rerun()

# TAB 2: Upload & Train
with tab2:
    st.header("📤 System Setup & Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Knowledge Base")
        st.info("Build vector database from PDF textbook")
        if st.button("🏗️ Build Knowledge Base", use_container_width=True, type="secondary"):
            with st.spinner("Building knowledge base from PDF..."):
                try:
                    from textbook_kb_builder import TextbookKnowledgeBase
                    kb = TextbookKnowledgeBase(persist_dir="./vector_db")
                    kb.build_knowledge_base("Knowledge_Base/physics.pdf")
                    st.success("✅ Knowledge base built successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to build KB: {e}")
    
    with col2:
        st.subheader("🧠 LoRA Training")
        st.info("Fine-tune model on physics dataset")
        if st.button("🚀 Train LoRA Model", use_container_width=True, type="secondary"):
            with st.spinner("Training LoRA model..."):
                try:
                    # First check if dataset exists
                    if not os.path.exists("real_physics_qa.csv"):
                        st.warning("📥 Generating physics dataset first...")
                        from load_physics_datasets import PhysicsDatasetLoader
                        loader = PhysicsDatasetLoader()
                        all_qa = loader.load_all_datasets()
                        loader.save_combined_dataset(all_qa)
                    
                    from train_lora_physics import train_lora_small
                    train_lora_small()
                    st.success("✅ LoRA model trained successfully!")
                except Exception as e:
                    st.error(f"❌ LoRA training failed: {e}")

# TAB 3: Examples
with tab3:
    st.header("📚 Example Questions")
    
    st.markdown("### 💡 Conceptual Questions")
    conceptual_examples = [
        "What is Newton's first law of motion?",
        "Explain the concept of conservation of energy",
        "What is the difference between speed and velocity?",
        "Describe how electromagnetic induction works"
    ]
    
    for ex in conceptual_examples:
        if st.button(f"→ {ex}", use_container_width=True, key=f"conc_{ex}"):
            st.session_state.current_question = ex
            st.rerun()
    
    st.markdown("### 📊 Numerical Questions") 
    numerical_examples = [
        "Calculate the kinetic energy of a 10kg object moving at 5m/s",
        "What force is required to accelerate a 50kg object at 2m/s²?",
        "A car accelerates from 0 to 60 km/h in 5 seconds. What is its acceleration?",
        "Calculate the gravitational force between two 100kg masses 1 meter apart"
    ]
    
    for ex in numerical_examples:
        if st.button(f"→ {ex}", use_container_width=True, key=f"num_{ex}"):
            st.session_state.current_question = ex
            st.rerun()

# TAB 4: Help
with tab4:
    st.header("ℹ️ How It Works")
    
    st.markdown("""
    ## 🔄 Automatic DNN Training
    
    **Every time you ask a question:**
    - System checks if DNN classifier is trained
    - If not found, automatically trains the classifier
    - Uses `numerical_questions.csv` and `conceptual_questions.csv`
    - Saves model as `question_classifier_dnn.pt`
    
    ## 🎯 Streamlined Question Flow
    
    **Step 1:** Ask question → DNN classification
    **Step 2:** Confirm/override classification → Automatic answer generation
    **Step 3:** View final answer with metrics (conceptual) or solution (numerical)
    
    ## 📊 Conceptual Questions
    - **No user selection required** - system automatically picks best model
    - **Detailed metrics** shown: Overall Quality, Question Relevance, Context Alignment
    - **Word/Sentence counts** for response quality assessment
    - **Automatic model selection** based on comparison metrics
    
    ## 🔢 Numerical Questions  
    - **Direct solution** without quality metrics
    - **Step-by-step** chain-of-thought reasoning
    - **Clean formatting** without LaTeX
    
    ## 🏗️ System Components
    - **Knowledge Base**: Vector database from PDF textbook
    - **LoRA Model**: Fine-tuned physics specialist
    - **DNN Classifier**: Question type detection
    - **Feedback Loop**: Continuous improvement via user corrections
    """)
    
    st.markdown("---")
    st.subheader("📁 File Structure")
    
    st.code("""
    📁 Physics-QA-System/
    ├── 📄 streamlit_app.py          # Main web interface
    ├── 📄 main.py                   # Command-line interface  
    ├── 📄 feedback_writer.py        # Feedback logging
    ├── 📄 merge_feedback_dnn.py     # Training data updates
    ├── 📁 Knowledge_Base/
    │   └── 📄 physics.pdf           # Textbook PDF
    ├── 📁 results/                  # JSON result storage
    ├── 📄 feedback.csv              # User feedback
    ├── 📄 numerical_questions.csv   # Numerical training data
    ├── 📄 conceptual_questions.csv  # Conceptual training data
    ├── 📄 question_classifier_dnn.pt    # Trained DNN model
    └── 📁 lora_finetuned_physics/   # Fine-tuned LoRA model
    """)

st.divider()
st.markdown('<div style="text-align:center;color:gray;font-size:11px;padding:15px;">Physics Q&A System | Auto-DNN Training | Smart Model Selection | Continuous Learning</div>',
    unsafe_allow_html=True)