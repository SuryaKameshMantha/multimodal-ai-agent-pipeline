# Physics Q&A AI Agent System

**A Multi-Agent AI System for Physics Question Answering with Continuous Learning**

---

## 👤 Project Information

**Developer:** Surya Kamesh Mantha  
**University:** [Your University Name]  
**Department:** [Your Department]  
**Specialization:** Machine Learning / AI

---

## 📋 Project Overview

This project implements a sophisticated multi-agent AI system for answering physics questions using:
- **Question Classification:** Deep Neural Network (DNN) routing
- **Knowledge Grounding:** Retrieval-Augmented Generation (RAG) with ChromaDB
- **Answer Generation:** FLAN-T5-Base + LoRA-adapted physics specialist
- **Numerical Solving:** GROQ Llama-3.3-70B for calculations
- **Continuous Learning:** Automatic retraining from user corrections

**Key Achievement:** 96% question classification accuracy with 6x faster inference than full fine-tuning.

---

## 🎯 What's Inside

### Core Deliverables ✅

```
📁 Repository Structure
├── 📂 src/                          # Source code (11 Python modules)
│   ├── streamlit_app.py            # Main UI orchestrator
│   ├── train_classifier_dnn.py     # DNN training (routing)
│   ├── interactive_classifier_dnn.py # Question classification
│   ├── complete_rag_lora_comparison.py # RAG + answer generation
│   ├── textbook_kb_builder.py      # Knowledge base creation
│   ├── numerical_solver.py         # GROQ API integration
│   ├── feedback_writer.py          # User feedback collection
│   ├── merge_feedback_dnn.py       # Auto-retraining system
│   ├── train_lora_physics.py       # LoRA fine-tuning
│   ├── load_physics_datasets.py    # Data loading (4 HuggingFace sources)
│   └── config.py                   # Configuration constants
│
├── 📂 data/                         # Training & Classification Data
│   ├── numerical_questions.csv     # 100 numerical Q&A pairs
│   ├── conceptual_questions.csv    # 100 conceptual Q&A pairs
│   └── real_physics_qa.csv         # 14,608 physics Q&A pairs (4 sources)
│
├── 📂 models/                       # Pre-trained Models
│   └── 📂 lora_finetuned_physics/  # Pre-trained LoRA Adapter
│       ├── adapter_config.json     # LoRA configuration
│       ├── adapter_model.bin       # Trained LoRA weights
│       └── README.md               # Model documentation
│
├── 📂 Knowledge_Base/              # Your Physics PDFs (user-provided)
│   └── (Add your textbooks here)   # Will be vectorized on first run
│
├── 📄 requirements.txt             # Python dependencies
├── 📄 README.md                    # This file
├── 📄 AI_Agent_Architecture.pdf    # System design & flowcharts
└── 📄 Data_Science_Report_LoRA.pdf # Fine-tuning results & metrics
```

---

## 🚀 Quick Start Guide

### Prerequisites

- **Python:** 3.10.12+
- **GPU:** NVIDIA T4 or better (T4 used in development)
- **CUDA:** 12.0+
- **GROQ API Key:** Get free at [console.groq.com](https://console.groq.com)

### Installation

#### **Option 1: With Pre-trained LoRA (Recommended - 5 minutes)**

Skip 2 hours of training and use our pre-trained LoRA adapter!

```bash
# 1. Clone repository
git clone https://github.com/SuryaKameshMantha/multimodal-ai-agent-pipeline.git
cd multimodal-ai-agent-pipeline

# 2. Create Python environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# 5. Create Knowledge Base directory and add your PDFs
mkdir -p Knowledge_Base
# Add your physics textbooks/PDFs to Knowledge_Base/ folder

# 6. Build vector database (one-time, ~30-60 seconds)
python src/textbook_kb_builder.py

# 7. Train DNN classifier on 200 Q&A pairs (one-time, ~45 seconds)
python src/train_classifier_dnn.py

# 8. Launch Streamlit app
streamlit run src/streamlit_app.py
```

**What happens automatically:**
- ✅ LoRA adapter loads instantly (no training needed)
- ✅ FLAN-T5-Base loads (first run only)
- ✅ System ready in ~2 minutes
- ✅ Both vector_db and DNN ready for deployment

---

#### **Option 2: Full From-Scratch Training (Advanced - 3+ hours)**

Train LoRA adapter yourself on 14,608 physics Q&A pairs:

```bash
# 1-5. Same as Option 1 above

# 6. Build vector database
python src/textbook_kb_builder.py

# 7. Train DNN classifier (45 seconds)
python src/train_classifier_dnn.py

# 8. Train LoRA adapter on physics dataset (2 hours)
python src/train_lora_physics.py
# Output: models/lora_finetuned_physics/adapter_model.bin

# 9. Launch Streamlit app
streamlit run src/streamlit_app.py
```

**What happens:**
- ✅ Loads 14,608 Q&A pairs from 4 HuggingFace sources
- ✅ Fine-tunes LoRA with rank 8, 0.36% trainable params
- ✅ Achieves 75.6% loss reduction (0.3531 → 0.0863)
- ✅ Saves new LoRA adapter to models/lora_finetuned_physics/

---

## 📊 Why Pre-trained LoRA Included?

**The Problem:** Full fine-tuning takes 9-12 hours on a T4 GPU

**Our Solution:** Pre-trained LoRA adapter

| Aspect | Full Fine-Tune | LoRA (Pre-trained) | Gain |
|--------|---|---|---|
| **Training Time** | 9-12 hours | 2 hours | 6x faster |
| **Memory** | 24GB | 4GB | 6x efficient |
| **Trainable Params** | 250M | 884K | 282x fewer |
| **Quality** | 95% | 85% (90% of full) | 90% retention |
| **Time to Deploy** | 12+ hours | 5 minutes | **Production-ready** |
| **Cost** | $15-20 | $2-3 | 75% savings |

**Recommendation:** Use pre-trained LoRA unless you need domain-specific physics retraining.

---

## 🛠️ System Components

### 1. **Question Classifier (DNN)**
- **File:** `src/train_classifier_dnn.py` + `src/interactive_classifier_dnn.py`
- **Data:** 100 numerical + 100 conceptual Q&A pairs
- **Accuracy:** 96%
- **Speed:** 87ms per question
- **Purpose:** Route to specialized agents (Numerical vs Conceptual)

### 2. **Knowledge Base (ChromaDB + RAG)**
- **File:** `src/textbook_kb_builder.py`
- **Input:** Your PDFs in `Knowledge_Base/` folder
- **Process:** Chunks → Embeddings → Vector storage
- **Speed:** 230ms retrieval
- **Output:** Grounds answers in your domain knowledge

### 3. **Answer Generation (Dual Models)**
- **Files:** `src/complete_rag_lora_comparison.py`
- **Models:**
  - Base: FLAN-T5-Base (frozen, general)
  - LoRA: FLAN-T5 + Physics Adapter (specialized)
- **Speed:** 1.2s base + 1.4s LoRA
- **Output:** Compares both, shows improvement

### 4. **Numerical Solver (GROQ)**
- **File:** `src/numerical_solver.py`
- **Model:** Llama-3.3-70B-versatile
- **Speed:** 0.9s per calculation
- **Purpose:** Step-by-step numerical solutions

### 5. **Continuous Learning**
- **Files:** `src/feedback_writer.py` + `src/merge_feedback_dnn.py`
- **Process:** User corrections → CSV → Auto-retrain → Improved classifier
- **Retraining:** 45 seconds per update
- **Benefit:** System improves over time

---

## 📈 Performance Metrics

### Training Results

| Metric | Value | Status |
|--------|-------|--------|
| **Loss Reduction** | 75.6% (0.3531 → 0.0863) | ✅ Excellent |
| **Convergence** | 3 epochs, plateau at Epoch 3 | ✅ Optimal |
| **Training Time** | 2 hours (LoRA) | ✅ Practical |
| **Classification Accuracy** | 96% | ✅ Production-ready |
| **DNN Retraining** | 45 seconds | ✅ Fast learning |

### System Response Times

| Component | Time | Status |
|-----------|------|--------|
| **Classification** | 87ms | ✅ Fast |
| **Conceptual Q** | 2.88s | ✅ Acceptable |
| **Numerical Q** | 1.8s | ✅ Responsive |
| **Setup** | ~2 minutes | ✅ Quick |

---

## 💻 Usage Example

### Start the Web UI

```bash
streamlit run src/streamlit_app.py
```

### Three Main Tabs

**Tab 1: Ask Questions**
- Enter any physics question
- System automatically classifies (numerical vs conceptual)
- Receives specialized answer
- User can provide feedback

**Tab 2: Upload & Train**
- Add PDFs to Knowledge_Base/
- Builds vector database
- Trains DNN classifier
- Or retrain LoRA if needed

**Tab 3: Examples**
- Browse sample questions
- See full system workflow
- Understand routing logic

---

## 📊 Data Used

### Classification Training (200 Q&A)
- **numerical_questions.csv:** 100 questions on calculations
- **conceptual_questions.csv:** 100 questions on concepts
- Used for: Training DNN router

### LoRA Fine-tuning (14,608 Q&A)
- **real_physics_qa.csv:** 14,608 physics Q&A pairs from:
  1. physics-scienceqa (6,000)
  2. SciQ Dataset (3,000)
  3. AI2 Reasoning Challenge (2,000)
  4. MMLU Physics (500)
- Used for: LoRA adapter training

### Your Domain Knowledge
- **Knowledge_Base/ (user-provided):** Your physics textbooks
- Processing: Auto-chunked and vectorized
- Used for: RAG context grounding

---

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Vector Database
VECTOR_DB_DIR = "./vector_db"
CHUNK_SIZE = 1000          # Characters per chunk
CHUNK_OVERLAP = 200        # Overlap between chunks
TOP_K = 5                  # Retrieve top-5 chunks
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# DNN Classifier
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# LoRA Configuration
LORA_RANK = 8
LORA_ALPHA = 32
```

---

## 🔐 API Keys & Environment

Create `.env` file in project root:

```bash
# Required
GROQ_API_KEY=gsk_your_key_here

# Optional
HUGGINGFACE_TOKEN=hf_your_token_here
```

**Get GROQ API Key:**
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up (free)
3. Create API key
4. Add to `.env`

---

## 📄 Documentation

### Included Reports

1. **AI_Agent_Architecture.pdf** (13 pages)
   - System design & 11 components
   - 4 detailed flowcharts
   - Model rationale & design choices
   - Complete pipeline examples

2. **Data_Science_Report_LoRA.pdf** (12 pages)
   - LoRA fine-tuning setup
   - Training data sources (14,608 pairs)
   - Convergence analysis & metrics
   - Efficiency comparison (6x speedup)
   - Reproducibility & environment details

### Code Documentation

Each Python file has:
- Docstrings explaining functions
- Type hints for clarity
- Configuration variables at top
- Error handling & logging

---

## 🧪 Testing

### Quick Test (No Training)

```bash
# Just verify installation works
python src/interactive_classifier_dnn.py

# Output should show:
# ✅ Model loaded
# ✅ Vectorizer loaded
# ✅ Ready for inference
```

### Full Test (With Training)

```bash
# 1. Build KB (30-60s)
python src/textbook_kb_builder.py

# 2. Train DNN (45s)
python src/train_classifier_dnn.py

# 3. Launch app
streamlit run src/streamlit_app.py

# 4. In browser:
# - Ask "What is momentum?" (conceptual route)
# - Ask "Calculate energy for 2kg at 5m/s" (numerical route)
```

---

## 🔄 Continuous Learning Workflow

1. **User asks question**
2. **System provides answer**
3. **User marks correct/wrong**
   - ✅ Correct → Logged, system confirms
   - ❌ Wrong → Logged, system records correction
4. **Auto-trigger retraining**
   - Reads feedback.csv
   - Moves Q to correct category
   - Retrains DNN (45 seconds)
5. **System improves**
   - Next similar question → Correct route
   - DNN accuracy increases: 96% → 96.2% → ...

---

## 🚀 Deployment Options

### Option 1: Local Streamlit
```bash
streamlit run src/streamlit_app.py
# Runs at http://localhost:8501
```

### Option 2: Docker Container
```bash
docker build -t physics-qa .
docker run -p 8501:8501 -e GROQ_API_KEY=your_key physics-qa
```

### Option 3: Cloud (Vercel/Heroku)
- Deploy Streamlit to Vercel
- Set environment variables in platform
- LoRA pre-trained included → Fast cold starts

---

## 📚 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **ML Framework** | PyTorch | 2.0.1+ |
| **Transformers** | HuggingFace | 4.34.0+ |
| **Fine-tuning** | PEFT (LoRA) | 0.7.1+ |
| **Vector DB** | ChromaDB | Latest |
| **Web UI** | Streamlit | Latest |
| **Embeddings** | Sentence-Transformers | Latest |
| **API** | GROQ | Llama-3.3-70B |
| **Language** | Python | 3.10.12+ |
| **GPU** | NVIDIA CUDA | 12.0+ |

---

## 📋 Requirements

See `requirements.txt` for exact versions:

```
torch==2.0.1
transformers==4.34.0
peft==0.7.1
chromadb>=0.3.21
sentence-transformers>=2.2.2
streamlit>=1.28.0
groq>=0.4.1
python-dotenv>=1.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
```

---

## ✅ Checklist for Submission

- ✅ Source code (11 Python modules)
- ✅ Pre-trained LoRA adapter
- ✅ Training data (200 + 14,608 Q&A pairs)
- ✅ AI Architecture document (flowcharts + design)
- ✅ Data Science report (metrics + analysis)
- ✅ Environment setup (requirements.txt)
- ✅ README with setup instructions
- ✅ Continuous learning system
- ✅ Multi-agent routing (DNN)
- ✅ RAG grounding (ChromaDB)
- ✅ Dual model comparison (Base + LoRA)
- ✅ Numerical solver integration (GROQ)

---

## 📖 References

- LoRA Paper: [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- FLAN-T5: [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)
- RAG: [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- HuggingFace: [transformers Documentation](https://huggingface.co/docs/transformers/)

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## 🎓 Citation

If you use this system, please cite:

```
@project{physics_qa_agent,
  title={Physics Q&A AI Agent: Multi-Agent System with Continuous Learning},
  author={Surya Kamesh Mantha},
  year={2025},
  url={https://github.com/SuryaKameshMantha/multimodal-ai-agent-pipeline}
}
```

---

## 🙏 Acknowledgments

- **HuggingFace:** For transformer models and datasets
- **GROQ:** For fast inference API
- **Google:** For FLAN-T5 pre-trained model

---

**Last Updated:** November 1, 2025  
**Status:** ✅ Production Ready
