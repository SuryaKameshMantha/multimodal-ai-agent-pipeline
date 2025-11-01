# Physics Q&A AI Agent System

**A Multi-Agent AI System for Physics Question Answering with Continuous Learning**

---

## 👤 Project Information

**Developer:** Surya Kamesh Mantha  
**University:** Indian Institute Of Technology, Roorkee  
**Department:** Chemical Engineering  
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

## 🎯 Repository Structure

```
multimodal-ai-agent-pipeline/
│
├── 📂 src/                                  # Source Code (11 Python Modules)
│   ├── streamlit_app.py                    # Main UI orchestrator
│   ├── train_classifier_dnn.py             # DNN training (routing)
│   ├── interactive_classifier_dnn.py       # Question classification
│   ├── complete_rag_lora_comparison.py     # RAG + answer generation
│   ├── textbook_kb_builder.py              # Knowledge base creation
│   ├── numerical_solver.py                 # GROQ API integration
│   ├── feedback_writer.py                  # User feedback collection
│   ├── merge_feedback_dnn.py               # Auto-retraining system
│   ├── train_lora_physics.py               # LoRA fine-tuning
│   ├── load_physics_datasets.py            # Data loading (4 HuggingFace sources)
│   └── config.py                           # Configuration constants
│
├── 📂 data/                                 # Training & Classification Data
│   ├── numerical_questions.csv             # 100 numerical Q&A pairs
│   ├── conceptual_questions.csv            # 100 conceptual Q&A pairs
│   └── real_physics_qa.csv                 # 14,608 physics Q&A pairs (4 sources)
│
├── 📂 models/                               # Pre-trained Models
│   └── 📂 lora_finetuned_physics/          # Pre-trained LoRA Adapter
│       ├── adapter_config.json             # LoRA configuration
│       ├── adapter_model.bin               # Trained LoRA weights
│       └── README.md                       # Model documentation
│
├── 📂 Knowledge_Base/                       # Your Physics PDFs (user-provided)
│   └── physics_textbook.pdf                # ← ADD YOUR PHYSICS TEXTBOOK HERE
│
├── 📄 requirements.txt                      # Python dependencies
├── 📄 README.md                             # This file
├── 📄 AI_Agent_Architecture.pdf             # System design & flowcharts
└── 📄 Data_Science_Report_LoRA.pdf          # Fine-tuning results & metrics
```

---

## ⚠️ IMPORTANT: Add a Physics Textbook!

Before running the system, you **MUST** add at least one physics PDF file to the `Knowledge_Base/` folder.

### Why?
The RAG system grounds answers in your PDFs, making them specific to your physics course.

**WITHOUT PDF:**
```
Q: "What is momentum?"
A: [Generic textbook definition from general knowledge]
```

**WITH YOUR PDF:**
```
Q: "What is momentum?"
A: [Answer using YOUR textbook definitions and notation]
   - Matches your course style
   - Uses your specific examples
   - Follows your textbook conventions
```

### Where to Get Free Physics PDFs

1. **OpenStax Physics** (Recommended - Free & Comprehensive)
   - https://openstax.org/books/physics
   - Covers: mechanics, waves, thermodynamics, optics, modern physics

2. **MIT OpenCourseWare**
   - https://ocw.mit.edu/courses/physics/
   - Lecture notes, problem sets, solutions

3. **ArXiv Physics Papers**
   - https://arxiv.org/list/physics/
   - Filter by topic (mechanics, quantum, thermodynamics)

4. **Your Own Textbooks**
   - Scan physical textbooks → Convert to PDF
   - Use e-textbooks → Extract as PDF
   - Use course materials → Save as PDF

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

# 5. Consolidate all files into root folder
## Move Python scripts from src/ to root
mv src/*.py .

## Move CSV data files to root
mv data/*.csv .

## Move model files to root
mv models/lora_finetuned_physics/* .

## Remove empty folders
rmdir src
rmdir data
rmdir models/lora_finetuned_physics
rmdir models

# 6. Update import paths in all Python files
python << 'EOF'
import os
import re

# Update all Python files in root
for file in [f for f in os.listdir('.') if f.endswith('.py')]:
    with open(file, 'r') as f:
        content = f.read()
    
    # Remove src. prefix from imports
    content = re.sub(r'from src\.', 'from ', content)
    content = re.sub(r'import src\.', 'import ', content)
    
    with open(file, 'w') as f:
        f.write(content)

print('✅ Updated import paths in all Python files')
EOF

# 7. Update file paths in config.py
# On Mac/Linux:
sed -i '' 's|models/lora_finetuned_physics/||g' config.py
sed -i '' 's|data/||g' config.py

# On Windows (PowerShell):
# (Get-Content config.py) -replace 'models/lora_finetuned_physics/', '' | Set-Content config.py
# (Get-Content config.py) -replace 'data/', '' | Set-Content config.py

# 8. Create Knowledge_Base folder for your PDFs
mkdir -p Knowledge_Base
# Download from OpenStax/MIT and copy here

# 9. Test imports
python -c "import streamlit_app, train_classifier_dnn, textbook_kb_builder; print('✅ All imports successful!')"

# 10. Build vector database (30-60 seconds)
python textbook_kb_builder.py
# Output: Creates vector_db/ folder

# 11. Train DNN classifier (45 seconds)
python train_classifier_dnn.py
# Output: Creates dnn_classifier.pth + vectorizer.pkl

# 12. Launch Streamlit app
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

**What happens automatically:**
- ✅ LoRA adapter loads instantly (pre-trained, no training)
- ✅ FLAN-T5-Base loads (first run only, ~2GB)
- ✅ Your PDFs vectorized and indexed in ChromaDB
- ✅ DNN classifier trained on 200 Q&A pairs
- ✅ System ready in ~2 minutes
- ✅ Both vector_db and DNN models ready for deployment

---

#### **Option 2: Full From-Scratch Training (Advanced - 3+ hours)**

Train LoRA adapter yourself on 14,608 physics Q&A pairs:

```bash
# 1-4. Same as Option 1

# 5. ⭐ ADD YOUR PHYSICS PDF TO Knowledge_Base/
mkdir -p Knowledge_Base
# Copy your PDF: cp ~/Downloads/physics_textbook.pdf Knowledge_Base/

# 6. Build vector database from your PDFs
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
- ✅ Your PDFs indexed in ChromaDB for RAG

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

**Recommendation:** Use pre-trained LoRA unless you need custom physics retraining.

---

## 🛠️ System Components

### 1. **Question Classifier (DNN)**
- **File:** `src/train_classifier_dnn.py` + `src/interactive_classifier_dnn.py`
- **Data:** 100 numerical + 100 conceptual Q&A pairs (in `data/`)
- **Accuracy:** 96%
- **Speed:** 87ms per question
- **Purpose:** Route to specialized agents (Numerical vs Conceptual)

### 2. **Knowledge Base (ChromaDB + RAG)**
- **File:** `src/textbook_kb_builder.py`
- **Input:** Your PDFs in `Knowledge_Base/` folder ← **CRITICAL!**
- **Process:** Chunks (1000 chars) → Embeddings (384-D) → Vector storage
- **Speed:** 230ms retrieval
- **Output:** Grounds answers in your domain knowledge

### 3. **Answer Generation (Dual Models)**
- **File:** `src/complete_rag_lora_comparison.py`
- **Models:**
  - Base: FLAN-T5-Base (frozen, general knowledge)
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
- **Benefit:** System improves over time from user feedback

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
| **Conceptual Question** | 2.88s | ✅ Acceptable |
| **Numerical Question** | 1.8s | ✅ Responsive |
| **Setup** | ~2 minutes | ✅ Quick |

---

## 💻 Usage Example

### Start the Web UI

```bash
streamlit run src/streamlit_app.py
# Opens at http://localhost:8501
```

### Three Main Tabs

**Tab 1: Ask Questions**
- Enter any physics question
- System automatically classifies (numerical vs conceptual)
- Uses YOUR PDF for grounding via RAG
- Receives specialized answer from LoRA
- User can provide feedback for continuous learning

**Tab 2: Upload & Train**
- Add/update PDFs in `Knowledge_Base/` folder
- Rebuilds vector database
- Trains DNN classifier
- Or retrain LoRA if needed

**Tab 3: Examples**
- Browse sample physics questions
- See full system workflow
- Understand routing logic

---

## 📊 Data Used

### Classification Training (200 Q&A)
- **data/numerical_questions.csv:** 100 numerical calculation questions
- **data/conceptual_questions.csv:** 100 conceptual physics questions
- **Used for:** Training DNN router

### LoRA Fine-tuning (14,608 Q&A)
- **data/real_physics_qa.csv:** 14,608 physics Q&A pairs from:
  1. physics-scienceqa (6,000 pairs)
  2. SciQ Dataset (3,000 pairs)
  3. AI2 Reasoning Challenge (2,000 pairs)
  4. MMLU Physics (500 pairs)
- **Used for:** LoRA adapter training

### Your Domain Knowledge ⭐ REQUIRED
- **Knowledge_Base/:** Your physics textbooks (PDFs)
- **Processing:** Auto-chunked and vectorized into embeddings
- **Used for:** RAG context grounding
- **Impact:** Makes answers specific to your physics course!

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
2. Sign up (free tier available)
3. Create API key in dashboard
4. Add to `.env` file

---

## 📄 Documentation

### Included Reports

1. **AI_Agent_Architecture.pdf** (13 pages)
   - Complete system design with 11 components
   - 4 detailed flowcharts with ASCII boxes
   - Model rationale & design choices
   - Complete pipeline examples (conceptual, numerical, correction)

2. **Data_Science_Report_LoRA.pdf** (12 pages)
   - LoRA fine-tuning setup & methodology
   - Training data sources (14,608 Q&A pairs from 4 sources)
   - Convergence analysis & epoch-by-epoch results
   - Efficiency comparison (6x speedup with 282x fewer params)
   - Full reproducibility details & environment specs

---

## 🧪 Testing

### Quick Test (No Training)

```bash
# Verify installation works
python src/interactive_classifier_dnn.py

# Output should show:
# ✅ Model loaded
# ✅ Vectorizer loaded
# ✅ Ready for inference
```

### Full Test (With Your PDFs)

```bash
# 1. Build KB from your PDFs (30-60s)
python src/textbook_kb_builder.py

# 2. Train DNN classifier (45s)
python src/train_classifier_dnn.py

# 3. Launch Streamlit app
streamlit run src/streamlit_app.py

# 4. In browser, test:
# - Ask "What is momentum?" (conceptual route - uses YOUR PDF)
# - Ask "Calculate energy for 2kg at 5m/s" (numerical route)
# - Provide feedback on answers
```

---

## 🔄 Continuous Learning Workflow

1. **User asks question**
2. **System provides answer** (grounded in your PDF via RAG)
3. **User marks correct/wrong**
   - ✅ Correct → Logged, system confirms
   - ❌ Wrong → Logged, system records for correction
4. **Auto-trigger retraining**
   - Reads feedback.csv
   - Moves Q to correct category
   - Retrains DNN (45 seconds)
5. **System improves**
   - Next similar question → Correct route
   - DNN accuracy increases: 96% → 96.2% → ...

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

## ✅ Submission Checklist

- ✅ Source code (11 Python modules in `src/`)
- ✅ Pre-trained LoRA adapter in `models/`
- ✅ Training data (200 + 14,608 Q&A pairs in `data/`)
- ✅ AI Architecture document (flowcharts + design)
- ✅ Data Science report (metrics + analysis)
- ✅ Environment setup (requirements.txt)
- ✅ README with setup instructions
- ✅ **Knowledge Base folder ready for your PDFs** ← USER ADDS PDF
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
- **OpenStax:** For free physics textbooks

---

**Last Updated:** November 1, 2025  
**Status:** ✅ Production Ready  
**⭐ REMEMBER: Add your Physics PDF to Knowledge_Base/ before running!**
