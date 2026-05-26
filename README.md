# Fraud Detection ML System
> Ironhack Data Science & ML Bootcamp — Individual Project
> Irish-Lev · Barcelona 2026

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link-here.streamlit.app)

---

## Overview
An end-to-end fraud detection system built on 1.3M simulated payment transactions.
Combines classical ML with a HuggingFace transformer for feature enrichment,
a 1D CNN deep learning comparison model, a RAG-powered fraud analyst assistant
for plain-English explanations, and a live 4-tab Streamlit demo.

Built by someone who worked in fraud operations at Stripe — domain knowledge
informed every feature engineering, modelling, and explainability decision.

---

## The Business Problem

Payment card fraud costs the global financial industry billions of dollars every year.
For payment platforms like Stripe, Revolut, and Adyen, fraud is not an abstract risk —
it is an operational reality that affects merchants, cardholders, and compliance teams daily.

The core challenge: out of every 1,000 transactions processed, fewer than 6 are fraudulent.
A naive system that flags nothing catches no fraud. A system that flags everything makes
the product unusable. The business needs a model that is precise enough to trust and
explainable enough to act on.

**Three specific problems this project addresses:**

**1. Detection accuracy on imbalanced data**
Traditional accuracy metrics are misleading when fraud is 0.58% of transactions.
A model predicting "legitimate" for everything achieves 99.4% accuracy but catches
zero fraud. This project uses Precision-Recall AUC as its primary metric and applies
SMOTE to handle class imbalance — the same approach used by Klarna and Stripe in production.

**2. Explainability for ops teams and regulators**
A fraud score without an explanation is not actionable. Under European PSD2 regulation,
financial institutions must justify automated decisions. This project uses SHAP to explain
every model decision at the feature level, and a RAG-powered analyst assistant to generate
plain-English narratives for ops agents.

**3. The gap between a model and a usable tool**
Most fraud detection projects stop at a trained model in a notebook. Real fraud ops teams
need something interactive — input a transaction, get a score, understand the reasoning,
make a decision in under 4 minutes. The Streamlit app closes that gap.

**Who this is built for:**
The primary user is a fraud analyst working a case queue. The secondary user is a hiring
manager at a fintech company evaluating whether a candidate understands the domain,
not just the algorithm.

---

## Tech Stack
| Layer | Tool |
|-------|------|
| Dataset | Sparkov (Kaggle) — 1.3M transactions |
| ML Model | XGBoost + SMOTE |
| Deep Learning | 1D CNN (Keras / TensorFlow) |
| Transformer | ProsusAI/finbert (HuggingFace) |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| SQL | SQLite (fraud_predictions.db) |
| RAG framework | LangChain |
| Vector store | ChromaDB |
| RAG embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | GPT-4o-mini (OpenAI) / Mistral 7B via Ollama (local) |
| App | Streamlit → Streamlit Cloud |

---

## Project Structure
```
fraud-detection-project/
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Feature engineering
│   ├── 03_finbert_embeddings.ipynb   # FinBERT transformer (run in Colab)
│   ├── 04_model_shap.ipynb           # XGBoost + SHAP + MLflow + SQLite
│   ├── 04b_cnn_model.ipynb           # 1D CNN deep learning comparison
│   └── 06_rag_pipeline.ipynb         # RAG knowledge base + LangChain
├── app/
│   ├── app.py                        # Streamlit live demo (4 tabs)
│   └── rag_assistant.py              # RAG module (v2 — score-filtered retrieval)
├── knowledge_base/                   # 4 .txt files powering the RAG system
│   ├── eda_findings.txt
│   ├── fraud_patterns.txt
│   ├── feature_explanations.txt
│   └── historical_cases.txt
├── prompts/                          # Claude Code prompt files
└── src/                              # Reusable Python modules
```

---

## Model Comparison
All three models evaluated on the same held-out test set. PR-AUC is the primary metric — ROC-AUC is optimistic under severe class imbalance (0.58% fraud rate). F1 scores use the default 0.5 threshold; PR-AUC reflects performance across all thresholds.

| Model | PR-AUC | ROC-AUC | F1 (fraud class) | Explainable |
|-------|--------|---------|------------------|-------------|
| Logistic Regression (baseline) | 0.2273 | 0.8997 | 0.1266 | Yes (coefficients) |
| XGBoost + FinBERT (primary) | **0.8263** | **0.9941** | 0.0733 | Yes (SHAP) |
| 1D CNN (deep learning) | 0.7150 | 0.9840 | 0.3818 | Limited |

> XGBoost's low F1 at 0.5 threshold reflects aggressive recall (99.5%) over precision — expected behaviour on 0.58% imbalance. PR-AUC of 0.83 confirms it is the strongest model across all operating thresholds.

---

## Streamlit App — 4 Tabs
| Tab | What it shows |
|-----|---------------|
| Fraud Checker | Input a transaction → fraud score + SHAP waterfall |
| RAG Analyst Assistant | Plain-English explanation + free-text Q&A |
| Model Performance | PR-AUC, confusion matrix, 3-model comparison table |
| Project Roadmap | What was built + planned future extensions |

---

## How to Run
```bash
# 1. Clone and setup
git clone https://github.com/Irish-Lev/fraud-detection-project
cd fraud-detection-project
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Download dataset (requires Kaggle API key)
kaggle datasets download -d kartik2112/fraud-detection -p data/raw --unzip

# 4. Run notebooks in order
jupyter notebook notebooks/
# Order: 01 → 02 → 03 (Colab) → 04 → 04b → 06

# 5. Run Streamlit app
streamlit run app/app.py
```

---

## How RAG Works in This Project
The RAG Fraud Analyst Assistant uses score-filtered retrieval (ChromaDB +
sentence-transformers/all-MiniLM-L6-v2) with a relevance threshold of 0.3.

- For conceptual questions: retrieves fraud pattern chunks, answers as a domain expert
- For transaction-specific questions: retrieves case examples, explains the specific flag
- LLM: GPT-4o-mini called directly with a structured prompt — bypasses Streamlit cache

Knowledge base covers: EDA findings, fraud patterns (CNP, card-testing, ATO,
geographic anomaly), SHAP feature explanations, and historical case examples.

---

## Tableau Dashboard
Interactive EDA dashboard built in Tableau Public:
[View Dashboard](https://public.tableau.com/app/profile/irish.levi.bawingan/viz/Fraud_Detection_17797233902490/Fraud_Dashboard)

Key visualisations:
- Fraud rate by merchant category
- Fraud rate by hour of day
- Class imbalance overview (0.58% fraud rate)

---

## Planned Extensions (Roadmap)
- **v1.1:** Adaptive threshold tuning, Isolation Forest anomaly detection
- **v2.0:** Drift detection (Evidently AI), FastAPI + Docker deployment
- **v3.0:** Graph Neural Network (PyTorch Geometric), Kafka real-time streaming,
  Federated Learning (Flower), Stripe Payments Foundation Model architecture

---

## Domain Context
Designed around real fraud ops workflows used at Stripe, Revolut, and Adyen —
where fraud analysts use Jira queues and Salesforce CRM to triage model-flagged
transactions. The SHAP + RAG explainability layer mirrors the analyst assistant
tooling that makes fraud ops teams faster and more consistent.

This project replicates Stripe's pre-2024 production architecture: XGBoost with
transformer-based feature enrichment. Stripe's May 2025 Payments Foundation Model
is the next evolution — the v3.0 roadmap item.
