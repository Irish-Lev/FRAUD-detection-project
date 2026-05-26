"""
Fraud Detection System — Live Demo
Streamlit app: 4 tabs — Fraud Checker, Model Performance, Project Roadmap, RAG Assistant
Run: streamlit run app/app.py
"""
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

# RAG module — optional; gracefully disabled if packages not installed or ChromaDB not built
try:
    from rag_assistant import load_rag_chain, explain_transaction, ask_knowledge_base
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Page config (must be the very first Streamlit call)
st.set_page_config(
    page_title="Fraud Detection System — Live Demo",
    page_icon="🛡️",
    layout="wide"
)

# Path constants 
APP_DIR    = Path(__file__).parent
ROOT       = APP_DIR.parent
MODELS_DIR = ROOT / 'models'
DATA_PROC  = ROOT / 'data' / 'processed'
DATA_EMB   = ROOT / 'data' / 'embeddings'

# Sparkov merchant categories 
CATEGORIES = sorted([
    'entertainment', 'food_dining', 'gas_transport', 'grocery_net',
    'grocery_pos', 'health_fitness', 'home', 'kids_pets',
    'misc_net', 'misc_pos', 'personal_care', 'shopping_net',
    'shopping_pos', 'travel'
])
# Alphabetical label encoding — matches notebook 02 LabelEncoder order
CATEGORY_ENCODING = {cat: i for i, cat in enumerate(CATEGORIES)}

# Human-readable labels for SHAP feature names
FEATURE_LABELS = {
    'amt':             'Transaction amount ($)',
    'log_amt':         'Log-scaled amount',
    'geo_distance_km': 'Distance from home (km)',
    'hour_of_day':     'Hour of day',
    'is_night':        'Late-night transaction',
    'is_weekend':      'Weekend transaction',
    'velocity_24h':    'Transactions in last 24h',
    'category_encoded':'Merchant category',
    'age':             'Cardholder age',
    'city_pop':        'City population',
    'day_of_week':     'Day of week',
    'gender':          'Gender (0=F, 1=M)',
    'lat':             'Cardholder latitude',
    'long':            'Cardholder longitude',
    'merch_lat':       'Merchant latitude',
    'merch_long':      'Merchant longitude',
}

# Cached resource loaders 
@st.cache_resource
def load_model():
    p = MODELS_DIR / 'xgb_fraud_model.pkl'
    if not p.exists():
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

@st.cache_resource
def get_rag_chain(llm_option: str = "openai"):
    """Load ChromaDB + LLM once. Returns None if ChromaDB not built yet."""
    if not RAG_AVAILABLE:
        return None
    return load_rag_chain(llm_option=llm_option)

@st.cache_resource
def load_embeddings():
    texts_path = DATA_EMB / 'unique_texts.npy'
    embs_path  = DATA_EMB / 'pca_embeddings_unique.npy'
    if not texts_path.exists() or not embs_path.exists():
        return None, None
    texts = np.load(texts_path, allow_pickle=True)
    embs  = np.load(embs_path)
    return texts, embs

@st.cache_data
def build_category_lookup(_texts, _embs):
    """Average the 30-dim PCA embeddings per category for fast live lookup."""
    if _texts is None:
        return {cat: np.zeros(30) for cat in CATEGORIES}
    lookup = {}
    for cat in CATEGORIES:
        mask = np.array([f'category: {cat}' in str(t) for t in _texts])
        lookup[cat] = _embs[mask].mean(axis=0) if mask.sum() > 0 else np.zeros(30)
    return lookup

@st.cache_data
def get_model_features(_model):
    """Read exact feature names from the trained XGBoost model — source of truth."""
    if _model is None:
        return []
    names = _model.get_booster().feature_names
    return names if names else []

# Feature vector builder 
def build_feature_vector(amt, category, hour, geo_dist, velocity,
                         is_weekend, category_lookup, feature_names):
    """
    Construct the exact feature vector the model expects.
    User-provided values override defaults; embed_* columns come from the
    pre-computed category lookup (30-dim PCA FinBERT embeddings).
    """
    # Start with zeros for every feature the model knows about
    row = {col: 0.0 for col in feature_names}

    # User-controlled features
    row['amt']             = float(amt)
    row['log_amt']         = float(np.log1p(amt))
    row['hour_of_day']     = float(hour)
    row['is_night']        = float(1 if (hour < 6 or hour >= 22) else 0)
    row['is_weekend']      = float(int(is_weekend))
    row['day_of_week']     = float(5 if is_weekend else 2)   # Sat=5 or Wed=2
    row['geo_distance_km'] = float(geo_dist)
    row['velocity_24h']    = float(velocity)
    row['category_encoded']= float(CATEGORY_ENCODING.get(category, 7))

    # Sensible defaults for features not exposed in the UI
    row['gender']    = 0.0        # Female (most common in Sparkov)
    row['age']       = 35.0
    row['city_pop']  = 50000.0
    # Fix all coordinates at US geographic center — do NOT compute merch_lat
    # from geo_dist because it pushes coordinates outside the US training
    # distribution and causes extreme out-of-distribution XGBoost behaviour.
    # geo_distance_km already carries the full geographic signal.
    row['lat']       = 38.5
    row['long']      = -96.0
    row['merch_lat'] = 38.5
    row['merch_long']= -96.0

    # FinBERT category embeddings (30 PCA components)
    emb = category_lookup.get(category, np.zeros(30))
    for i, val in enumerate(emb):
        key = f'embed_{i}'
        if key in row:
            row[key] = float(val)

    return pd.DataFrame([row], columns=feature_names)

# Plotly gauge 

def fraud_gauge(prob):
    pct   = prob * 100
    color = '#27ae60' if pct < 30 else ('#f39c12' if pct < 70 else '#e74c3c')
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=pct,
        number={'suffix': '%', 'font': {'size': 40, 'color': color}},
        title={'text': 'Fraud Risk Score', 'font': {'size': 15}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar':  {'color': color, 'thickness': 0.28},
            'bgcolor': 'white',
            'steps': [
                {'range': [0,  30], 'color': '#d5f5e3'},
                {'range': [30, 70], 'color': '#fef9e7'},
                {'range': [70, 100],'color': '#fadbd8'},
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 4},
                'thickness': 0.8,
                'value': pct
            }
        }
    ))
    fig.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20))
    return fig

# Plain-English SHAP explanation 

def generate_explanation(shap_vals, feature_names, input_df, fraud_prob):
    contributions = pd.Series(shap_vals, index=feature_names)
    top3 = contributions.nlargest(3)
    top_feat = top3.index[0]

    # Sentence 1 — overall verdict
    if fraud_prob >= 0.7:
        s1 = (f"This transaction carries a **{fraud_prob*100:.0f}% fraud risk** — "
              f"well above the threshold and should be routed to analyst review.")
    elif fraud_prob >= 0.3:
        s1 = (f"This transaction shows **elevated risk at {fraud_prob*100:.0f}%** — "
              f"a secondary review is advised before clearing.")
    else:
        s1 = (f"This transaction appears **low risk at {fraud_prob*100:.0f}%** "
              f"and is likely legitimate based on the available signals.")

    # Sentence 2 — top driver in plain English
    driver_map = {
        'amt':             f"the transaction amount of ${input_df['amt'].iloc[0]:,.0f} is unusually high for this merchant category",
        'log_amt':         f"the transaction amount of ${input_df['amt'].iloc[0]:,.0f} is unusually high for this merchant category",
        'geo_distance_km': f"the merchant is {input_df['geo_distance_km'].iloc[0]:.0f} km from the cardholder's home — a geographic velocity flag",
        'is_night':        "the transaction occurred late at night (midnight–6am) when fraud rates spike significantly",
        'hour_of_day':     f"the transaction time (hour {int(input_df['hour_of_day'].iloc[0])}:00) falls in a high-risk window",
        'velocity_24h':    f"the cardholder made {int(input_df['velocity_24h'].iloc[0])} transactions in the last 24 hours — above normal velocity",
        'category_encoded':"the merchant category has an above-average historical fraud rate in the training data",
        'is_weekend':      "weekend transactions carry elevated fraud risk for this category",
    }
    driver_desc = driver_map.get(top_feat, f"the model identified '{FEATURE_LABELS.get(top_feat, top_feat)}' as the primary risk driver")
    s2 = f"The primary risk driver is {driver_desc}."

    # Sentence 3 — second driver (if elevated risk)
    if fraud_prob >= 0.3 and len(top3) > 1:
        second_feat = top3.index[1]
        second_desc = driver_map.get(second_feat, f"elevated '{FEATURE_LABELS.get(second_feat, second_feat)}'")
        s3 = f"Secondary signal: {second_desc}."
        return f"{s1}\n\n{s2} {s3}"

    return f"{s1}\n\n{s2}"

# ══════════════════════════════════════════════════════════════════════════════
# Load all resources at startup
# ══════════════════════════════════════════════════════════════════════════════
model         = load_model()
feature_names = get_model_features(model)
texts, embs   = load_embeddings()
cat_lookup    = build_category_lookup(texts, embs)
explainer     = load_explainer(model) if model is not None else None

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Fraud Checker",
    "📊 Model Performance",
    "🗺️ Project Roadmap",
    "🤖 RAG Analyst Assistant",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Fraud Checker
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.title("🛡️ Fraud Detection System — Live Demo")
    st.caption("XGBoost · 46 features (16 engineered + 30 FinBERT embeddings) · SHAP explainability · Trained on 1M+ Sparkov transactions")

    if model is None:
        st.error(
            "⚠️ Model not found at `models/xgb_fraud_model.pkl`. "
            "Run notebook `04_model_shap.ipynb` first to train and save the model."
        )
        st.stop()

    # Sidebar 
    with st.sidebar:
        st.header("Transaction Details")
        st.caption("Enter the transaction information to check fraud risk")

        amt      = st.number_input("Transaction Amount ($)", min_value=0.0,
                                   max_value=10000.0, value=250.0, step=10.0)
        merchant = st.text_input("Merchant Name",
                                 value="Hartmann's Fresh Market",
                                 help="Free text — used for display only")
        category = st.selectbox("Merchant Category", CATEGORIES,
                                index=CATEGORIES.index('shopping_net'))
        hour     = st.slider("Hour of Day", 0, 23, value=14,
                             help="0 = midnight · 14 = 2pm · 23 = 11pm")
        geo_dist = st.number_input("Distance from Home (km)", min_value=0.0,
                                   max_value=2000.0, value=50.0, step=10.0)
        velocity = st.number_input("Transactions in Last 24h",
                                   min_value=1, max_value=50, value=1)
        is_wknd  = st.checkbox("Weekend transaction", value=False)

        st.divider()
        check_btn = st.button("🔍 Check Transaction", type="primary",
                              use_container_width=True)

        # Demo scenarios guide
        with st.expander("Try these demo scenarios"):
            st.markdown("""
**LOW RISK** (normal daytime shopping)
`$250 · shopping_net · 2pm · 50km · vel=1`

**MEDIUM RISK** (late night, unfamiliar merchant)
`$45 · gas_transport · 11pm · 90km · vel=4`

**HIGH RISK** (card-testing pattern: small amount, night, nearby)
`$1000 · misc_net · 2am · 5km · vel=3`

> In the Sparkov dataset, fraud manifests as **card testing** — small-to-mid amounts at nearby merchants late at night to verify stolen card details, not the large single transaction you might expect from real-world whale fraud.
""")
        st.divider()

    # Main panel — idle state 
    if not check_btn:
        st.info("👈 Fill in the transaction details in the sidebar, then click **Check Transaction**.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", "XGBoost")
        with col2:
            st.metric("Training rows", "1,037,340")
        with col3:
            st.metric("Primary metric", "PR-AUC")

    # Main panel — after button click 
    else:
        input_df   = build_feature_vector(amt, category, hour, geo_dist,
                                          velocity, is_wknd, cat_lookup, feature_names)
        fraud_prob = float(model.predict_proba(input_df)[0, 1])

        # Save basics for Tab 4 RAG explanation
        st.session_state['last_transaction'] = {
            'fraud_prob': fraud_prob,
            'category': category,
            'amount': amt,
            'hour': hour,
            'geo_distance_km': geo_dist,
        }

        # Row 1: gauge + verdict
        col_gauge, col_verdict = st.columns([1, 1.8])

        with col_gauge:
            st.plotly_chart(fraud_gauge(fraud_prob), use_container_width=True)

        with col_verdict:
            st.markdown("### Risk Verdict")
            if fraud_prob >= 0.7:
                st.error("🚨  **HIGH RISK** — Route to analyst review immediately")
            elif fraud_prob >= 0.3:
                st.warning("⚠️  **MEDIUM RISK** — Secondary review advised")
            else:
                st.success("✅  **LOW RISK** — Transaction appears legitimate")

            st.markdown("**Transaction Summary**")
            summary = {
                "Merchant":           merchant,
                "Category":           category,
                "Amount":             f"${amt:,.2f}",
                "Hour":               f"{hour:02d}:00",
                "Distance from home": f"{geo_dist:.0f} km",
                "Velocity (24h)":     str(int(velocity)),
                "Weekend":            "Yes" if is_wknd else "No",
            }
            for k, v in summary.items():
                st.markdown(f"**{k}:** {v}")

        st.divider()

        # Row 2: SHAP waterfall + plain-English explanation
        col_shap, col_text = st.columns([1.5, 1])

        with col_shap:
            st.markdown("### Why did the model flag this transaction?")
            st.caption("Red bars push toward fraud · Blue bars push toward legitimate · Width = magnitude")
            try:
                shap_vals = explainer.shap_values(input_df)
                # Save top SHAP features for Tab 4 RAG explanation
                contribs = pd.Series(shap_vals[0], index=feature_names)
                st.session_state['last_shap_vals'] = dict(
                    contribs.reindex(contribs.abs().nlargest(5).index)
                )
                explanation = shap.Explanation(
                    values        = shap_vals[0],
                    base_values   = float(explainer.expected_value),
                    data          = input_df.iloc[0].values,
                    feature_names = [FEATURE_LABELS.get(f, f) for f in feature_names]
                )
                plt.figure(figsize=(9, 6))
                shap.plots.waterfall(explanation, show=False, max_display=12)
                st.pyplot(plt.gcf(), bbox_inches='tight')
                plt.close()
            except Exception as e:
                st.warning(f"SHAP plot unavailable: {e}")
                shap_vals = None

        with col_text:
            st.markdown("### Plain-English Explanation")
            if shap_vals is not None:
                st.markdown(generate_explanation(
                    shap_vals[0], feature_names, input_df, fraud_prob
                ))
            st.markdown("---")
            st.markdown(
                "**About this model:** XGBoost classifier trained on 1M+ Sparkov "
                "simulated transactions. 46 features: 16 engineered signals "
                "(amount, time, geography, velocity, category) + 30 FinBERT PCA embeddings.\n\n"
                "**Sparkov fraud pattern:** The dataset simulates fraud as "
                "**card testing** — fraudsters verify stolen card details with "
                "small-to-mid amounts at nearby merchants late at night, not the "
                "large single transactions seen in real-world whale fraud. "
                "The model learned these patterns from data."
            )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Performance
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.title("📊 Model Performance")
    st.caption("All models evaluated on the same held-out test set (time-based split, ~259K transactions)")

    # Model comparison table
    st.subheader("Three-Model Comparison")
    st.caption("★ PR-AUC is the primary metric — precision-recall AUC is more reliable than ROC-AUC under class imbalance")

    comp_path = DATA_PROC / 'model_comparison.csv'
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
    else:
        st.info("model_comparison.csv not yet generated. Run notebooks 04 and 04b first.")

    st.markdown("""
**Why XGBoost wins on tabular fraud data:**
- Tree-based splits handle non-linear feature interactions naturally — no scaling, no architecture tuning
- Full SHAP explainability makes every prediction auditable for compliance
- Faster inference than neural networks — critical for sub-100ms production scoring
- 46 features (16 engineered + 30 FinBERT PCA embeddings) — tabular signals enriched by transformer-based merchant context
""")

    st.divider()

    # SHAP global importance
    st.subheader("Global Feature Importance (SHAP Beeswarm)")
    st.caption("Top features ranked by mean |SHAP value| — each dot is one of 2,000 test transactions")
    shap_img = DATA_PROC / 'plot_shap_summary.png'
    if shap_img.exists():
        st.image(str(shap_img), use_container_width=True)
    else:
        st.info("SHAP summary plot not found. Run notebook 04_model_shap.ipynb first.")

    st.divider()

    # PR curve + confusion matrix
    st.subheader("Precision-Recall Curve & Confusion Matrix")
    eval_img = DATA_PROC / 'plot_model_evaluation.png'
    if eval_img.exists():
        st.image(str(eval_img), use_container_width=True)
    else:
        st.info("Evaluation plot not found. Run notebook 04_model_shap.ipynb first.")

    # CNN training history (if available)
    cnn_hist = DATA_PROC / 'plot_cnn_training_history.png'
    if cnn_hist.exists():
        st.divider()
        st.subheader("1D CNN Training History")
        st.caption("Training vs validation loss and AUC across epochs — EarlyStopping restored best weights")
        st.image(str(cnn_hist), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Project Roadmap
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.title("🗺️ Project Roadmap")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### ✅ Built for this presentation")
        st.markdown("""
- **EDA** — 11 charts, fraud pattern analysis, Tableau dashboard
- **Feature Engineering** — 16 engineered features: geo-distance, velocity, time signals, log-amount
- **FinBERT embeddings** — 768-dim → 30-dim PCA on Colab T4 GPU (HuggingFace)
- **XGBoost + SHAP** — primary model with full explainability + MLflow experiment tracking
- **1D CNN** — deep learning comparison model (Keras, Colab)
- **SQLite audit database** — all predictions logged for compliance
- **This Streamlit app** — 4-tab live fraud demo
""")

    with col2:
        st.markdown("### 🔧 Completing now")
        st.markdown("""
- **RAG Fraud Analyst Assistant** ✅ — LangChain + ChromaDB + GPT-4o-mini
  Ask plain-English questions about fraud patterns and get sourced answers from the knowledge base

- **GitHub repo** ✅ — clean phased commits with README and documented notebooks

- **Streamlit Cloud deployment** ✅ — public URL for the live demo
""")

    with col3:
        st.markdown("### 🚀 v2.0 Roadmap")
        st.markdown("""
- **Drift detection** — Evidently AI for monitoring model degradation over time
- **Anomaly detection** — Isolation Forest + Autoencoder for unsupervised signals
- **Graph Neural Network** — model cardholder-merchant relationships as a graph
- **Kafka streaming** — real-time transaction scoring pipeline
- **Federated learning** — train across institutions without sharing raw data
- **FastAPI + Docker** — production API container for model serving
""")

    st.divider()
    st.subheader("Bootcamp Requirements Coverage")

    reqs = pd.DataFrame({
        "Requirement": [
            "Dataset + goals", "Data preparation", "EDA with insights",
            "Tableau dashboard", "ML model", "Deep Learning",
            "Evaluation metrics", "Gen AI component",
            "GitHub repo", "Optional SQL", "Trello board", "App to showcase"
        ],
        "How it's met": [
            "Sparkov 1.3M transaction dataset (Kaggle)",
            "02_feature_engineering.ipynb — 16 engineered features",
            "01_eda.ipynb — 11 cells, 6 plots",
            "3 charts published to Tableau Public",
            "XGBoost with SMOTE + SHAP + MLflow",
            "1D CNN (Keras) comparison model",
            "PR-AUC, ROC-AUC, F1, confusion matrix — all 3 models",
            "FinBERT (HuggingFace) + RAG (LangChain + ChromaDB)",
            "README + documented notebooks + requirements.txt",
            "SQLite predictions database — notebook 04 Cell 11",
            "Set up at trello.com",
            "This Streamlit app (4 tabs)"
        ],
        "Status": [
            "✅", "✅", "✅", "✅", "✅", "✅",
            "✅", "✅", "✅", "✅", "✅", "✅"
        ]
    })
    st.dataframe(reqs, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — RAG Analyst Assistant
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.title("🤖 RAG Fraud Analyst Assistant")
    st.caption("LangChain · ChromaDB · sentence-transformers/all-MiniLM-L6-v2 · Retrieval-Augmented Generation")

    if not RAG_AVAILABLE:
        st.error(
            "RAG packages not installed. Run: "
            "`pip install langchain langchain-community langchain-huggingface chromadb openai python-dotenv`"
        )
        st.stop()

    # LLM selector
    col_llm, _ = st.columns([1, 2])
    with col_llm:
        llm_choice = st.radio(
            "LLM backend",
            options=["openai", "claude", "ollama"],
            format_func=lambda x: {
                "openai": "OpenAI GPT-4o-mini (default)",
                "claude": "Claude Haiku (API)",
                "ollama": "Ollama / Mistral (local)",
            }[x],
            horizontal=True,
            help="OpenAI key already in .env · Claude needs ANTHROPIC_API_KEY · Ollama needs `ollama serve`",
        )
    if llm_choice == "ollama":
        st.warning("⚠️ Ollama / Mistral requires a local `ollama serve` instance running on your machine — it will not work on Streamlit Cloud. Switch to **OpenAI GPT-4o-mini** for the live demo.")
    if llm_choice == "claude":
        st.warning("⚠️ Claude Haiku requires an ANTHROPIC_API_KEY — no API key has been configured yet. Switch to **OpenAI GPT-4o-mini** for the live demo.")

    # Load RAG chain (cached — only runs once per llm_choice)
    chroma_path = ROOT / 'data' / 'chroma_db'
    if not chroma_path.exists():
        st.warning(
            "ChromaDB not found. Run `notebooks/06_rag_pipeline.ipynb` (Cells 1–5) first "
            "to build the vector store from the knowledge base."
        )
        st.markdown("""
**Setup steps:**
1. Open `notebooks/06_rag_pipeline.ipynb` in VS Code
2. Run all cells top-to-bottom (Cells 1–5 build and persist ChromaDB)
3. Come back to this tab — it will load automatically
        """)
    else:
        with st.spinner("Loading RAG chain..."):
            chain = get_rag_chain(llm_option=llm_choice)

        if chain is None:
            st.error("RAG chain failed to load. Check that your LLM backend is running.")
        else:
            llm_label = {"openai": "OpenAI GPT-4o-mini", "claude": "Claude Haiku", "ollama": "Ollama/Mistral"}[llm_choice]
            st.success(f"Knowledge base loaded · LLM: {llm_label}")

            st.divider()

            # ── Section 1: Explain last scored transaction ────────────────────
            st.subheader("Explain last scored transaction")

            last_tx   = st.session_state.get('last_transaction')
            last_shap = st.session_state.get('last_shap_vals')

            if last_tx is None:
                st.info("Go to the **Fraud Checker** tab, score a transaction, then come back here.")
            else:
                pct = last_tx['fraud_prob'] * 100
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Fraud score",   f"{pct:.1f}%")
                col_b.metric("Category",      last_tx['category'])
                col_c.metric("Amount",        f"${last_tx['amount']:,.2f}")
                col_d.metric("Distance",      f"{last_tx['geo_distance_km']:.0f} km")

                if st.button("Generate RAG Explanation", type="primary"):
                    with st.spinner("Retrieving from knowledge base..."):
                        try:
                            top_feats = last_shap if last_shap else {
                                'geo_distance_km': 0.0, 'hour_of_day': 0.0
                            }
                            explanation, source_docs = explain_transaction(
                                chain,
                                fraud_score  = last_tx['fraud_prob'],
                                top_features = top_feats,
                                transaction  = {
                                    'category':       last_tx['category'],
                                    'amount':         last_tx['amount'],
                                    'hour':           last_tx['hour'],
                                    'geo_distance_km':last_tx['geo_distance_km'],
                                },
                            )
                            st.markdown("**Analyst Explanation:**")
                            st.info(explanation)
                            with st.expander("Knowledge base sources retrieved"):
                                for doc in source_docs:
                                    src = Path(doc.metadata.get('source', 'unknown')).name
                                    st.caption(f"**{src}**")
                                    st.text(doc.page_content[:250] + "...")
                        except Exception as e:
                            st.error(f"RAG error: {e}")
                            if llm_choice == "ollama":
                                st.caption("Is `ollama serve` running? Try: `ollama serve` in a terminal.")

            st.divider()

            # ── Section 2: Free-text Q&A ──────────────────────────────────────
            st.subheader("Ask the fraud knowledge base")
            st.caption("Ask anything about fraud patterns, model features, or EDA findings")

            example_qs = [
                "Which merchant categories have the highest fraud rate and why?",
                "What is card-testing fraud and how does the model detect it?",
                "Why is geographic distance such a strong fraud signal?",
                "What does velocity_24h measure and when is it suspicious?",
                "How does time of day affect fraud risk?",
            ]
            with st.expander("Example questions"):
                for q in example_qs:
                    st.markdown(f"- *{q}*")

            user_question = st.text_input(
                "Your question",
                placeholder="e.g. Why is geo_distance_km a strong fraud signal?",
                label_visibility="collapsed",
            )

            if st.button("Ask", type="primary") and user_question.strip():
                with st.spinner("Retrieving from knowledge base and generating answer..."):
                    try:
                        # ── Use score-filtered retrieval directly ──────────────
                        # Bypasses the cached chain prompt entirely.
                        # Calls ChromaDB with relevance scores, then calls the
                        # LLM with a fresh structured prompt every time.
                        from rag_assistant import _get_context_with_scores, QA_PROMPT_TEMPLATE

                        vectorstore = chain.get("vectorstore") if isinstance(chain, dict) else None

                        if vectorstore is not None:
                            # Score-filtered retrieval — teacher's approach
                            context, kb_source = _get_context_with_scores(
                                vectorstore,
                                user_question.strip(),
                                score_threshold=0.3,
                                k=5,
                            )
                        else:
                            context = (
                                "The fraud knowledge base does not contain a direct answer. "
                                "Use your broader fraud detection expertise to answer."
                            )
                            kb_source = "domain_expertise"

                        # Build prompt fresh — not from cached chain
                        full_prompt = QA_PROMPT_TEMPLATE.format(
                            context=context,
                            question=user_question.strip(),
                        )

                        # Call LLM directly
                        import openai as _openai
                        import os
                        _api_key = os.getenv("OPENAI_API_KEY")
                        if not _api_key:
                            env_path = ROOT / ".env"
                            if env_path.exists():
                                for line in env_path.read_text(encoding="utf-8").splitlines():
                                    if line.startswith("OPENAI_API_KEY="):
                                        _api_key = line.split("=", 1)[1].strip()

                        _client = _openai.OpenAI(api_key=_api_key)
                        _response = _client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": full_prompt}],
                            temperature=0.3,
                            max_tokens=600,
                            top_p=0.9,
                            frequency_penalty=0.3,
                            presence_penalty=0.2,
                        )
                        answer = _response.choices[0].message.content
                        source_docs = []

                        st.markdown("**Answer:**")
                        st.success(answer)

                        with st.expander("Sources retrieved from knowledge base"):
                            st.caption(f"Source: **{kb_source.upper()}** "
                                       f"(score-filtered retrieval, threshold=0.3)")
                            if context and "does not contain" not in context:
                                st.text(context[:600] + "...")

                    except Exception as e:
                        st.error(f"RAG error: {e}")
                        if llm_choice == "ollama":
                            st.caption("Is `ollama serve` running? Try: `ollama serve` in a terminal.")

            st.divider()

            # ── How it works explainer ────────────────────────────────────────
            with st.expander("How the RAG pipeline works"):
                st.markdown("""
**RAG = Retrieval-Augmented Generation**

Instead of the LLM answering from training memory alone, it first retrieves
relevant chunks from *your* knowledge base, then generates a response grounded
in those facts. This prevents hallucination on project-specific details.

**Pipeline:**
1. Your question → embedded to a 384-dim vector (all-MiniLM-L6-v2)
2. ChromaDB finds the 3 most semantically similar chunks from the knowledge base
3. Retrieved chunks + your question → filled into the fraud analyst prompt
4. LLM generates a 2–3 sentence answer grounded in the retrieved text
5. Source documents are returned for full transparency

**Knowledge base files:**

| File | Contents |
|------|----------|
| `eda_findings.txt` | Key patterns from the Sparkov EDA |
| `fraud_patterns.txt` | Fraud typologies: card-testing, CNP, account takeover |
| `feature_explanations.txt` | What each SHAP feature means in fraud ops terms |
| `historical_cases.txt` | Example flagged transactions with outcomes |
""")
