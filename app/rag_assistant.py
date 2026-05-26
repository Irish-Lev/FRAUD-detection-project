"""
rag_assistant.py
RAG module for the Fraud Detection Streamlit app.

Key improvement over v1: uses similarity_search_with_relevance_scores()
with a score threshold — inspired by teacher's RAG lab approach.
If no chunk scores above the threshold, the LLM answers from its own
fraud domain expertise rather than hallucinating from irrelevant chunks.

LLM Options:
  openai  — GPT-4o-mini (set OPENAI_API_KEY in .env)
  claude  — Claude Haiku (set ANTHROPIC_API_KEY in .env)
  ollama  — Mistral local (run `ollama serve` first)
"""
import os
from pathlib import Path

APP_DIR        = Path(__file__).parent
ROOT           = APP_DIR.parent
CHROMA_DB_PATH = ROOT / 'data' / 'chroma_db'


# ── Prompt 1: Transaction explanation ─────────────────────────────────────────
EXPLAIN_PROMPT_TEMPLATE = """## SYSTEM ROLE
You are a fraud analyst assistant at a payment platform like Stripe or Revolut.
A transaction has been flagged by the ML model and you must explain WHY it was flagged
in plain English that an ops agent can immediately act on.

## TASK
Explain why this specific transaction was flagged as fraud.
Be direct and specific. Use fraud operations terminology.
Keep your response to 2-3 sentences maximum.

## CONTEXT FROM FRAUD KNOWLEDGE BASE
{context}

## FLAGGED TRANSACTION DETAILS
{question}

## ANALYST EXPLANATION (2-3 sentences):"""


# ── Prompt 2: General Q&A ─────────────────────────────────────────────────────
QA_PROMPT_TEMPLATE = """## SYSTEM ROLE
You are a senior fraud analyst and ML systems expert with deep knowledge of payment fraud
detection, risk operations, machine learning models, and fraud ops workflows.
You are equivalent to a lead fraud specialist at Stripe, Revolut, or Adyen.

## QUESTION TYPE DETECTION — READ THIS FIRST
Before answering, classify the question:
- TYPE A — CONCEPTUAL: asks how something works, what differentiates X from Y,
  how a model would detect something, what signals indicate something.
  → Answer as an expert explaining a mechanism, methodology, or concept.
  → Use examples. Structure in clear paragraphs. NEVER say "This transaction was flagged".
- TYPE B — TRANSACTION: explicitly describes a specific transaction with a fraud score,
  SHAP values, amount, category etc.
  → Answer as an analyst explaining that specific flagged case.

## GUIDELINES
1. Answer the EXACT question asked — do not reframe it as something else.
2. For TYPE A questions: explain clearly with examples. 2-3 paragraphs.
3. For TYPE B questions: explain the specific transaction in 2-3 sentences.
4. Use the KNOWLEDGE BASE as your primary source.
5. If the knowledge base context says "does not contain a direct answer",
   answer from your broader fraud domain expertise — do not refuse to answer.
6. Never speculate beyond fraud domain knowledge.
7. Format your response in clean, readable prose.

## CONTEXT FROM FRAUD KNOWLEDGE BASE
{context}

## USER QUESTION
{question}

## ANSWER:"""


def _load_api_key(env_var: str) -> str | None:
    """Read an API key from environment or .env file."""
    key = os.getenv(env_var)
    if key:
        return key
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{env_var}="):
                return line.split("=", 1)[1].strip()
    return None


def _get_context_with_scores(vectorstore, question: str,
                              score_threshold: float = 0.3,
                              k: int = 5) -> tuple[str, str]:
    """
    Retrieve chunks from ChromaDB using relevance scores.
    Only keeps chunks above score_threshold — prevents hallucination
    from irrelevant chunks (teacher's approach from RAG lab).

    Returns:
        (formatted_context, source)
        source: "knowledge_base" or "domain_expertise"
    """
    results = vectorstore.similarity_search_with_relevance_scores(question, k=k)
    good_docs = [doc for doc, score in results if score >= score_threshold]

    if good_docs:
        context = "\n\n".join([
            f"[Source: {Path(d.metadata.get('source', '?')).name}]\n{d.page_content}"
            for d in good_docs
        ])
        return context, "knowledge_base"

    # No relevant chunks — tell LLM to use its domain expertise
    return (
        "The fraud knowledge base does not contain a direct answer to this question. "
        "Use your broader fraud detection and payment operations expertise to answer.",
        "domain_expertise"
    )


def load_rag_chain(llm_option: str = "openai"):
    """
    Load ChromaDB and return RAG chains dict.
    Called once at Streamlit startup via @st.cache_resource.

    Returns:
        {'explain': chain, 'qa': chain, 'vectorstore': vs} or None
    """
    if not CHROMA_DB_PATH.exists():
        return None

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.prompts import PromptTemplate
    from langchain_classic.chains import RetrievalQA

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DB_PATH),
        embedding_function=embeddings,
    )

    explain_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_retriever      = vectorstore.as_retriever(search_kwargs={"k": 5})

    if llm_option == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=_load_api_key("OPENAI_API_KEY"),
        )
    elif llm_option == "claude":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=0.3,
            anthropic_api_key=_load_api_key("ANTHROPIC_API_KEY"),
        )
    else:
        from langchain_community.llms import Ollama
        llm = Ollama(model="mistral", temperature=0.3)

    explain_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=explain_retriever,
        chain_type_kwargs={"prompt": PromptTemplate(
            input_variables=["context", "question"],
            template=EXPLAIN_PROMPT_TEMPLATE,
        )},
        return_source_documents=True,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=qa_retriever,
        chain_type_kwargs={"prompt": PromptTemplate(
            input_variables=["context", "question"],
            template=QA_PROMPT_TEMPLATE,
        )},
        return_source_documents=True,
    )

    # Return vectorstore too so ask_knowledge_base() can use score filtering
    return {"explain": explain_chain, "qa": qa_chain, "vectorstore": vectorstore}


def explain_transaction(chains,
                         fraud_score: float,
                         top_features: dict,
                         transaction: dict) -> tuple[str, list]:
    """
    Generate plain-English fraud explanation for a specific transaction.
    Uses EXPLAIN_PROMPT — tight, 2-3 sentences, transaction-specific.
    """
    chain = chains["explain"] if isinstance(chains, dict) else chains
    feat_str = ", ".join(
        f"{k}={v:+.3f}" for k, v in
        sorted(top_features.items(), key=lambda x: abs(x[1]), reverse=True)
    )
    query = (
        f"Transaction flagged at {fraud_score:.1%} fraud probability. "
        f"Key SHAP risk signals: {feat_str}. "
        f"Category: {transaction.get('category', 'unknown')}, "
        f"Amount: ${transaction.get('amount', 0):.2f}, "
        f"Hour: {transaction.get('hour', 0):02d}:00, "
        f"Distance from home: {transaction.get('geo_distance_km', 0):.0f} km. "
        f"Why was this transaction flagged as fraud?"
    )
    result = chain.invoke({"query": query})
    return result["result"], result.get("source_documents", [])


def ask_knowledge_base(chains, question: str,
                        score_threshold: float = 0.3) -> tuple[str, list]:
    """
    Free-text Q&A using score-filtered retrieval (teacher's approach).
    Conceptual questions get answered from domain expertise when KB
    chunks are not relevant enough.

    Args:
        chains:          Dict from load_rag_chain()
        question:        Plain-English question from analyst
        score_threshold: Minimum relevance score. Default 0.3.

    Returns:
        (answer_text, source_docs)
    """
    vectorstore = chains.get("vectorstore") if isinstance(chains, dict) else None

    # Use score-filtered retrieval if vectorstore is available
    if vectorstore is not None:
        context, source = _get_context_with_scores(
            vectorstore, question, score_threshold=score_threshold, k=5
        )
        # Build prompt manually and call LLM directly for full control
        llm_chain = chains["qa"]
        # Inject context directly — bypass chain's own retriever
        full_prompt = QA_PROMPT_TEMPLATE.format(
            context=context, question=question
        )
        # Use the chain's LLM via a simple invoke
        from langchain_core.messages import HumanMessage
        try:
            llm = llm_chain.combine_documents_chain.llm_chain.llm
            response = llm.invoke([HumanMessage(content=full_prompt)])
            answer = response.content if hasattr(response, 'content') else str(response)
            return answer, []
        except Exception:
            # Fallback to standard chain if direct LLM call fails
            pass

    # Standard chain fallback
    chain = chains["qa"] if isinstance(chains, dict) else chains
    result = chain.invoke({"query": question})
    return result["result"], result.get("source_documents", [])
