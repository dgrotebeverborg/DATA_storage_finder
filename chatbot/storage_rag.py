# logic/rag.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp

# Correcte imports voor LangChain 1.2.x (2026)
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from typing import List as TypingList


# Simple ensemble retriever implementation
class EnsembleRetriever:
    def __init__(self, retrievers: TypingList, weights: TypingList[float]):
        self.retrievers = retrievers
        self.weights = weights

    def invoke(self, query: str) -> TypingList[Document]:
        all_docs = []
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query) if hasattr(retriever, 'invoke') else retriever.get_relevant_documents(query)
            all_docs.extend(docs)

        # Remove duplicates based on page_content
        seen_content = set()
        unique_docs = []
        for doc in all_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)

        return unique_docs

# Voor reranking
from sentence_transformers import CrossEncoder


# -------------------------
# Config
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = str(BASE_DIR / "chroma_storage")
COLLECTION_NAME = "storage_unified_2026"          # pas aan naar jouw nieuwe collectie

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
LLAMA_MODEL_PATH = str(BASE_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

RETRIEVE_K = 20
RERANK_TOP = 6
MAX_DISTANCE_THRESHOLD = 0.75
MAX_CONTEXT_CHARS = 9000
DEBUG_RETRIEVAL = True

RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"

LLM_TEMPERATURE = 0.15
LLM_MAX_TOKENS = 450


# -------------------------
# Singletons
# -------------------------
_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectordb: Optional[Chroma] = None
_llm: Optional[LlamaCpp] = None
_reranker: Optional[CrossEncoder] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _get_vectordb() -> Chroma:
    global _vectordb
    if _vectordb is None:
        _vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=_get_embeddings(),
            collection_name=COLLECTION_NAME,
        )
    return _vectordb


def _get_llm() -> LlamaCpp:
    global _llm
    if _llm is None:
        _llm = LlamaCpp(
            model_path=LLAMA_MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=8192,
            temperature=LLM_TEMPERATURE,
            top_p=0.9,
            repeat_penalty=1.15,
            max_tokens=LLM_MAX_TOKENS,
            verbose=False,
        )
    return _llm


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL, device="cuda", trust_remote_code=True)
    return _reranker


def _retrieve_with_scores(query: str, k: int = RETRIEVE_K) -> List[Tuple[Document, float]]:
    """
    Hybrid retrieval: BM25 + vector + reranking
    """
    # Haal alle docs op voor BM25 (kan later geoptimaliseerd worden)
    collection_data = _get_vectordb().get()
    docs = [
        Document(page_content=content, metadata=meta or {})
        for content, meta in zip(collection_data["documents"], collection_data["metadatas"])
    ]

    # BM25
    bm25_retriever = BM25Retriever.from_documents(docs, k=k // 2)

    # Vector
    vector_retriever = _get_vectordb().as_retriever(search_kwargs={"k": k // 2})

    # Ensemble
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.35, 0.65],
    )

    retrieved_docs = ensemble_retriever.invoke(query)

    if DEBUG_RETRIEVAL:
        print(f"DEBUG: Ensemble retrieved {len(retrieved_docs)} docs")

    # Fake scores voor compatibiliteit (reranker komt later)
    return [(doc, 0.0) for doc in retrieved_docs]


def _is_relevant(retrieved: List[Tuple[Document, float]]) -> bool:
    return len(retrieved) >= 3  # aangepast voor hybrid


def _build_prompt(
    question: str,
    context_blocks: List[str],
    chat_history: Optional[List[Tuple[str, str]]] = None
) -> str:
    context_text = "\n\n".join(context_blocks).strip()
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated]"

    history_text = ""
    if chat_history:
        history_text = "\nRecent conversation:\n" + "\n".join(
            [f"User: {q}\nAnswer: {a}" for q, a in chat_history[-2:]]
        )

    return (
        "You are Storage Finder, a factual assistant for Utrecht University data storage guidance.\n\n"
        "STRICT RULES (must follow):\n"
        "- Use ONLY the provided context. Do NOT use outside knowledge.\n"
        "- Do NOT invent numbers, limits, prices, or policies.\n"
        "- Do NOT suggest tools or services unless explicitly mentioned in the context.\n"
        "- Treat 'Yoda' as the Utrecht University data management system (not Star Wars).\n"
        "- Do NOT imitate characters, writing styles, or personas.\n"
        "- Do NOT include meta-commentary, notes, or self-evaluation.\n"
        "- Do NOT repeat the same fact in different words.\n"
        "- ONLY if the context does NOT contain enough information to answer the question at all,\n"
        "  reply exactly:\n"
        "  I don't know based on the documentation.\n"
        "- If partial information IS available, answer using ONLY that information and do NOT add the fallback sentence.\n\n"
        "ANSWER STYLE:\n"
        "- Use a single-level bullet list only.\n"
        "- Do NOT nest bullets and do NOT use section headers.\n"
        "- Each bullet must be a complete sentence.\n"
        "- Use 4–8 bullets.\n"
        "- If the question is 'what is X', include (only if supported by context):\n"
        "  what it is, main purpose, key benefits, and a typical use-case.\n"
        "- Include concrete numbers only if they appear in the context, and only once.\n\n"
        f"{history_text}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question:\n{question}\n\n"
        "Answer (bullet points only):"
    )


def _strip_meta(text: str) -> str:
    BAD_PREFIXES = (
        "note:", "the final answer", "corrected answer", "this answer",
        "i have followed", "the answer includes",
    )
    lines = []
    for ln in text.splitlines():
        l = ln.strip()
        if not l:
            continue
        if l.lower().startswith(BAD_PREFIXES):
            break
        lines.append(l)
    return "\n".join(lines).strip()


def _dedupe_lines(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    out = []
    seen = set()
    for ln in lines:
        key = ln.lower()
        if key not in seen:
            seen.add(key)
            out.append(ln)
    return "\n".join(out).strip()


def ask_storage_question(
    user_input: str,
    chat_history: List[Tuple[str, str]] | List[Dict[str, Any]] | None = None,
) -> str:
    question = (user_input or "").strip()
    if not question:
        return "I don't know based on the documentation."

    retrieved = _retrieve_with_scores(question, k=RETRIEVE_K)

    if DEBUG_RETRIEVAL:
        print("DEBUG pre-rerank:", len(retrieved))
        for i, (d, s) in enumerate(retrieved, 1):
            snippet = d.page_content.replace("\n", " ")[:200]
            print(i, f"score={s:.3f}", d.metadata, snippet)

    if not _is_relevant(retrieved):
        return "I don't know based on the documentation."

    # Rerank
    reranker = _get_reranker()
    pairs = [[question, doc.page_content] for doc, _ in retrieved]
    scores = reranker.predict(pairs)

    sorted_retrieved = sorted(zip(scores, [d for d, _ in retrieved]), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in sorted_retrieved[:RERANK_TOP]]

    if DEBUG_RETRIEVAL:
        print("\nDEBUG post-rerank:")
        for i, doc in enumerate(top_docs, 1):
            snippet = doc.page_content.replace("\n", " ")[:200]
            print(i, doc.metadata, snippet)

    # Context blocks
    context_blocks = []
    for doc in top_docs:
        src = doc.metadata.get("source", "unknown")
        text = doc.page_content.strip()
        if text:
            context_blocks.append(f"[source: {src}]\n{text}")

    prompt = _build_prompt(question, context_blocks, chat_history)

    llm = _get_llm()
    if hasattr(llm, "invoke"):
        answer = llm.invoke(prompt)
    elif hasattr(llm, "predict"):
        answer = llm.predict(prompt)
    else:
        raise TypeError("LlamaCpp does not support invoke() or predict().")

    if isinstance(answer, dict):
        answer = answer.get("text") or answer.get("output") or str(answer)

    answer = str(answer).strip()
    answer = _dedupe_lines(answer)
    answer = _strip_meta(answer)

    if not answer:
        return "I don't know based on the documentation."

    return answer