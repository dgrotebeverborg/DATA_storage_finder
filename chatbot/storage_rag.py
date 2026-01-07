# logic/rag.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp


# -------------------------
# Config (absolute paths)
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # pas aan als jouw structuur anders is

CHROMA_DIR = str(BASE_DIR / "chroma_storage")
COLLECTION_NAME = "storage_unified"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLAMA_MODEL_PATH = str(BASE_DIR / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

TOP_K = 5

# Chroma score is meestal "distance": lager = beter
MAX_DISTANCE_THRESHOLD = 0.80  # tune: hoger = minder "I don't know", lager = strenger

# Hoeveel context we max in de prompt stoppen (karakters)
MAX_CONTEXT_CHARS = 9000

DEBUG_RETRIEVAL = False


# -------------------------
# Singletons
# -------------------------
_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectordb: Optional[Chroma] = None
_llm: Optional[LlamaCpp] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,  model_kwargs={"device": "cuda"}  )
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
            temperature=0.0,       # 🔥 strak: geen creativiteit
            top_p=0.9,
            repeat_penalty=1.15,   # 🔥 minder herhaling
            max_tokens=220,        # 🔥 compact
            verbose=False,
        )
    return _llm


def _retrieve_with_scores(query: str, k: int = TOP_K):
    """
    Returns list of (Document, score). For Chroma this is typically a distance score (lower is better).
    """
    return _get_vectordb().similarity_search_with_score(query, k=k)


def _is_relevant(retrieved) -> bool:
    if not retrieved:
        return False

    # Count how many docs are reasonably close
    good_hits = [score for _, score in retrieved if score <= MAX_DISTANCE_THRESHOLD]

    # Require at least 2 good hits for confidence
    return len(good_hits) >= 2



def _build_prompt(question: str, context_blocks: List[str]) -> str:
    context_text = "\n\n".join(context_blocks).strip()
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated]"

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
        f"Context:\n{context_text}\n\n"
        f"Question:\n{question}\n\n"
        "Answer (bullet points only):"
    )


def _strip_meta(text: str) -> str:
    BAD_PREFIXES = (
        "note:",
        "the final answer",
        "corrected answer",
        "this answer",
        "i have followed",
        "the answer includes",
    )

    lines = []
    for ln in text.splitlines():
        l = ln.strip()
        if not l:
            continue
        if l.lower().startswith(BAD_PREFIXES):
            break  # alles daarna weggooien
        lines.append(l)

    return "\n".join(lines).strip()


def _dedupe_lines(text: str) -> str:
    """
    Optional safety net against repeated paragraphs/lines.
    """
    lines = [ln.strip() for ln in (text or "").splitlines()]
    out = []
    seen = set()
    for ln in lines:
        if not ln:
            continue
        key = ln.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(ln)
    return "\n".join(out).strip()


def ask_storage_question(
    user_input: str,
    chat_history: List[Tuple[str, str]] | List[Dict[str, Any]] | None = None,
) -> str:
    """
    Strict RAG answerer.
    - chat_history accepted for compatibility, but intentionally NOT used (prevents persona leakage and bad rewrites).
    """
    question = (user_input or "").strip()
    hits = _get_vectordb().similarity_search_with_score("what is surfdrive", k=5)
    for d, s in hits:
        print("score", s, "src", d.metadata.get("source"), d.page_content[:120])

    if not question:
        return "I don't know based on the documentation."

    retrieved = _retrieve_with_scores(question, k=TOP_K)

    if DEBUG_RETRIEVAL:
        print("DEBUG retrieved:", len(retrieved))
        for i, (d, score) in enumerate(retrieved, 1):
            snippet = (d.page_content or "").replace("\n", " ")[:200]
            print(i, f"score={score:.3f}", d.metadata, snippet)

    # HARD GATE: no good context => no answer
    if not _is_relevant(retrieved):
        return "I don't know based on the documentation."

    # Build context blocks with lightweight source labels
    context_blocks: List[str] = []
    for doc, score in retrieved:
        src = doc.metadata.get("source", "unknown")
        text = (doc.page_content or "").strip()
        if not text:
            continue
        context_blocks.append(f"[source: {src} | score: {score:.3f}]\n{text}")

    prompt = _build_prompt(question, context_blocks)

    llm = _get_llm()


    # LangChain compat: sommige versies gebruiken .invoke(), andere .predict()
    if hasattr(llm, "invoke"):
        answer = llm.invoke(prompt)
    elif hasattr(llm, "predict"):
        answer = llm.predict(prompt)
    else:
        raise TypeError("Your LlamaCpp wrapper does not support invoke() or predict().")

    # Some LLM wrappers may return dict-like; normalize
    if isinstance(answer, dict):
        answer = answer.get("text") or answer.get("output") or str(answer)

    answer = str(answer).strip()
    answer = _dedupe_lines(answer)
    answer = _strip_meta(answer)

    # Extra guard: if model still tries to be helpful without support, enforce the fallback
    if not answer:
        return "I don't know based on the documentation."

    return answer
