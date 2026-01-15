from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import re

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import LlamaCpp


# =========================
# Configuration
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = "chroma"
COLLECTION_NAME = "storage_unified"
EMBEDDING_MODEL_NAME = "nomic-embed-text"

LLAMA_MODEL_PATH = str(BASE_DIR / "models" / "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf")

# Retrieval
TOP_K = 7
MMR_FETCH_K = 28
MMR_LAMBDA = 0.6

# Context size
MAX_CONTEXT_CHARS = 4500  # keep tight -> less hallucination & faster

# Gate (distance, lower=better)
GATE_K = 5
RELEVANCE_MARGIN = 0.22
MIN_GOOD_HITS = 2

# Output policy
FALLBACK = "I don't know based on the documentation."
MIN_BULLETS = 4
MAX_BULLETS = 8

# Debug
DEBUG_RETRIEVAL = False
DEBUG_PROMPT = False


# =========================
# Lazy singletons
# =========================
_embeddings: Optional[OllamaEmbeddings] = None
_vectordb: Optional[Chroma] = None
_llm: Optional[LlamaCpp] = None


def _get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    return _embeddings


def _get_vectordb() -> Chroma:
    global _vectordb
    if _vectordb is None:
        _vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            collection_name=COLLECTION_NAME,
            embedding_function=_get_embeddings(),
        )
    return _vectordb


def _get_llm() -> LlamaCpp:
    """
    IMPORTANT:
    - We use a chat-style format inside a single prompt (system + user),
      because many llama.cpp setups won’t apply the model’s chat template automatically.
    - Keep n_ctx moderate for speed.
    """
    global _llm
    if _llm is None:
        _llm = LlamaCpp(
            model_path=LLAMA_MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=4096,
            temperature=0.0,
            top_p=0.9,
            repeat_penalty=1.15,
            max_tokens=220,
            verbose=False,
        )
    return _llm


# =========================
# Query helpers
# =========================
_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "with", "by",
    "is", "are", "was", "were", "be", "being", "been", "as", "at", "from",
    "this", "that", "these", "those", "it", "its", "your", "you", "we",
    "how", "why", "what", "when", "where", "who", "which",
    "should", "can", "could", "would", "do", "does", "did",
}

def _query_terms(q: str) -> List[str]:
    """
    Extract key terms to keep context focused.
    """
    q = (q or "").strip().lower()
    # keep alphanum tokens, allow hyphen
    toks = re.findall(r"[a-z0-9][a-z0-9\-]{1,}", q)
    toks = [t for t in toks if t not in _STOPWORDS and len(t) >= 3]
    # de-dup preserve order
    seen = set()
    out = []
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _is_definition_question(q: str) -> bool:
    ql = (q or "").strip().lower()
    return (
        ql.startswith("what is ")
        or ql.startswith("wat is ")
        or ql.startswith("define ")
        or ql.startswith("meaning of ")
    )


def _doc_matches_terms(doc, terms: List[str]) -> bool:
    """
    Require at least one key term to appear in doc.
    """
    if not terms:
        return True
    t = (doc.page_content or "").lower()
    return any(term in t for term in terms)


# =========================
# Context cleaning
# =========================
_BAD_LINES_PATTERNS = [
    r"^\s*another question\??\s*$",
    r"^\s*rss\s*$",
    r"^\s*blog in the spotlight\s*$",
    r"^\s*story in the spotlight\s*$",
    r"^\s*make your software fair\s*$",
    r"^\s*new metadata catalogue\s*$",
    r"^\s*\d{1,2}\s+\w+\s+\d{4}\s+\d{2}:\d{2}\s+to\s+\d{1,2}\s+\w+\s+\d{4}\s+\d{2}:\d{2}\s*$",
]

def _clean_text(text: str) -> str:
    """
    Remove common webpage noise that causes the model to improvise.
    """
    if not text:
        return ""
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        if any(re.match(pat, low) for pat in _BAD_LINES_PATTERNS):
            continue
        # drop super-short nav-like fragments
        if len(s) <= 2:
            continue
        lines.append(s)
    # also collapse repeated whitespace
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    return cleaned


def _truncate_context(blocks: List[str]) -> List[str]:
    joined = "\n\n".join(blocks)
    if len(joined) <= MAX_CONTEXT_CHARS:
        return blocks
    # truncate by cutting blocks from the end
    out = []
    total = 0
    for b in blocks:
        if total + len(b) + 2 > MAX_CONTEXT_CHARS:
            break
        out.append(b)
        total += len(b) + 2
    return out


# =========================
# Retrieval + gating
# =========================
def _gate_relevance(query: str) -> bool:
    scored = _get_vectordb().similarity_search_with_score(query, k=GATE_K)
    if not scored:
        return False
    best = min(score for _, score in scored)
    good_hits = [score for _, score in scored if score <= best + RELEVANCE_MARGIN]
    ok = len(good_hits) >= MIN_GOOD_HITS

    if DEBUG_RETRIEVAL:
        print("DEBUG gate:", best, "good:", len(good_hits), "ok:", ok)
        for i, (d, s) in enumerate(scored, 1):
            snip = (d.page_content or "").replace("\n", " ")[:140]
            print(i, f"{s:.3f}", d.metadata, snip)

    return ok


def _retrieve_mmr_docs(query: str, k: int = TOP_K):
    return _get_vectordb().max_marginal_relevance_search(
        query=query,
        k=k,
        fetch_k=MMR_FETCH_K,
        lambda_mult=MMR_LAMBDA,
    )


def _prefer_definition_docs(docs, query: str):
    """
    Generic heuristic: for "what is X" keep definition/overview-like chunks,
    drop narrow FAQ/feature fragments (DOI/version/etc.) unless nothing else exists.
    """
    if not _is_definition_question(query):
        return docs

    good = []
    rest = []
    for d in docs:
        t = (d.page_content or "").lower()
        if (" is a " in t or " is an " in t or " enables you to " in t or " designed " in t):
            good.append(d)
        else:
            rest.append(d)

    # If we found definition-like chunks, use those first
    if good:
        return good + rest
    return docs


# =========================
# Prompt + output enforcement
# =========================
def _build_prompt(question: str, context: str) -> str:
    """
    Chat-style prompt (system + user) in one string.
    This tends to obey rules better than a plain instruction block.
    """
    system = (
        "You are Storage Finder.\n"
        "You answer questions about Utrecht University research data storage using ONLY the provided context.\n"
        "If the context is insufficient, you MUST reply exactly:\n"
        f"{FALLBACK}\n"
        "Output format rules:\n"
        f"- Output ONLY bullet points starting with '- ' and nothing else.\n"
        f"- Use {MIN_BULLETS} to {MAX_BULLETS} bullets.\n"
        "- Do not mention sources.\n"
        "- Do not add advice, contacts, subscriptions, or access procedures unless explicitly stated in the context.\n"
    )
    user = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:\n"
    )

    # Lightweight ChatML-ish wrapper; works reasonably across llama.cpp setups
    return (
        "<|begin_of_text|>\n"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system}\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user}\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def _ensure_bullets_or_fallback(text: str) -> str:
    """
    Hard enforcement: return bullets only, else fallback.
    """
    t = (text or "").strip()
    if not t:
        return FALLBACK

    # If the model included fallback anywhere, enforce the exact fallback only.
    if FALLBACK.lower() in t.lower():
        return FALLBACK

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    bullets = [ln for ln in lines if ln.startswith("- ")]
    if MIN_BULLETS <= len(bullets):
        return "\n".join(bullets[:MAX_BULLETS]).strip()

    return FALLBACK


def _filter_bullets_by_terms(answer: str, terms: List[str]) -> str:
    """
    Prevent “extra helpful” bullets unrelated to the question.
    Keep bullets that mention at least one key term (if we have terms).
    """
    if not terms:
        return answer

    lines = [ln.strip() for ln in (answer or "").splitlines() if ln.strip()]
    bullets = [ln for ln in lines if ln.startswith("- ")]
    if not bullets:
        return answer

    kept = []
    for b in bullets:
        bl = b.lower()
        if any(t in bl for t in terms):
            kept.append(b)

    if len(kept) >= MIN_BULLETS:
        return "\n".join(kept[:MAX_BULLETS]).strip()

    return "\n".join(bullets[:MAX_BULLETS]).strip()


# =========================
# Public API
# =========================
def ask_storage_question(
    user_input: str,
    chat_history: List[Tuple[str, str]] | List[Dict[str, Any]] | None = None,
) -> str:
    question = (user_input or "").strip()
    if not question:
        return FALLBACK

    if not _gate_relevance(question):
        return FALLBACK

    docs = _retrieve_mmr_docs(question, k=TOP_K)
    if not docs:
        return FALLBACK

    # Focus by query terms (generic)
    terms = _query_terms(question)
    if terms:
        term_docs = [d for d in docs if _doc_matches_terms(d, terms)]
        if term_docs:
            docs = term_docs

    # For "what is X" prefer definition-ish chunks
    docs = _prefer_definition_docs(docs, question)

    # Build context (clean + compact)
    context_blocks: List[str] = []
    for d in docs:
        txt = _clean_text(d.page_content or "")
        if not txt:
            continue
        context_blocks.append(txt)

    if not context_blocks:
        return FALLBACK

    context_blocks = _truncate_context(context_blocks)
    context = "\n\n".join(context_blocks).strip()

    prompt = _build_prompt(question, context)

    if DEBUG_PROMPT:
        print(prompt)

    llm = _get_llm()
    raw = llm.invoke(prompt) if hasattr(llm, "invoke") else llm.predict(prompt)

    if isinstance(raw, dict):
        raw = raw.get("text") or raw.get("output") or str(raw)

    answer = _ensure_bullets_or_fallback(str(raw))
    answer = _filter_bullets_by_terms(answer, terms)
    return answer
