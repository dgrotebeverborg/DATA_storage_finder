# logic/rag.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import requests
from langchain_core.documents import Document
from langchain_chroma import Chroma


# -------------------------
# Config
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = str(BASE_DIR / "chroma_storage")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "storage_unified_2026_ollama")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

RETRIEVE_K = int(os.environ.get("RETRIEVE_K", "8"))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "9000"))
DEBUG_RETRIEVAL = os.environ.get("DEBUG_RETRIEVAL", "true").lower() == "true"

LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.15"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "250"))


# -------------------------
# Ollama adapters
# -------------------------
class OllamaEmbeddings:
    """Minimale embedding adapter voor LangChain/Chroma via Ollama /api/embeddings."""
    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        r = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["embedding"]


def _ollama_chat(messages: List[Dict[str, str]]) -> str:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_CHAT_MODEL,
            "messages": messages,
            "options": {
                "temperature": LLM_TEMPERATURE,
                "num_predict": LLM_MAX_TOKENS,
            },
            "stream": False,
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def _ensure_ollama_models_exist() -> None:
    # geeft een duidelijke fout als iemand vergeet te pullen
    r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=30)
    r.raise_for_status()
    models = {m["name"] for m in r.json().get("models", [])}

    def exists(name: str) -> bool:
        return name in models or any(m.startswith(name + ":") for m in models)

    missing = [m for m in (OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL) if not exists(m)]

    if missing:
        raise RuntimeError(
            "Ollama models missing: "
            + ", ".join(missing)
            + ". Run: "
            + " && ".join([f"ollama pull {m}" for m in missing])
        )


# -------------------------
# Singletons
# -------------------------
_embeddings: Optional[OllamaEmbeddings] = None
_vectordb: Optional[Chroma] = None


def _get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBED_MODEL)
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


# -------------------------
# Prompt + post-processing
# -------------------------
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
        "- NEVER speculate, guess, or infer information that is not explicitly present in the context.\n"
        "- Do NOT explain what a term might refer to if it is not described in the context.\n"
        "- Do NOT invent numbers, limits, prices, classifications, or policies.\n"
        "- Do NOT suggest tools or services unless explicitly mentioned in the context.\n"
        "- Treat 'Yoda' as the Utrecht University data management system (not Star Wars), "
        "but ONLY if this is explicitly supported by the context.\n"
        "- Do NOT include meta-commentary, notes, self-evaluation, or reasoning about missing information.\n"
        "- Do NOT repeat the same fact in different words.\n"
        "- If no usable context exists for ANY part of the question, reply exactly:\n"
        "  I don't know based on the documentation.\n"
        "- If partial information IS available, answer using ONLY that information and do NOT add the fallback sentence.\n"
        "- If no usable context exists for a specific item mentioned in the question, do NOT describe that item at all.\n\n"
        "ANSWER STYLE:\n"
        "- Write in clear, concise paragraphs (not bullet points).\n"
        "- Write paragraphs ONLY for storage solutions or concepts for which usable context is available.\n"
        "- Use one short paragraph per described storage solution or concept.\n"
        "- For comparison questions, explicitly contrast the described items using natural language "
        "(e.g. 'while', 'whereas', 'in contrast'), but ONLY when both items are supported by context.\n"
        "- Do NOT introduce sub-types, variants, alternatives, or examples unless explicitly mentioned in the context.\n"
        "- Keep the total answer concise (typically 2–5 short paragraphs).\n"
        "- Use neutral, explanatory language suitable for research support and policy contexts.\n\n"
        f"{history_text}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )




def _strip_meta(text: str) -> str:
    BAD_PREFIXES = (
        "note:", "the final answer", "corrected answer", "this answer",
        "i have followed", "the answer includes",
    )
    lines = []
    for ln in (text or "").splitlines():
        l = ln.strip()
        if not l:
            continue
        if l.lower().startswith(BAD_PREFIXES):
            break
        lines.append(l)
    return "\n".join(lines).strip()


def _dedupe_lines(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    out, seen = [], set()
    for ln in lines:
        key = ln.lower()
        if key not in seen:
            seen.add(key)
            out.append(ln)
    return "\n".join(out).strip()


# -------------------------
# Public API
# -------------------------
def ask_storage_question(
    user_input: str,
    chat_history: List[Tuple[str, str]] | List[Dict[str, Any]] | None = None,
) -> str:
    question = (user_input or "").strip()
    if not question:
        return "I don't know based on the documentation."

    _ensure_ollama_models_exist()

    # Retrieval
    retriever = _get_vectordb().as_retriever(search_kwargs={"k": RETRIEVE_K})
    docs: List[Document]
    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(question)
    else:
        docs = retriever.get_relevant_documents(question)

    if DEBUG_RETRIEVAL:
        print(f"DEBUG: retrieved {len(docs)} docs")
        for i, d in enumerate(docs, 1):
            snippet = (d.page_content or "").replace("\n", " ")[:200]
            print(i, d.metadata, snippet)

    if not docs or len(docs) < 2:
        return "I don't know based on the documentation."

    # Context blocks
    context_blocks = []
    for doc in docs:
        src = doc.metadata.get("source") or doc.metadata.get("url") or "unknown"
        text = (doc.page_content or "").strip()
        if text:
            context_blocks.append(f"[source: {src}]\n{text}")

    prompt = _build_prompt(question, context_blocks, None)

    # LLM call via Ollama
    messages = [
        {"role": "system", "content": "Return only the final answer. No meta-text."},
        {"role": "user", "content": prompt},
    ]
    answer = _ollama_chat(messages)

    answer = _dedupe_lines(answer)
    answer = _strip_meta(answer)

    if not answer:
        return "I don't know based on the documentation."
    return answer
