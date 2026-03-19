from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
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
MIN_DOC_RELEVANCE = float(os.environ.get("MIN_DOC_RELEVANCE", "0.18"))
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("LOW_CONFIDENCE_THRESHOLD", "0.31"))
LOG_DIR = str(BASE_DIR / "logs")
METRICS_LOG_FILE = str(Path(LOG_DIR) / "chat_metrics.jsonl")
MAX_RERANK_DOCS = int(os.environ.get("MAX_RERANK_DOCS", "6"))
RECENCY_WINDOW_MONTHS = int(os.environ.get("RECENCY_WINDOW_MONTHS", "18"))


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

STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with", "is", "are",
    "de", "het", "een", "en", "of", "voor", "van", "op", "met", "is", "zijn", "wat",
    "which", "what", "how", "waar", "hoe", "ik", "you", "je",
}

KNOWN_STORAGE_TERMS = {
    "yoda", "onedrive", "surfdrive", "teams", "sharepoint",
    "storage", "opslag", "archive", "archief", "gdpr", "privacy", "surf",
}


@dataclass
class RAGResult:
    response: str
    citations: List[Dict[str, Any]]
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    confidence: float = 0.0
    route: str = "rag"
    used_docs: int = 0
    detected_topic: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    chat_history: Optional[List[Tuple[str, str]]] = None,
    answer_language: str = "nl",
) -> str:
    context_text = "\n\n".join(context_blocks).strip()
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated]"

    history_text = ""
    if chat_history:
        history_text = "\nRecent conversation:\n" + "\n".join(
            [f"User: {q}\nAnswer: {a}" for q, a in chat_history[-2:]]
        )

    lang_rule = "English" if answer_language == "en" else "Dutch"

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
        f"- Respond in {lang_rule}. Keep that language consistently.\n"
        "- Every key factual statement must include a source marker like [S1], [S2], etc.\n"
        "- Use only source markers that exist in the Context.\n"
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
        "FOR COMPARISON QUESTIONS:\n"
        "- Provide only direct comparisons between the asked options.\n"
        "- Do NOT include meta statements like 'based on the provided context'.\n"
        "- If one option lacks evidence, state briefly: 'There is insufficient documented information for X.'\n\n"
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


def _normalize_chat_history(
    chat_history: List[Tuple[str, str]] | List[Dict[str, Any]] | None,
) -> List[Tuple[str, str]]:
    normalized: List[Tuple[str, str]] = []
    if not chat_history:
        return normalized
    for item in chat_history:
        if isinstance(item, dict):
            q = str(item.get("q", "")).strip()
            a = str(item.get("a", "")).strip()
            if q and a:
                normalized.append((q, a))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            q = str(item[0]).strip()
            a = str(item[1]).strip()
            if q and a:
                normalized.append((q, a))
    return normalized


def _retrieve_with_scores(question: str) -> List[Tuple[Document, float]]:
    db = _get_vectordb()
    docs_scores: List[Tuple[Document, float]] = []
    # Prefer distance-based API; relevance API can be mis-scaled in some Chroma setups.
    if hasattr(db, "similarity_search_with_score"):
        raw = db.similarity_search_with_score(question, k=RETRIEVE_K)
        dist_rows = [(doc, max(float(distance), 0.0)) for doc, distance in raw]
        if dist_rows:
            vals = [d for _, d in dist_rows]
            min_d, max_d = min(vals), max(vals)
            spread = max(max_d - min_d, 1e-9)
            n = max(len(dist_rows), 1)
            mapped: List[Tuple[Document, float]] = []
            for idx, (doc, dist) in enumerate(dist_rows):
                # Lower distance -> higher relevance.
                minmax_rel = (max_d - dist) / spread if max_d > min_d else 0.5
                rank_rel = 1.0 - (idx / (n + 1.0))
                rel = (0.8 * minmax_rel) + (0.2 * rank_rel)
                mapped.append((doc, rel))
            docs_scores = mapped

    elif hasattr(db, "similarity_search_with_relevance_scores"):
        raw = db.similarity_search_with_relevance_scores(question, k=RETRIEVE_K)
        rel_scores = [(doc, float(score)) for doc, score in raw]
        docs_scores = rel_scores
    else:
        raw_docs = db.similarity_search(question, k=RETRIEVE_K)
        docs_scores = [(doc, 0.4) for doc in raw_docs]

    seen = set()
    unique_docs_scores: List[Tuple[Document, float]] = []
    for doc, score in docs_scores:
        key = ((doc.page_content or "")[:160], doc.metadata.get("source"), doc.metadata.get("url"))
        if key in seen:
            continue
        seen.add(key)
        unique_docs_scores.append((doc, max(0.0, min(1.0, score))))
    return unique_docs_scores


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9\-]{2,}", (text or "").lower())
    return {t for t in tokens if t not in STOPWORDS}


def _keyword_overlap_score(question: str, doc: Document) -> float:
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0
    d_tokens = _tokenize((doc.page_content or "")[:2400])
    if not d_tokens:
        return 0.0
    overlap = len(q_tokens.intersection(d_tokens))
    return min(overlap / max(len(q_tokens), 1), 1.0)


def _trust_boost(doc: Document) -> float:
    trust = str(doc.metadata.get("source_trust", "")).lower()
    if trust == "high":
        return 0.08
    if trust == "medium":
        return 0.03
    return 0.0


def _recency_boost(doc: Document) -> float:
    raw = str(doc.metadata.get("fetch_date", "")).strip()
    if not raw:
        return 0.0
    try:
        year, month = raw.split("-")
        y = int(year)
        m = int(month)
        now = datetime.now(timezone.utc)
        delta_months = (now.year - y) * 12 + (now.month - m)
        if delta_months < 0:
            return 0.02
        if delta_months <= RECENCY_WINDOW_MONTHS:
            return 0.06
        if delta_months <= RECENCY_WINDOW_MONTHS * 2:
            return 0.02
    except Exception:
        return 0.0
    return 0.0


def _topic_match_boost(doc: Document, topic: Optional[str]) -> float:
    if not topic:
        return 0.0
    topic_l = topic.lower()
    hay = " ".join(
        [
            str(doc.metadata.get("source", "")),
            str(doc.metadata.get("title", "")),
            str(doc.metadata.get("slug", "")),
            (doc.page_content or "")[:1200],
        ]
    ).lower()
    return 0.12 if topic_l in hay else -0.02


def _hybrid_rerank(
    question: str,
    docs_scores: List[Tuple[Document, float]],
    topic: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    scored: List[Tuple[Document, float]] = []
    for doc, vec_score in docs_scores:
        kw = _keyword_overlap_score(question, doc)
        final = (0.68 * vec_score) + (0.24 * kw) + _trust_boost(doc) + _recency_boost(doc) + _topic_match_boost(doc, topic)
        scored.append((doc, max(0.0, min(1.0, final))))

    # Source diversity: avoid overfilling context with near-duplicate chunks from one source.
    scored.sort(key=lambda x: x[1], reverse=True)
    by_source_count: Dict[str, int] = {}
    diversified: List[Tuple[Document, float]] = []
    for doc, score in scored:
        src = str(doc.metadata.get("source") or doc.metadata.get("url") or "unknown")
        if src == "unknown":
            src = f"doc:{doc.metadata.get('doc_id', id(doc))}"
        cnt = by_source_count.get(src, 0)
        if cnt >= 1:
            continue
        by_source_count[src] = cnt + 1
        diversified.append((doc, score))
        if len(diversified) >= MAX_RERANK_DOCS:
            break
    return diversified


def _estimate_confidence(docs_scores: List[Tuple[Document, float]]) -> float:
    if not docs_scores:
        return 0.0
    top = sorted([s for _, s in docs_scores], reverse=True)[:3]
    avg_top = sum(top) / max(len(top), 1)
    coverage = min(len(docs_scores) / 4.0, 1.0)
    srcs = {
        d.metadata.get("source") or d.metadata.get("url") or "unknown"
        for d, _ in docs_scores[:4]
    }
    diversity = min(len(srcs) / 3.0, 1.0)
    confidence = (0.6 * avg_top) + (0.25 * coverage) + (0.15 * diversity)
    return max(0.0, min(1.0, confidence))


def _is_recommendation_intent(question: str) -> bool:
    q = question.lower()
    patterns = (
        "welke storage", "welke oplossing", "wat moet ik kiezen", "wat raad je aan",
        "advies", "opslag kiezen", "storage kiezen",
        "recommend", "which storage", "best storage", "what should i use",
        "choose storage", "storage advice", "which one should i choose", "which one to choose",
        "help me find", "find a good storage solution", "can you find a good storage solution",
    )
    return any(p in q for p in patterns)


def _looks_like_specific_info_question(question: str) -> bool:
    q = (question or "").strip().lower()
    info_starts = (
        "what is", "what are", "who is", "explain", "tell me about",
        "wat is", "wat zijn", "leg uit", "vertel over",
    )
    if any(q.startswith(s) for s in info_starts):
        return True
    tokens = _tokenize(q)
    return any(t in KNOWN_STORAGE_TERMS for t in tokens)


def _is_compare_question(question: str) -> bool:
    q = (question or "").lower()
    markers = (
        "compare", "difference", "differences", "versus", "vs",
        "vergelijk", "verschil", "verschillen", "tegenover",
    )
    return any(m in q for m in markers)


def _detect_language(question: str, chat_history: List[Tuple[str, str]]) -> str:
    sample = f"{question} " + " ".join(q for q, _ in chat_history[-2:])
    s = sample.lower()
    s_stripped = s.strip()
    if s_stripped.startswith(("what ", "which ", "how ", "can ", "could ", "should ", "is ", "are ", "do ", "does ")):
        return "en"
    if s_stripped.startswith(("wat ", "welke ", "hoe ", "kan ", "kun ", "is ", "zijn ")):
        return "nl"

    nl_markers = {"welke", "opslag", "gevoelig", "persoonsgegevens", "samenwerken", "hoeveel", "grootte", "kan ik"}
    en_markers = {
        "what", "which", "how", "can i", "should i", "please", "use",
        "storage", "sensitive", "personal data", "collaboration", "how much", "size"
    }
    nl_score = sum(1 for m in nl_markers if m in s)
    en_score = sum(1 for m in en_markers if m in s)
    return "en" if en_score > nl_score else "nl"


def _clarification_text(key: str, lang: str) -> str:
    texts = {
        "scope": {
            "nl": "Waar gaat je vraag precies over: opslag kiezen, vergelijken, of beleid/privacy?",
            "en": "What do you want to do exactly: choose storage, compare options, or ask about policy/privacy?",
        },
        "sensitivity": {
            "nl": "Gaat het om gevoelige data of persoonsgegevens (GDPR)?",
            "en": "Does this involve sensitive data or personal data (GDPR)?",
        },
        "collaboration": {
            "nl": "Moeten meerdere mensen tegelijk toegang hebben tot deze data?",
            "en": "Do multiple people need access to this data at the same time?",
        },
        "volume": {
            "nl": "Over welke orde van grootte gaat het ongeveer (GB of TB)?",
            "en": "What data volume are we talking about (GB or TB)?",
        },
    }
    return texts[key][lang]


def _is_smalltalk(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    # Keep this intentionally strict to avoid hijacking real recommendation questions.
    smalltalk_markers = (
        "thanks", "thank you", "thx", "dat ging goed", "ging best goed",
        "top", "prima", "mooi", "super", "dank",
    )
    # Avoid catching real recommendation questions that also include "good/best".
    if _is_recommendation_intent(q):
        return False
    return any(m in q for m in smalltalk_markers)


def _smalltalk_response(lang: str) -> str:
    if lang == "en":
        return "Good to hear. If you want, I can now help narrow down the best storage option for your specific case."
    return "Mooi om te horen. Als je wilt, kan ik nu helpen om de beste opslagoptie voor jouw situatie te kiezen."


def _needs_intake_question(question: str) -> bool:
    q = (question or "").strip().lower()
    if not _is_recommendation_intent(q):
        return False
    sensitivity_terms = ("gevoelig", "sensitive", "gdpr", "privacy", "persoonsgegevens", "confidential")
    collab_terms = ("samenwerken", "samenwerking", "collaboration", "delen", "share", "team", "group", "external")
    volume_terms = ("tb", "gb", "grote bestanden", "volume", "size", "capaciteit", "storage size")
    service_terms = ("yoda", "onedrive", "surfdrive", "sharepoint", "teams", "u-drive", "o-drive", "research drive")
    has_specifics = any(t in q for t in sensitivity_terms + collab_terms + volume_terms + service_terms)
    return not has_specifics


def _intake_question(lang: str) -> str:
    if lang == "en":
        return "Does this involve sensitive data or personal data (GDPR), and do multiple people (including external partners) need access?"
    return "Gaat het om gevoelige data of persoonsgegevens (GDPR), en moeten meerdere mensen (ook extern) toegang hebben?"


def detect_question_language(question: str, chat_history: Optional[List[Tuple[str, str]]] = None) -> str:
    return _detect_language(question, chat_history or [])


def clarification_prompt(key: str, lang: str) -> str:
    return _clarification_text(key, lang)


def _rewrite_query(question: str) -> str:
    q = (question or "").strip()
    ql = q.lower()
    expansions = []
    if "privacy" in ql or "gdpr" in ql or "persoonsgegevens" in ql or "sensitive" in ql:
        expansions.append("sensitive data GDPR privacy access control")
    if "samenwerk" in ql or "share" in ql or "collab" in ql or "team" in ql:
        expansions.append("collaboration sharing access permissions")
    if "archiv" in ql or "preserv" in ql:
        expansions.append("archive preservation long-term storage")
    if "yoda" in ql:
        expansions.append("Yoda Utrecht University research data management")
    if "onedrive" in ql:
        expansions.append("OneDrive for Business Microsoft Teams")
    if "surfdrive" in ql:
        expansions.append("SURFdrive cloud storage Dutch research")
    if not expansions:
        return q
    return f"{q}\nRelated terms: {'; '.join(expansions)}"


def _contains_followup_pronoun(question: str) -> bool:
    q = (question or "").lower()
    pronouns = ("that", "it", "this", "those", "die", "dat", "dit")
    return any(re.search(rf"\b{p}\b", q) for p in pronouns)


def _extract_topic_from_text(text: str) -> Optional[str]:
    q = (text or "").lower()
    if "object store" in q:
        return "surf object store"
    if "surf drive" in q:
        return "surfdrive"
    if re.search(r"\bsurf\b", q):
        return "surfdrive"
    for term in (
        "yoda", "surfdrive", "onedrive", "sharepoint", "teams",
        "research drive", "research-drive", "u-drive", "o-drive",
        "surf object store",
    ):
        if term in q:
            return term
    return None


def _inject_topic_if_insufficient(answer: str, topic: Optional[str], lang: str) -> str:
    if not answer or not topic:
        return answer
    a = answer.strip()
    al = a.lower()
    tl = topic.lower()
    if tl in al:
        return answer
    if "insufficient documented information" in al:
        if lang == "en":
            return f"There is insufficient documented information for {topic}."
        return f"Er is onvoldoende gedocumenteerde informatie over {topic}."
    if al in {
        "i don't know based on the documentation.",
        "ik weet het niet op basis van de documentatie.",
        "ik weet het niet op basis van de beschikbare documentatie.",
    }:
        if lang == "en":
            return f"I don't know based on the documentation for {topic}."
        return f"Ik weet het niet op basis van de beschikbare documentatie over {topic}."
    return answer


def _augment_followup_question(question: str, latest_topic: Optional[str]) -> str:
    if not latest_topic:
        return question
    if _extract_topic_from_text(question):
        return question
    if not _contains_followup_pronoun(question):
        return question
    return f"{question}\nReferenced service/topic: {latest_topic}"


def _is_bare_topic_query(question: str, topic: Optional[str]) -> bool:
    if not question or not topic:
        return False
    q = question.strip().lower()
    # Short label-like prompt without a clear question verb.
    if len(q.split()) > 4:
        return False
    question_markers = ("what", "which", "how", "can", "is", "are", "wat", "welke", "hoe", "?")
    return (topic.lower() in q) and not any(m in q for m in question_markers)


def _expand_bare_topic_query(question: str, topic: str, lang: str) -> str:
    if lang == "en":
        return f"What is {topic} used for in Utrecht University research storage?"
    return f"Waarvoor wordt {topic} gebruikt binnen onderzoeksopslag aan de Universiteit Utrecht?"


def _needs_basic_clarification(question: str, chat_history: List[Tuple[str, str]]) -> Optional[str]:
    q = question.strip().lower()
    lang = _detect_language(q, chat_history)

    # Prevent clarification loops by not repeating the exact same prompt.
    last_bot_answer = (chat_history[-1][1].strip() if chat_history else "")

    # Only ask broad scope question for very short AND ambiguous inputs.
    if len(q.split()) <= 3 and not _is_recommendation_intent(q) and not _looks_like_specific_info_question(q):
        prompt = _clarification_text("scope", lang)
        return None if last_bot_answer == prompt else prompt

    # Specific info questions should go directly to retrieval (e.g. "what is yoda?")
    if _looks_like_specific_info_question(q) and not _is_recommendation_intent(q):
        return None

    # Clarification flow is only for recommendation/intake questions.
    if not _is_recommendation_intent(q):
        return None

    history_blob = " ".join([f"{a} {b}" for a, b in chat_history]).lower()
    full_text = f"{q} {history_blob}"

    sensitivity_terms = ("gevoelig", "sensitive", "gdpr", "privacy", "persoonsgegevens", "confidential")
    collab_terms = ("samenwerken", "samenwerking", "collaboration", "delen", "share", "team", "group")
    volume_terms = ("tb", "gb", "grote bestanden", "volume", "size", "capaciteit", "storage size")
    known_service_terms = ("yoda", "onedrive", "surfdrive", "sharepoint", "teams", "u-drive", "o-drive", "research drive")

    has_sensitivity = any(t in full_text for t in sensitivity_terms)
    has_collab = any(t in full_text for t in collab_terms)
    has_volume = any(t in full_text for t in volume_terms)
    has_service = any(t in full_text for t in known_service_terms)
    filled_dimensions = sum([has_sensitivity, has_collab, has_volume])

    # If user already gives enough recommendation constraints, do not block with clarification.
    # Example: "sensitive data + collaboration" should proceed directly.
    if has_service or filled_dimensions >= 2:
        return None

    if not has_sensitivity:
        prompt = _clarification_text("sensitivity", lang)
        return None if last_bot_answer == prompt else prompt
    if not has_collab:
        prompt = _clarification_text("collaboration", lang)
        return None if last_bot_answer == prompt else prompt
    if not has_volume:
        prompt = _clarification_text("volume", lang)
        return None if last_bot_answer == prompt else prompt
    return None


def _fallback_message(lang: str) -> str:
    if lang == "en":
        return "I don't know based on the documentation."
    return "Ik weet het niet op basis van de documentatie."


def _is_likely_dutch(text: str) -> bool:
    t = (text or "").lower()
    nl_hints = (" de ", " het ", " een ", " en ", " voor ", " niet ", " je ", " kan ", " gebruik ")
    return sum(1 for h in nl_hints if h in f" {t} ") >= 2


def _is_likely_english(text: str) -> bool:
    t = (text or "").lower()
    en_hints = (" the ", " and ", " for ", " with ", " not ", " you ", " can ", " use ")
    return sum(1 for h in en_hints if h in f" {t} ") >= 2


def _enforce_answer_language(answer: str, target_lang: str) -> str:
    if not answer:
        return answer
    if target_lang == "en" and _is_likely_dutch(answer):
        translated = _ollama_chat(
            [
                {"role": "system", "content": "Translate to English only. Keep citation markers like [S1] unchanged."},
                {"role": "user", "content": answer},
            ]
        )
        return translated.strip() or answer
    if target_lang == "nl" and _is_likely_english(answer):
        translated = _ollama_chat(
            [
                {"role": "system", "content": "Vertaal naar het Nederlands. Laat bronmarkeringen zoals [S1] ongewijzigd."},
                {"role": "user", "content": answer},
            ]
        )
        return translated.strip() or answer
    return answer


def _clean_compare_answer(answer: str, user_lang: str) -> str:
    lines = [ln.strip() for ln in (answer or "").splitlines() if ln.strip()]
    bad_phrases = (
        "based on the provided context",
        "i will look for",
        "recent conversation",
        "there is no information about",
        "however, since the question mentions",
    )
    cleaned = [ln for ln in lines if not any(bp in ln.lower() for bp in bad_phrases)]
    text = "\n".join(cleaned).strip()
    if not text:
        return _fallback_message(user_lang)
    return text


def _build_citations(docs_scores: List[Tuple[Document, float]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    context_blocks: List[str] = []
    citations: List[Dict[str, Any]] = []
    for idx, (doc, score) in enumerate(docs_scores, start=1):
        src_id = f"S{idx}"
        src = doc.metadata.get("source") or doc.metadata.get("url")
        if not src:
            src = f"url_doc:{doc.metadata.get('doc_id', 'unknown')}"
        url = doc.metadata.get("url")
        text = (doc.page_content or "").strip()
        if not text:
            continue
        context_blocks.append(f"[{src_id}] [source: {src}]\n{text}")
        citations.append(
            {
                "id": src_id,
                "source": src,
                "url": url,
                "score": round(score, 4),
            }
        )
    return context_blocks, citations


def _is_bad_source(source: str) -> bool:
    s = (source or "").strip().lower()
    if s.endswith("all_factsheets.md"):
        return True
    return False


def _filter_docs_quality(docs_scores: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    out: List[Tuple[Document, float]] = []
    for doc, score in docs_scores:
        src = str(doc.metadata.get("source") or doc.metadata.get("url") or "")
        if _is_bad_source(src):
            continue
        if not (doc.page_content or "").strip():
            continue
        out.append((doc, score))
    return out


def _log_metrics(question: str, result: RAGResult) -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "question": question,
            **result.to_dict(),
        }
        with open(METRICS_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Metrics logging must never break the chat flow.
        return


# -------------------------
# Public API
# -------------------------
def ask_storage_question_structured(
    user_input: str,
    chat_history: List[Tuple[str, str]] | List[Dict[str, Any]] | None = None,
    user_language: Optional[str] = None,
    latest_topic: Optional[str] = None,
    skip_clarification: bool = False,
) -> RAGResult:
    question = (user_input or "").strip()
    normalized_history = _normalize_chat_history(chat_history)
    user_lang = user_language or _detect_language(question, normalized_history)
    topic_from_question = _extract_topic_from_text(question)
    effective_question = _augment_followup_question(question, latest_topic)
    if _is_bare_topic_query(question, topic_from_question or latest_topic):
        effective_question = _expand_bare_topic_query(
            question,
            topic_from_question or latest_topic or "",
            user_lang,
        )

    if not question:
        result = RAGResult(
            response=_fallback_message(user_lang),
            citations=[],
            route="empty_input",
            detected_topic=topic_from_question or latest_topic,
        )
        _log_metrics(question, result)
        return result

    if _is_smalltalk(question):
        result = RAGResult(
            response=_smalltalk_response(user_lang),
            citations=[],
            route="smalltalk_ack",
            detected_topic=topic_from_question or latest_topic,
        )
        _log_metrics(question, result)
        return result

    if not skip_clarification and _needs_intake_question(effective_question):
        intake = _intake_question(user_lang)
        result = RAGResult(
            response=intake,
            citations=[],
            needs_clarification=True,
            clarification_question=intake,
            route="clarification_gate",
            detected_topic=topic_from_question or latest_topic,
        )
        _log_metrics(question, result)
        return result

    clarification = None if skip_clarification else _needs_basic_clarification(effective_question, normalized_history)
    if clarification:
        result = RAGResult(
            response=clarification,
            citations=[],
            needs_clarification=True,
            clarification_question=clarification,
            route="clarification_gate",
            detected_topic=topic_from_question or latest_topic,
        )
        _log_metrics(question, result)
        return result

    _ensure_ollama_models_exist()

    retrieval_query = _rewrite_query(effective_question)
    docs_scores = _retrieve_with_scores(retrieval_query)
    reranked = _hybrid_rerank(question, docs_scores, topic=topic_from_question or latest_topic)
    quality_docs = _filter_docs_quality(reranked)
    filtered_docs_scores = [(d, s) for d, s in quality_docs if s >= MIN_DOC_RELEVANCE][:MAX_RERANK_DOCS]

    if DEBUG_RETRIEVAL:
        print(f"DEBUG: retrieved {len(docs_scores)} docs, {len(filtered_docs_scores)} after threshold")
        for i, (d, s) in enumerate(filtered_docs_scores, 1):
            snippet = (d.page_content or "").replace("\n", " ")[:200]
            print(i, "score=", round(s, 4), d.metadata, snippet)

    if len(filtered_docs_scores) < 2:
        fallback = _fallback_message(user_lang)
        clarification = None if skip_clarification else _needs_basic_clarification(effective_question, normalized_history)
        if clarification:
            result = RAGResult(
                response=clarification,
                citations=[],
                needs_clarification=True,
                clarification_question=clarification,
                confidence=0.0,
                route="clarification_after_retrieval",
                used_docs=len(filtered_docs_scores),
                detected_topic=topic_from_question or latest_topic,
            )
            _log_metrics(question, result)
            return result
        result = RAGResult(
            response=fallback,
            citations=[],
            confidence=0.0,
            route="fallback_low_docs",
            used_docs=len(filtered_docs_scores),
            detected_topic=topic_from_question or latest_topic,
        )
        _log_metrics(question, result)
        return result

    confidence = _estimate_confidence(filtered_docs_scores)
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        if topic_from_question and len(filtered_docs_scores) >= 2:
            confidence = LOW_CONFIDENCE_THRESHOLD
        clarification = None if skip_clarification else _needs_basic_clarification(effective_question, normalized_history)
        if clarification:
            result = RAGResult(
                response=clarification,
                citations=[],
                needs_clarification=True,
                clarification_question=clarification,
                confidence=confidence,
                route="clarification_low_confidence",
                used_docs=len(filtered_docs_scores),
                detected_topic=topic_from_question or latest_topic,
            )
            _log_metrics(question, result)
            return result
        result = RAGResult(
            response=_fallback_message(user_lang),
            citations=[],
            confidence=confidence,
            route="fallback_low_confidence",
            used_docs=len(filtered_docs_scores),
            detected_topic=topic_from_question or latest_topic,
        )
        _log_metrics(question, result)
        return result

    context_blocks, citations = _build_citations(filtered_docs_scores)
    prompt = _build_prompt(effective_question, context_blocks, normalized_history, answer_language=user_lang)

    messages = [
        {"role": "system", "content": "Return only the final answer. No meta-text."},
        {"role": "user", "content": prompt},
    ]
    answer = _ollama_chat(messages)
    answer = _dedupe_lines(_strip_meta(answer))
    answer = _enforce_answer_language(answer, user_lang)
    answer = _inject_topic_if_insufficient(answer, topic_from_question or latest_topic, user_lang)
    if _is_compare_question(effective_question):
        answer = _clean_compare_answer(answer, user_lang)
    if not answer:
        answer = _fallback_message(user_lang)

    # If the model forgot citations, add a short reference line.
    if citations and not re.search(r"\[S\d+\]", answer):
        cite_ids = ", ".join(c["id"] for c in citations[:3])
        answer = f"{answer}\n\nSources: {cite_ids}"

    result = RAGResult(
        response=answer,
        citations=citations,
        confidence=confidence,
        route="rag_answer",
        used_docs=len(filtered_docs_scores),
        detected_topic=topic_from_question or latest_topic,
    )
    _log_metrics(question, result)
    return result


def ask_storage_question(
    user_input: str,
    chat_history: List[Tuple[str, str]] | List[Dict[str, Any]] | None = None,
) -> str:
    return ask_storage_question_structured(user_input, chat_history).response
