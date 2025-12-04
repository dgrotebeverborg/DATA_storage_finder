# logic/rag.py
import os
import re
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatLlamaCpp  # ⬅️ chat-wrapper gebruiken
from langchain.prompts import ChatPromptTemplate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_storage")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLAMA_MODEL_PATH = "/home/dgrotebeve/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
COLLECTION_NAME = "storage_unified"

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert assistant for research data management at Utrecht University. "
     "Answer ONLY using facts present inside <context>…</context>. "
     "If the context truly contains no relevant information, output exactly:\n"
     "I don’t know based on my available sources.\n"
     "Rules:\n"
     "- Output MUST be exactly one <answer>…</answer> block.\n"
     "- Do not repeat or restate the question.\n"
     "- Do not include labels like 'Question:', 'Answer:', 'Note:', 'Human:', or any extra text.\n"
     "- No greetings or preambles.\n"
     "- 2–4 sentences max."
     ),
    ("human",
     "<context>\n{context}\n</context>\n"
     "<question>{question}</question>\n"
     "Return only:\n<answer>…your answer…</answer>")
])

STOP_TOKENS: List[str] = [
    "</answer>", "Question:", "Answer:", "Note:", "Human:",
    "<question>", "</question>", "<context>", "</context>",
    "<|eot_id|>"  # Llama-3 end-of-turn
]

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_LABEL_CLEAN_RE = re.compile(r"^(?:\s*(?:Question|Answer|Note|Human)\s*:\s*)+", re.IGNORECASE | re.MULTILINE)

_singleton = None

def _init():
    global _singleton
    if _singleton is not None:
        return _singleton

    print("🧠 Initializing storage RAG…")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"}
    )

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    # ✅ Gebruik ChatLlamaCpp met stops en lage temperatuur
    llm = ChatLlamaCpp(
        model_path=LLAMA_MODEL_PATH,
        temperature=0.0,
        top_p=0.95,
        max_tokens=512,
        n_ctx=32768,
        n_gpu_layers=-1,
        n_batch=512,
        repeat_penalty=1.1,
        stop=STOP_TOKENS,
        verbose=False,
        # chat_format wordt automatisch uit het model gehaald (llama-3.*)
    )

    _singleton = {"llm": llm, "vectordb": vectordb}
    print("✅ RAG ready.")
    return _singleton


def _extract_answer(text: str) -> str:
    if not text:
        return ""
    if "</answer>" in text:
        text = text.split("</answer>", 1)[0] + "</answer>"
    m = _ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).strip()
    return _LABEL_CLEAN_RE.sub("", text).strip()


def _is_grounded(context: str, answer: str) -> bool:
    if not context.strip():
        return False
    ctx_lower = context.lower()
    sentences = re.split(r"[.!?]\s+", answer)
    for s in sentences:
        tokens = [w.lower() for w in re.findall(r"\b\w+\b", s) if len(w) >= 4]
        if sum(1 for w in tokens if w in ctx_lower) >= 2:
            return True
    return False


def ask_storage_question(user_input: str, chat_history=None, debug: bool = True) -> str:
    chain = _init()

    # 🔎 Iets strengere retrieval (MMR + hogere threshold) om generieke adviestekst te vermijden
    retriever = chain["vectordb"].as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.4,           # diverser
            "score_threshold": 0.30        # iets strenger
        },
    )

    docs = retriever.get_relevant_documents(user_input)
    context_text = "\n\n".join([d.page_content for d in docs]) if docs else ""

    if debug:
        if docs:
            print("\n🔍 Retrieved context documents:")
            for i, doc in enumerate(docs, 1):
                meta = getattr(doc, "metadata", {})
                src = meta.get("source", "unknown")
                snippet = (doc.page_content or "")[:200].replace("\n", " ")
                print(f"  {i}. {src} → {snippet}...")
        else:
            print("⚠️ No relevant documents retrieved.")

    if not context_text.strip():
        return "I don’t know based on my available sources."

    # ✅ Geef de messages rechtstreeks aan het chatmodel (niet samenplakken!)
    messages = QA_PROMPT.format_messages(context=context_text, question=user_input)
    ai_msg = chain["llm"].invoke(messages)     # ChatLlamaCpp → AIMessage
    raw_text = ai_msg.content or ""

    answer = _extract_answer(raw_text).strip()

    if not answer or not _is_grounded(context_text, answer):
        return "I don’t know based on my available sources."

    # Trim tot max 4 zinnen
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    if len(sentences) > 4:
        answer = " ".join(sentences[:4]).strip()

    return answer
