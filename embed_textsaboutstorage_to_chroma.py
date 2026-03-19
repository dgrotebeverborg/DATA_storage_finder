import os
import glob
from pathlib import Path
from datetime import datetime
import hashlib
import requests
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader

# --- Werk vanuit de map waarin dit script zich bevindt ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# === Config ===
CHROMA_DIR = "chroma_storage"
COLLECTION_NAME = "storage_unified_2026_ollama"  # nieuwe naam ivm nieuwe embedding provider
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

URLS = [
    "https://www.uu.nl/en/research/yoda",
    "https://www.uu.nl/en/research/yoda/about-yoda",
    "https://www.uu.nl/en/research/yoda/for-researchers",
    "https://www.uu.nl/en/research/yoda/faq",
    "https://www.uu.nl/en/research/research-data-management/guides/storing-and-preserving-data",
    "https://www.uu.nl/en/research/research-data-management/guides/storing-data-during-your-research",
    "https://www.uu.nl/en/research/research-data-management/guides/archiving-your-data-after-your-project-ends",
    "https://www.uu.nl/en/research/research-data-management/guides/working-together",
    "https://www.uu.nl/en/research/research-data-management/guides/microsoft-teams-and-onedrive",
    "https://www.uu.nl/en/research/research-data-management/guides/surfdrive",
    "https://www.uu.nl/en/research/research-data-management/guides/sensitive-data",
    "https://www.uu.nl/en/research/research-data-management/guides/privacy-and-personal-data",
    "https://www.uu.nl/en/research/research-data-management/guides/legal-and-ethical-aspects",
    "https://www.uu.nl/en/research/research-data-management/guides/fair-data",
    "https://www.uu.nl/en/research/research-data-management",
    "https://www.surf.nl/en/surfdrive-safe-and-reliable-cloud-storage",
    "https://www.surf.nl/en/surfdrive",
    "https://www.surf.nl/en/knowledge-base/surfdrive-for-researchers-and-lecturers",
]

PDF_DIR = "pdf"
FACTSHEETS_DIR = "data/factsheets"
FACTSHEETS_GLOB = "*.md"
NOW_YYYY_MM = datetime.now().strftime("%Y-%m")
EXCLUDED_FACTSHEETS = {"all_factsheets.md"}

SOURCE_TRUST = {
    "factsheet": "high",
    "pdf": "high",
    "url": "medium",
}


class OllamaEmbeddings:
    """Minimale embedding adapter voor LangChain/Chroma via Ollama /api/embeddings."""
    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for t in texts:
            vectors.append(self._embed_one(t))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        r = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data["embedding"]


def get_text_splitter(source_type: str) -> RecursiveCharacterTextSplitter:
    if source_type == "factsheet":
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=True,
            add_start_index=True,
        )
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=180,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=True,
            add_start_index=True,
        )


def deduplicate_documents(docs: list[Document]) -> list[Document]:
    seen = set()
    unique_docs = []
    for doc in docs:
        content = (doc.page_content or "").strip()
        if not content:
            continue
        src = str(doc.metadata.get("source") or doc.metadata.get("url") or "unknown")
        hash_input = f"{src}\n{content}".encode("utf-8", errors="ignore")
        hash_val = hashlib.sha256(hash_input).hexdigest()
        if hash_val not in seen:
            seen.add(hash_val)
            unique_docs.append(doc)
    print(f"Deduplicatie: {len(docs)} → {len(unique_docs)} unieke documenten/chunks")
    return unique_docs


def split_markdown_sections(text: str) -> List[Tuple[str, str]]:
    """
    Splits markdown in semantic sections using '### Heading' boundaries.
    Returns [(section_name, section_text)].
    """
    lines = text.splitlines()
    sections: List[Tuple[str, str]] = []
    current_title = "Overview"
    current_lines: List[str] = []

    for line in lines:
        if line.startswith("### "):
            if current_lines:
                block = "\n".join(current_lines).strip()
                if block:
                    sections.append((current_title, block))
            current_title = line.lstrip("#").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        block = "\n".join(current_lines).strip()
        if block:
            sections.append((current_title, block))
    return sections


def chunk_section_text(section_text: str, source_type: str) -> List[str]:
    splitter = get_text_splitter(source_type)
    chunks = splitter.split_text(section_text)
    return [c.strip() for c in chunks if c and c.strip()]


def extract_solution_name(markdown_text: str, fallback: str) -> str:
    first_line = markdown_text.split("\n", 1)[0].strip()
    if first_line.startswith("#"):
        name = first_line.lstrip("#").strip()
        if name:
            return name
    return fallback


def infer_language_from_url(url: str) -> str:
    if "/en/" in url:
        return "en"
    return "nl"


def build_doc_id(source: str, chunk_text: str, chunk_index: int) -> str:
    raw = f"{source}|{chunk_index}|{chunk_text[:140]}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()


def load_factsheet_docs() -> list[Document]:
    docs: list[Document] = []
    fs_dir = Path(FACTSHEETS_DIR)

    if not fs_dir.exists():
        print(f"⚠️ Factsheets map niet gevonden: {fs_dir.resolve()}")
        return docs

    md_files = sorted(fs_dir.glob(FACTSHEETS_GLOB))
    md_files = [p for p in md_files if p.name not in EXCLUDED_FACTSHEETS]
    if not md_files:
        print(f"⚠️ Geen .md factsheets gevonden in: {fs_dir.resolve()}")
        return docs

    for md_path in md_files:
        try:
            text = md_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue

            title = extract_solution_name(text, md_path.stem)
            source_path = str(md_path).replace("\\", "/")
            sections = split_markdown_sections(text)
            chunk_counter = 0

            for section_name, section_text in sections:
                section_chunks = chunk_section_text(section_text, "factsheet")
                for local_idx, chunk_text in enumerate(section_chunks):
                    chunk_counter += 1
                    docs.append(
                        Document(
                            page_content=chunk_text,
                            metadata={
                                "doc_id": build_doc_id(source_path, chunk_text, chunk_counter),
                                "source_type": "factsheet",
                                "source": source_path,
                                "filename": md_path.name,
                                "slug": md_path.stem,
                                "title": title,
                                "section": section_name,
                                "section_chunk_index": local_idx,
                                "language": "en",
                                "fetch_date": NOW_YYYY_MM,
                                "source_trust": SOURCE_TRUST["factsheet"],
                            },
                        )
                    )
        except Exception as e:
            print(f"⚠️ Fout bij verwerken factsheet {md_path.name}: {e}")

    print(f"📄 Factsheets: {len(md_files)} bestanden → {len(docs)} chunks")
    return docs


def load_url_docs() -> list[Document]:
    docs: list[Document] = []
    splitter = get_text_splitter("url")
    loaded_pages = 0

    for url in URLS:
        try:
            # Load per URL so every chunk can always be tied back to its exact source.
            loader = UnstructuredURLLoader(urls=[url], mode="elements", strategy="auto")
            raw_docs = loader.load()
            if not raw_docs:
                continue

            # Guarantee non-empty URL metadata before splitting.
            for d in raw_docs:
                d.metadata["source"] = (d.metadata.get("source") or url).strip() or url
                d.metadata["url"] = (d.metadata.get("url") or url).strip() or url

            split_docs = splitter.split_documents(raw_docs)
            for i, d in enumerate(split_docs):
                source_url = (d.metadata.get("url") or d.metadata.get("source") or url).strip() or url
                text = (d.page_content or "").strip()
                d.metadata.update({
                    "doc_id": build_doc_id(source_url, text, i),
                    "source_type": "url",
                    "source": source_url,
                    "url": source_url,
                    "title": d.metadata.get("title", "").strip() or "Untitled page",
                    "language": infer_language_from_url(source_url),
                    "fetch_date": NOW_YYYY_MM,
                    "source_trust": SOURCE_TRUST["url"],
                })
                if "start_index" in d.metadata:
                    d.metadata["chunk_start_index"] = d.metadata["start_index"]

            docs.extend(split_docs)
            loaded_pages += 1
        except Exception as e:
            print(f"⚠️ Fout bij laden URL {url}: {e}")

    print(f"🌐 URLs: {loaded_pages}/{len(URLS)} pagina's geladen → {len(docs)} chunks")
    return docs


def load_pdf_docs() -> list[Document]:
    docs: list[Document] = []
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    splitter = get_text_splitter("pdf")

    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            pdf_pages = loader.load()

            for page in pdf_pages:
                page.metadata.update({
                    "source_type": "pdf",
                    "source": os.path.basename(pdf_file),
                    "filename": os.path.basename(pdf_file),
                    "page_number": page.metadata.get("page", 0) + 1,
                    "language": "en",
                    "fetch_date": NOW_YYYY_MM,
                    "source_trust": SOURCE_TRUST["pdf"],
                })

            split_pages = splitter.split_documents(pdf_pages)
            for i, page in enumerate(split_pages):
                source = str(page.metadata.get("source") or os.path.basename(pdf_file))
                text = (page.page_content or "").strip()
                page.metadata["doc_id"] = build_doc_id(source, text, i)
            docs.extend(split_pages)
        except Exception as e:
            print(f"⚠️ Fout bij PDF {pdf_file}: {e}")

    print(f"📚 PDFs: {len(pdf_files)} bestanden → {len(docs)} chunks")
    return docs


def ensure_ollama_ready():
    # simpele sanity check
    r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=30)
    r.raise_for_status()


if __name__ == "__main__":
    print(f"Start embedding pipeline – Ollama embed model: {OLLAMA_EMBED_MODEL}")
    print(f"Ollama base URL: {OLLAMA_BASE_URL}")
    print(f"Huidige datum: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    ensure_ollama_ready()

    # 1) Oude store opruimen
    if os.path.exists(CHROMA_DIR):
        print("🧹 Verwijder oude Chroma store...")
        import shutil
        shutil.rmtree(CHROMA_DIR)

    # 2) Alle documenten laden
    all_docs = []
    all_docs.extend(load_factsheet_docs())
    all_docs.extend(load_url_docs())
    all_docs.extend(load_pdf_docs())

    print(f"\nTotaal ruwe chunks vóór dedup: {len(all_docs)}")

    # 3) Dedupliceren
    all_docs = deduplicate_documents(all_docs)

    if not all_docs:
        print("❌ Geen documenten gevonden — stop.")
        raise SystemExit(1)

    # 4) Embedden via Ollama
    print(f"Embedden via Ollama ({OLLAMA_EMBED_MODEL}) ...")
    embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBED_MODEL)

    filtered_docs = filter_complex_metadata(all_docs)
    print(f"Na metadata filtering: {len(all_docs)} → {len(filtered_docs)} documenten")

    vectordb = Chroma.from_documents(
        documents=filtered_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    print(f"\n✅ Klaar! Collectie '{COLLECTION_NAME}' aangemaakt met {len(all_docs)} chunks")
    print(f"   Opslaglocatie: {Path(CHROMA_DIR).resolve()}")
