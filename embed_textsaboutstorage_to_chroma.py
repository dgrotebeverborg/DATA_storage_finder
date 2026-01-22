import os
import glob
from pathlib import Path
from datetime import datetime
import hashlib

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata

from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader

# --- Werk vanuit de map waarin dit script zich bevindt ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# === Config ===
CHROMA_DIR = "chroma_storage"
COLLECTION_NAME = "storage_unified_2026"  # verander naam bij nieuwe embedding-model / strategie

# Kies hier je embedding model (probeer deze volgorde)
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"  # beste multilingual 2025
# Alternatieven:
# EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
# EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # oude fallback

URLS = [
    # --- Core Yoda pages ---
    "https://www.uu.nl/en/research/yoda",
    "https://www.uu.nl/en/research/yoda/about-yoda",
    "https://www.uu.nl/en/research/yoda/for-researchers",
    "https://www.uu.nl/en/research/yoda/faq",

    # --- Data storage & preservation guides ---
    "https://www.uu.nl/en/research/research-data-management/guides/storing-and-preserving-data",
    "https://www.uu.nl/en/research/research-data-management/guides/storing-data-during-your-research",
    "https://www.uu.nl/en/research/research-data-management/guides/archiving-your-data-after-your-project-ends",

    # --- Collaboration & cloud storage ---
    "https://www.uu.nl/en/research/research-data-management/guides/working-together",
    "https://www.uu.nl/en/research/research-data-management/guides/microsoft-teams-and-onedrive",
    "https://www.uu.nl/en/research/research-data-management/guides/surfdrive",

    # --- Privacy, ethics & sensitive data ---
    "https://www.uu.nl/en/research/research-data-management/guides/sensitive-data",
    "https://www.uu.nl/en/research/research-data-management/guides/privacy-and-personal-data",
    "https://www.uu.nl/en/research/research-data-management/guides/legal-and-ethical-aspects",

    # --- FAIR & general RDM ---
    "https://www.uu.nl/en/research/research-data-management/guides/fair-data",
    "https://www.uu.nl/en/research/research-data-management",

    # --- SURF ===
    "https://www.surf.nl/en/surfdrive-safe-and-reliable-cloud-storage",
    "https://www.surf.nl/en/surfdrive",
    "https://www.surf.nl/en/knowledge-base/surfdrive-for-researchers-and-lecturers",
]

PDF_DIR = "pdf"
FACTSHEETS_DIR = "data/factsheets"
FACTSHEETS_GLOB = "*.md"


def get_text_splitter(source_type: str) -> RecursiveCharacterTextSplitter:
    """Type-specifieke splitter met meer overlap en betere separators"""
    if source_type == "factsheet":
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=True,
            add_start_index=True,
        )
    else:  # url + pdf
        return RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=180,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=True,
            add_start_index=True,
        )


def deduplicate_documents(docs: list[Document]) -> list[Document]:
    """Simpele deduplicatie op basis van inhoud (sha256 hash)"""
    seen = set()
    unique_docs = []
    for doc in docs:
        content = (doc.page_content or "").strip()
        if not content:
            continue
        hash_val = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()
        if hash_val not in seen:
            seen.add(hash_val)
            unique_docs.append(doc)
    print(f"Deduplicatie: {len(docs)} → {len(unique_docs)} unieke documenten/chunks")
    return unique_docs


def load_factsheet_docs() -> list[Document]:
    docs: list[Document] = []
    fs_dir = Path(FACTSHEETS_DIR)

    if not fs_dir.exists():
        print(f"⚠️ Factsheets map niet gevonden: {fs_dir.resolve()}")
        return docs

    md_files = sorted(fs_dir.glob(FACTSHEETS_GLOB))
    if not md_files:
        print(f"⚠️ Geen .md factsheets gevonden in: {fs_dir.resolve()}")
        return docs

    splitter = get_text_splitter("factsheet")

    for md_path in md_files:
        try:
            text = md_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue

            first_line = text.split("\n", 1)[0].strip()
            title = first_line.lstrip("# ").strip() if first_line.startswith("#") else md_path.stem

            base_doc = Document(
                page_content=text,
                metadata={
                    "source_type": "factsheet",
                    "source": str(md_path).replace("\\", "/"),
                    "filename": md_path.name,
                    "slug": md_path.stem,
                    "title": title,
                    "language": "nl",
                    "fetch_date": datetime.now().strftime("%Y-%m"),
                }
            )

            chunks = splitter.split_documents([base_doc])
            for chunk in chunks:
                chunk.metadata["chunk_start_index"] = chunk.metadata.get("start_index", 0)

            docs.extend(chunks)
        except Exception as e:
            print(f"⚠️ Fout bij verwerken factsheet {md_path.name}: {e}")

    print(f"📄 Factsheets: {len(md_files)} bestanden → {len(docs)} chunks")
    return docs


def load_url_docs() -> list[Document]:
    docs: list[Document] = []
    try:
        loader = UnstructuredURLLoader(urls=URLS, mode="elements", strategy="auto")
        raw_docs = loader.load()

        splitter = get_text_splitter("url")
        split_docs = splitter.split_documents(raw_docs)

        for d in split_docs:
            d.metadata.update({
                "source_type": "url",
                "url": d.metadata.get("source", ""),
                "title": d.metadata.get("title", "").strip() or "Untitled page",
                "language": "nl",  # bijna alles is Nederlands/Engels
                "fetch_date": datetime.now().strftime("%Y-%m"),
            })
            if "start_index" in d.metadata:
                d.metadata["chunk_start_index"] = d.metadata["start_index"]

        docs.extend(split_docs)
        print(f"🌐 URLs: {len(URLS)} pagina's → {len(docs)} chunks")
    except Exception as e:
        print(f"⚠️ Fout bij laden URLs: {e}")
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
                    "language": "nl",
                    "fetch_date": datetime.now().strftime("%Y-%m"),
                })

            split_pages = splitter.split_documents(pdf_pages)
            docs.extend(split_pages)
        except Exception as e:
            print(f"⚠️ Fout bij PDF {pdf_file}: {e}")

    print(f"📚 PDFs: {len(pdf_files)} bestanden → {len(docs)} chunks")
    return docs


if __name__ == "__main__":
    print(f"Start embedding pipeline – model: {EMBEDDING_MODEL_NAME}")
    print(f"Huidige datum: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

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

    # 4) Embedden
    # Voor de embedding stap
    print(f"Embedden met {EMBEDDING_MODEL_NAME} ... (dit kan even duren)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ← Dit is de cruciale toevoeging
    filtered_docs = filter_complex_metadata(all_docs)

    print(f"Na metadata filtering: {len(all_docs)} → {len(filtered_docs)} documenten")

    vectordb = Chroma.from_documents(
        documents=filtered_docs,           # gebruik de gefilterde versie
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    print(f"\n✅ Klaar! Collectie '{COLLECTION_NAME}' aangemaakt met {len(all_docs)} chunks")
    print(f"   Opslaglocatie: {Path(CHROMA_DIR).resolve()}")
    print("   Je kunt nu retrieval testen in storage_rag.py")