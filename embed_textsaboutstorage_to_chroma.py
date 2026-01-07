import os
import glob
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader


# --- Werk vanuit de map waarin dit script zich bevindt ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# === Config ===
CHROMA_DIR = "chroma_storage"
COLLECTION_NAME = "storage_unified"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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

    # --- Collaboration & cloud storage (Teams, OneDrive, SURFdrive) ---
    "https://www.uu.nl/en/research/research-data-management/guides/working-together",
    "https://www.uu.nl/en/research/research-data-management/guides/microsoft-teams-and-onedrive",
    "https://www.uu.nl/en/research/research-data-management/guides/surfdrive",

    # --- Privacy, ethics & sensitive data ---
    "https://www.uu.nl/en/research/research-data-management/guides/sensitive-data",
    "https://www.uu.nl/en/research/research-data-management/guides/privacy-and-personal-data",
    "https://www.uu.nl/en/research/research-data-management/guides/legal-and-ethical-aspects",

    # --- FAIR & general RDM background ---
    "https://www.uu.nl/en/research/research-data-management/guides/fair-data",
    "https://www.uu.nl/en/research/research-data-management",

    # --- SURF stuff ---
    "https://www.surf.nl/en/surfdrive-safe-and-reliable-cloud-storage",
    "https://www.surf.nl/en/surfdrive",
    "https://www.surf.nl/en/knowledge-base/surfdrive-for-researchers-and-lecturers",
]

PDF_DIR = "pdf"
FACTSHEETS_DIR = "data/factsheets"   # <-- jouw nieuwe output map met .md
FACTSHEETS_GLOB = "*.md"


def load_factsheet_docs() -> list[Document]:
    """
    Laadt factsheets/*.md en splitst ze in grotere chunks.
    Factsheets zijn al narratief, dus grotere chunk_size werkt beter.
    """
    docs: list[Document] = []
    fs_dir = Path(FACTSHEETS_DIR)

    if not fs_dir.exists():
        print(f"⚠️ Factsheets map niet gevonden: {fs_dir.resolve()}")
        return docs

    md_files = sorted(fs_dir.glob(FACTSHEETS_GLOB))
    if not md_files:
        print(f"⚠️ Geen factsheets gevonden in: {fs_dir.resolve()}")
        return docs

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        # Maak 1 document, split daarna
        base_doc = Document(
            page_content=text,
            metadata={
                "source_type": "factsheet",
                "source": str(md_path).replace("\\", "/"),
                "solution_name": md_path.stem,  # slug; als je liever titel wilt: parse de eerste "# ..."
            },
        )

        chunks = splitter.split_documents([base_doc])
        docs.extend(chunks)

    print(f"📄 Loaded and split {len(docs)} factsheet chunks from {len(md_files)} factsheets")
    return docs


def load_url_docs() -> list[Document]:
    """Laadt teksten van webpagina's en splitst in kleine chunks."""
    docs: list[Document] = []
    try:
        loader = UnstructuredURLLoader(urls=URLS)
        raw_docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60)
        docs = splitter.split_documents(raw_docs)

        for d in docs:
            # Laat de oorspronkelijke url (als aanwezig) staan; voeg alleen type toe
            d.metadata["source_type"] = "url"
            if "source" not in d.metadata:
                d.metadata["source"] = "url"
        print(f"🌐 Loaded and split {len(docs)} URL chunks")
    except Exception as e:
        print("⚠️ Fout bij laden van URLs:", e)
    return docs


def load_pdf_docs() -> list[Document]:
    """Laadt alle PDF's uit pdf/."""
    docs: list[Document] = []
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            pdf_docs = loader.load()
            for d in pdf_docs:
                d.metadata["source_type"] = "pdf"
                d.metadata["source"] = os.path.basename(pdf_file)
            docs.extend(pdf_docs)
        except Exception as e:
            print(f"⚠️ Fout bij laden van {pdf_file}: {e}")
    print(f"📚 Loaded {len(docs)} pages from {len(pdf_files)} PDFs")
    return docs


if __name__ == "__main__":
    # 1) Oude Chroma-store opruimen
    if os.path.exists(CHROMA_DIR):
        print("🧹 Removing old Chroma store...")
        import shutil
        shutil.rmtree(CHROMA_DIR)

    # 2) Combineer bronnen (factsheets eerst: die wil je het meest)
    all_docs: list[Document] = []
    all_docs.extend(load_factsheet_docs())
    all_docs.extend(load_url_docs())
    all_docs.extend(load_pdf_docs())

    print(f"🧾 Total combined documents: {len(all_docs)}")

    if not all_docs:
        print("⚠️ Geen documenten gevonden — stop.")
        raise SystemExit(1)

    # 3) Embed (GPU)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
    )

    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    print(f"✅ Unified Chroma collection built with {len(all_docs)} docs at '{CHROMA_DIR}'")
