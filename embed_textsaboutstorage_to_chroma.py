import os
import json
import glob
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader

# --- Werken vanuit de map waarin dit script zich bevindt ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# === Configuratie ===
CHROMA_DIR = "chroma_storage"
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

    # SURF stuff
    "https://www.surf.nl/en/surfdrive-safe-and-reliable-cloud-storage",
    "https://www.surf.nl/en/surfdrive",
    "https://www.surf.nl/en/knowledge-base/surfdrive-for-researchers-and-lecturers",

]


PDF_DIR = "pdf"  # map met lokale PDF's
JSON_PATH = "data/storage_data_2.json"


def load_json_docs():
    """Laadt de JSON en zet om naar Documenten."""
    docs = []
    if not os.path.exists(JSON_PATH):
        print("⚠️ JSON-bestand niet gevonden:", JSON_PATH)
        return docs

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        data = data.get("storage_data_2", data)

    for s in data:
        text = f"{s.get('name','')}\n"
        for k, v in s.items():
            if k not in ["name", "categories"]:
                text += f"{k}: {v}\n"
        docs.append(Document(page_content=text, metadata={"source": "json"}))

    print(f" Loaded {len(docs)} JSON entries")
    return docs


def load_url_docs():
    """Laadt teksten van UU-webpagina's."""
    docs = []
    try:
        loader = UnstructuredURLLoader(urls=URLS)
        raw_docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        docs = splitter.split_documents(raw_docs)
        for d in docs:
            d.metadata["source"] = "url"
        print(f" Loaded and split {len(docs)} URL chunks")
    except Exception as e:
        print("⚠️ Fout bij laden van URLs:", e)
    return docs


def load_pdf_docs():
    """Laadt alle PDF's uit de map pdf/."""
    docs = []
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            pdf_docs = loader.load()
            for d in pdf_docs:
                d.metadata["source"] = os.path.basename(pdf_file)
            docs.extend(pdf_docs)
        except Exception as e:
            print(f"⚠️ Fout bij laden van {pdf_file}: {e}")
    print(f" Loaded {len(docs)} pages from {len(pdf_files)} PDFs")
    return docs


if __name__ == "__main__":
    # 1️⃣  Oude Chroma-store opruimen
    if os.path.exists(CHROMA_DIR):
        print("粒 Removing old Chroma store...")
        import shutil
        shutil.rmtree(CHROMA_DIR)

    # 2️⃣  Combineer alle bronnen
    all_docs = []
    all_docs.extend(load_json_docs())
    all_docs.extend(load_url_docs())
    all_docs.extend(load_pdf_docs())

    print(f"茶 Total combined documents: {len(all_docs)}")

    if not all_docs:
        print("⚠️ Geen documenten gevonden — stop.")
        exit()

    # 3️⃣  Embed alles op GPU
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"}
    )

    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="storage_unified"
    )

    print(f"✅ Unified Chroma collection built with {len(all_docs)} docs at '{CHROMA_DIR}'")
