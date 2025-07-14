import os
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# === Configuration ===
PDF_DIR = "pdf"
CHROMA_DIR = "chroma_storage"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, good

def load_pdfs(pdf_folder):
    all_docs = []
    pdf_paths = Path(pdf_folder).rglob("*.pdf")
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = str(path.name)
        all_docs.extend(docs)
    return all_docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def embed_and_store(chunks, persist_dir):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB at {persist_dir}")

if __name__ == "__main__":
    print("📥 Loading PDFs...")
    documents = load_pdfs(PDF_DIR)

    print("✂️ Splitting into chunks...")
    chunks = split_docs(documents)

    print("🧠 Embedding and storing in ChromaDB...")
    embed_and_store(chunks, CHROMA_DIR)
