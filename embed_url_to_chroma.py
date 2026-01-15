import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Configuration ===
CHROMA_DIR = "chroma_bge_m3"# 💡
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Voeg hier je URL's toe
URLS = [
    "https://www.uu.nl/en/research/research-data-management/guides/storing-and-preserving-data",
    "https://www.uu.nl/en/research/yoda"
    # voeg meer UU-pagina’s toe als je wilt
]


def load_urls(url_list):
    """Haalt webpagina's op en zet ze om naar LangChain-documenten"""
    all_docs = []
    for url in url_list:
        loader = WebBaseLoader(url)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = url
        all_docs.extend(docs)
    return all_docs


def split_docs(docs):
    """Splitst lange teksten in overlappende chunks"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)


def embed_and_store(chunks, persist_dir):
    """Embedt alle chunks en slaat ze op in Chroma met GPU-versnelling"""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"}  # 💥 gebruik GPU voor embeddings
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="uu_web_data"
    )
    vectordb.persist()
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB at '{persist_dir}'")


if __name__ == "__main__":
    print("🌐 Loading URLs...")
    documents = load_urls(URLS)

    print("✂️ Splitting into chunks...")
    chunks = split_docs(documents)

    print("🧠 Embedding and storing in ChromaDB (GPU)...")
    embed_and_store(chunks, CHROMA_DIR)
