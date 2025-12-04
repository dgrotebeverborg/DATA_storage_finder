import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# === Pad fix ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_storage")

# === Embeddings ===
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"}
)

# === Chroma ===
db = Chroma(
    persist_directory=CHROMA_DIR,
    collection_name="storage_unified",
    embedding_function=emb
)

print("Aantal documenten:", db._collection.count())

# === Test query ===
results = db.similarity_search("Yoda data storage", k=3)
for i, r in enumerate(results, 1):
    print(f"\nResult {i}:\n{r.page_content[:300]}\nSource:", r.metadata.get("source"))
