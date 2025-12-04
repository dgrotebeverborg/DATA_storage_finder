from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import json, os

# Config
CHROMA_DIR = "chroma_storage"  # 💡 zelfde map als in rag.py
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 1️⃣ Load JSON
with open("data/storage_data_2.json", "r", encoding="utf-8") as f:
    data = json.load(f)["storage_data_2"]

# 2️⃣ Convert each storage solution to a Document
docs = []
for s in data:
    text = f"{s['name']}\n"
    for k, v in s.items():
        if k not in ["name", "categories"]:
            text += f"{k}: {v}\n"
    docs.append(Document(page_content=text, metadata={"id": s["name"]}))

print(f"📄 Loaded {len(docs)} storage entries from JSON")

# 3️⃣ GPU embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cuda"}  # 💥 run embeddings op GPU
)

# 4️⃣ Build Chroma store
db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="storage_data",
    persist_directory=CHROMA_DIR
)

db.persist()
print(f"✅ Embedded {len(docs)} storage profiles into ChromaDB (GPU).")
