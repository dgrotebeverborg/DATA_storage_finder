from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import json, os

# 1. Load JSON
with open("data/storage_data_2.json", "r", encoding="utf-8") as f:
    data = json.load(f)["storage_data_2"]

# 2. Convert each storage solution to a document
docs = []
for s in data:
    text = f"{s['name']}\n"
    for k, v in s.items():
        if k not in ["name", "categories"]:
            text += f"{k}: {v}\n"
    docs.append({"id": s["name"], "text": text})

# 3. Embed and store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(collection_name="storage_data", embedding_function=embeddings, persist_directory="chromadb_storage")

for d in docs:
    db.add_texts([d["text"]], ids=[d["id"]])

db.persist()
print("✅ Embedded", len(docs), "storage profiles into ChromaDB.")
