import json
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# === Config ===
JSON_PATH = "data/storage_data.json"
CHROMA_DIR = "chroma_storage"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# === Load JSON ===
with open(JSON_PATH, "r") as f:
    data = json.load(f)

docs = []

# === Extract and convert to LangChain Documents ===
for category, entries in data.items():
    if isinstance(entries, list):
        for item in entries:
            name = item.get("name", "Unnamed")
            # Combine all key-value pairs into a flat string
            content = f"Category: {category}\nStorage Name: {name}\n" + "\n".join([
                f"{k}: {v}" for k, v in item.items() if v and isinstance(v, str)
            ])
            docs.append(Document(page_content=content, metadata={"source": "storage_data.json", "name": name}))

print(f"✅ Converted {len(docs)} entries to documents")

# === Load and add to ChromaDB ===
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

vectordb.add_documents(docs)
vectordb.persist()

print("✅ Storage solutions embedded and added to ChromaDB.")
