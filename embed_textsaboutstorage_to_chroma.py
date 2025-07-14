import json
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# === CONFIG ===
CHROMA_COLLECTION_NAME = "storage_embeddings"
JSON_PATH = "pdf/storage_profiles.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # or your preferred one

# === LOAD JSON ===
with open(JSON_PATH, "r") as f:
    profiles = json.load(f)

texts_to_embed = []
metadatas = []
ids = []

for i, profile in enumerate(profiles):
    text = profile.get("profile_text", "")
    if not text.strip():
        continue
    texts_to_embed.append(text)
    metadatas.append({
        "name": profile.get("name", f"storage_{i}"),
        "category": profile.get("category", ""),
        "source": "storage_profiles"
    })
    ids.append(f"storage_{i}")



# === EMBED TEXTS ===
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(texts_to_embed).tolist()

# === SAVE TO CHROMA ===
client = Client(Settings())


collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)
results = collection.get()
existing_ids = results.get("ids", [])
if existing_ids:
    collection.delete(ids=existing_ids)



collection.add(
    documents=texts_to_embed,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print(f"✅ Added {len(texts_to_embed)} storage profiles to ChromaDB collection '{CHROMA_COLLECTION_NAME}'.")
