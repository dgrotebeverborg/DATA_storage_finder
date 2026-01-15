# rebuild_chroma.py
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
import glob

CHROMA_DIR = "chroma_bge_m3"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

vectordb = Chroma(persist_directory="chroma_storage")
vectordb.delete_collection()  # verwijdert alle opgeslagen items
vectordb.persist()
