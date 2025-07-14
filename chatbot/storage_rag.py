# logic/rag.py
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
import os

# Config
CHROMA_DIR = "chroma_storage"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLAMA_MODEL_PATH = "models/llama3/llama-pro-8b-instruct.Q4_K_M.gguf"

# Singleton pattern: laad alles pas als het nodig is
qa_chain = None

def get_qa_chain():
    global qa_chain
    if qa_chain is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        llm = LlamaCpp(
            model_path=LLAMA_MODEL_PATH,
            temperature=0.3,
            max_tokens=512,
            top_p=0.95,
            n_ctx=8192,
            verbose=False
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )
    return qa_chain

def ask_storage_question(user_input, chat_history):
    """
    Stelt een vraag aan de chatbot met behulp van de sessiegeschiedenis.
    """
    chain = get_qa_chain()
    result = chain.invoke({
        "question": user_input,
        "chat_history": chat_history
    })
    return result["answer"]
