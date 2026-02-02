import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "db/chroma_db"

def initialize_retriever():
    print("🔹 Initializing retrieval pipeline...")

    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError("❌ Chroma DB not found. Run ingestion pipeline first.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 5}
    )

    print("✅ Retrieval pipeline ready.")
    print("📌 Vector store loaded successfully.")

    return retriever


if __name__ == "__main__":
    initialize_retriever()