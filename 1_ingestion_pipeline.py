import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredExcelLoader   
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ---------------------------
# Paths (CHANGED)
# ---------------------------
PDF_DIR = "data"        
EXCEL_DIR = "data/excels"    
PERSIST_DIR = "db/chroma_db"

def ingest_documents():
    print("🔹 Starting ingestion pipeline...")

    documents = []

    # ---------------------------
    # Load PDFs (OLD + adjusted)
    # ---------------------------
    if os.path.exists(PDF_DIR):
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
        for pdf in pdf_files:
            loader = PyPDFLoader(os.path.join(PDF_DIR, pdf))
            documents.extend(loader.load())
    else:
        print("⚠️ PDF directory not found, skipping PDFs")

    # ---------------------------
    # Load Excel files (NEW)
    # ---------------------------
    if os.path.exists(EXCEL_DIR):
        excel_files = [
            f for f in os.listdir(EXCEL_DIR)
            if f.endswith(".xlsx") or f.endswith(".xls")
        ]
        for excel in excel_files:
            loader = UnstructuredExcelLoader(os.path.join(EXCEL_DIR, excel))
            documents.extend(loader.load())
    else:
        print("⚠️ Excel directory not found, skipping Excels")

    if not documents:
        raise FileNotFoundError("❌ No PDF or Excel documents found")

    print(f"✅ Loaded {len(documents)} documents/pages")

    # ---------------------------
    # Text splitting (UNCHANGED)
    # ---------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    print(f"✅ Created {len(chunks)} text chunks")

    # ---------------------------
    # Embeddings (UNCHANGED)
    # ---------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---------------------------
    # Vector store (UNCHANGED)
    # ---------------------------
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    print("✅ Ingestion complete. ChromaDB created.")

if __name__ == "__main__":
    ingest_documents()
