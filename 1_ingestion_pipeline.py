"""
IMPROVED Ingestion Pipeline

Improvements:
1. Better chunking strategy for improved retrieval
2. More metadata for better context
3. Better overlap to preserve context
4. Progress tracking
"""

import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm

# ---------------------------
# IMPROVED Config
# ---------------------------
PDF_DIR = "data"        
EXCEL_DIR = "data/excels"    
PERSIST_DIR = "db/chroma_db"

# IMPROVED: Better chunking parameters
CHUNK_SIZE = 1000  # Keep at 1000 for good context
CHUNK_OVERLAP = 300  # INCREASED from 100 to 300 for better context preservation

print("=" * 70)
print("📚 IMPROVED INGESTION PIPELINE")
print("=" * 70)
print(f"📁 PDF Directory: {PDF_DIR}")
print(f"📁 Excel Directory: {EXCEL_DIR}")
print(f"💾 Database: {PERSIST_DIR}")
print(f"📏 Chunk Size: {CHUNK_SIZE} characters")
print(f"🔄 Chunk Overlap: {CHUNK_OVERLAP} characters")
print("=" * 70 + "\n")

def ingest_documents():
    print("🔹 Starting ingestion pipeline...\n")

    documents = []

    # ---------------------------
    # Load PDFs with Progress
    # ---------------------------
    if os.path.exists(PDF_DIR):
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
        
        if pdf_files:
            print(f"📄 Found {len(pdf_files)} PDF file(s)")
            
            for pdf in tqdm(pdf_files, desc="Loading PDFs", unit="file"):
                try:
                    pdf_path = os.path.join(PDF_DIR, pdf)
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    
                    # IMPROVED: Add better metadata
                    for doc in docs:
                        doc.metadata['source'] = pdf
                        doc.metadata['file_type'] = 'pdf'
                        # Page number already added by PyPDFLoader
                    
                    documents.extend(docs)
                    print(f"  ✅ {pdf}: {len(docs)} pages")
                    
                except Exception as e:
                    print(f"  ❌ Error loading {pdf}: {str(e)}")
                    continue
        else:
            print("⚠️  No PDF files found in", PDF_DIR)
    else:
        print(f"⚠️  PDF directory not found: {PDF_DIR}")

    # ---------------------------
    # Load Excel files
    # ---------------------------
    if os.path.exists(EXCEL_DIR):
        excel_files = [
            f for f in os.listdir(EXCEL_DIR)
            if f.endswith(".xlsx") or f.endswith(".xls")
        ]
        
        if excel_files:
            print(f"\n📊 Found {len(excel_files)} Excel file(s)")
            
            for excel in tqdm(excel_files, desc="Loading Excel", unit="file"):
                try:
                    excel_path = os.path.join(EXCEL_DIR, excel)
                    loader = UnstructuredExcelLoader(excel_path)
                    docs = loader.load()
                    
                    # IMPROVED: Add metadata
                    for doc in docs:
                        doc.metadata['source'] = excel
                        doc.metadata['file_type'] = 'excel'
                    
                    documents.extend(docs)
                    print(f"  ✅ {excel}: {len(docs)} sections")
                    
                except Exception as e:
                    print(f"  ❌ Error loading {excel}: {str(e)}")
                    continue
        else:
            print(f"⚠️  No Excel files found in {EXCEL_DIR}")
    else:
        print(f"⚠️  Excel directory not found: {EXCEL_DIR}")

    if not documents:
        raise FileNotFoundError(
            "❌ No PDF or Excel documents found.\n"
            f"Please add files to:\n  - {PDF_DIR}/ (for PDFs)\n  - {EXCEL_DIR}/ (for Excel files)"
        )

    print(f"\n✅ Total documents loaded: {len(documents)}")

    # ---------------------------
    # IMPROVED: Text Splitting with Better Strategy
    # ---------------------------
    print("\n🔹 Chunking documents...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],  # IMPROVED: More separators
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = splitter.split_documents(documents)
    
    # IMPROVED: Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['chunk_length'] = len(chunk.page_content)

    print(f"✅ Created {len(chunks)} text chunks")
    
    # Calculate and display chunk statistics
    avg_chunk_size = sum(c.metadata['chunk_length'] for c in chunks) / len(chunks)
    min_chunk_size = min(c.metadata['chunk_length'] for c in chunks)
    max_chunk_size = max(c.metadata['chunk_length'] for c in chunks)
    
    print(f"   📊 Chunk Statistics:")
    print(f"      • Average size: {avg_chunk_size:.0f} characters")
    print(f"      • Min size: {min_chunk_size} characters")
    print(f"      • Max size: {max_chunk_size} characters")

    # ---------------------------
    # IMPROVED: Embeddings with Progress
    # ---------------------------
    print("\n🔹 Generating embeddings (this may take a few minutes)...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("✅ Embedding model loaded")

    # ---------------------------
    # IMPROVED: Vector Store Creation with Progress
    # ---------------------------
    print("\n🔹 Creating vector database...")
    
    # Create directory if it doesn't exist
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    # Create vector store with progress indication
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    print(f"✅ Vector store created and persisted to: {PERSIST_DIR}")
    
    # ---------------------------
    # Summary
    # ---------------------------
    print("\n" + "=" * 70)
    print("📊 INGESTION SUMMARY")
    print("=" * 70)
    print(f"📄 Total Pages/Documents: {len(documents)}")
    print(f"✂️  Total Chunks Created: {len(chunks)}")
    print(f"🔧 Chunk Size: {CHUNK_SIZE} characters")
    print(f"🔄 Chunk Overlap: {CHUNK_OVERLAP} characters")
    print(f"💾 Database Location: {PERSIST_DIR}")
    print(f"📊 Average Chunk Size: {avg_chunk_size:.0f} characters")
    print("=" * 70)
    print("\n✅ Ingestion complete! You can now run the chatbot.")
    print("   Next step: python 3_answer_generation_FIXED.py\n")

if __name__ == "__main__":
    try:
        ingest_documents()
    except Exception as e:
        print(f"\n❌ Error during ingestion: {str(e)}")
        print("\nPlease check:")
        print("1. PDF files are in the 'data/' folder")
        print("2. Files are valid PDF/Excel format")
        print("3. You have enough disk space")
        raise