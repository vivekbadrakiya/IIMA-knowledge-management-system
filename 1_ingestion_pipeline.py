"""
Page-Wise Ingestion Pipeline

Changes from original:
1. NO RecursiveCharacterTextSplitter
2. Each page = 1 chunk
3. Direct page-to-embedding conversion

For your 211-page PDF:
- Before: 478 chunks (1000 chars each)
- After: 211 chunks (1 page each)
"""

import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ---------------------------
# Configuration
# ---------------------------
PDF_DIR = "data"        
EXCEL_DIR = "data/excels"    
PERSIST_DIR = "db/chroma_db"

def ingest_documents():
    """
    Ingest documents with PAGE-WISE chunking
    Each page becomes one chunk (no splitting)
    """
    print("=" * 70)
    print("📚 PAGE-WISE INGESTION PIPELINE")
    print("=" * 70)
    print("Strategy: Each page = 1 chunk (no splitting)")
    print("=" * 70 + "\n")
    
    print("🔹 Step 1: Loading documents...\n")
    
    all_pages = []

    # ---------------------------
    # Load PDFs - Page by Page
    # ---------------------------
    if os.path.exists(PDF_DIR):
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
        
        if pdf_files:
            print(f"📄 Found {len(pdf_files)} PDF file(s)")
            
            for pdf in pdf_files:
                pdf_path = os.path.join(PDF_DIR, pdf)
                
                try:
                    # Load PDF
                    loader = PyPDFLoader(pdf_path)
                    pages = loader.load()
                    
                    # Each page is already a separate Document object
                    # PyPDFLoader gives us 1 Document per page by default
                    for page in pages:
                        # Ensure metadata is set
                        page.metadata['source'] = pdf
                        page.metadata['file_type'] = 'pdf'
                        # page.metadata['page'] is already set by PyPDFLoader
                    
                    all_pages.extend(pages)
                    
                    print(f"  ✅ {pdf}: {len(pages)} pages loaded")
                    
                except Exception as e:
                    print(f"  ❌ Error loading {pdf}: {str(e)}")
                    continue
        else:
            print("⚠️  No PDF files found in", PDF_DIR)
    else:
        print(f"⚠️  PDF directory not found: {PDF_DIR}")

    # ---------------------------
    # Load Excel Files (if any)
    # ---------------------------
    if os.path.exists(EXCEL_DIR):
        excel_files = [
            f for f in os.listdir(EXCEL_DIR)
            if f.endswith(".xlsx") or f.endswith(".xls")
        ]
        
        if excel_files:
            print(f"\n📊 Found {len(excel_files)} Excel file(s)")
            
            for excel in excel_files:
                excel_path = os.path.join(EXCEL_DIR, excel)
                
                try:
                    loader = UnstructuredExcelLoader(excel_path)
                    docs = loader.load()
                    
                    for doc in docs:
                        doc.metadata['source'] = excel
                        doc.metadata['file_type'] = 'excel'
                    
                    all_pages.extend(docs)
                    
                    print(f"  ✅ {excel}: {len(docs)} sections loaded")
                    
                except Exception as e:
                    print(f"  ❌ Error loading {excel}: {str(e)}")
                    continue

    # ---------------------------
    # Validation
    # ---------------------------
    if not all_pages:
        raise FileNotFoundError(
            "❌ No documents found.\n"
            f"Please add PDF files to: {PDF_DIR}/"
        )

    print(f"\n✅ Total pages/documents loaded: {len(all_pages)}")
    
    # ---------------------------
    # NO CHUNKING - Each page is already a chunk!
    # ---------------------------
    print("\n🔹 Step 2: Chunking strategy...")
    print("   Strategy: PAGE-WISE (no splitting)")
    print(f"   Each page = 1 complete chunk")
    print(f"   Total chunks: {len(all_pages)} (same as pages)")
    
    # Calculate page size statistics
    page_sizes = [len(page.page_content) for page in all_pages]
    avg_size = sum(page_sizes) / len(page_sizes)
    min_size = min(page_sizes)
    max_size = max(page_sizes)
    
    print(f"\n   📊 Page Size Statistics:")
    print(f"      • Average: {avg_size:.0f} characters")
    print(f"      • Minimum: {min_size} characters")
    print(f"      • Maximum: {max_size} characters")
    
    # ---------------------------
    # Generate Embeddings
    # ---------------------------
    print("\n🔹 Step 3: Generating embeddings...")
    print("   Model: sentence-transformers/all-MiniLM-L6-v2")
    print("   This may take a few minutes...\n")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("✅ Embedding model loaded")

    # ---------------------------
    # Create Vector Store
    # ---------------------------
    print("\n🔹 Step 4: Creating vector database...")
    
    # Create directory if needed
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    # Create ChromaDB with page-wise chunks
    vector_store = Chroma.from_documents(
        documents=all_pages,  # Each page is a chunk
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    print(f"✅ Vector store created at: {PERSIST_DIR}")
    print(f"   Total vectors: {len(all_pages)}")

    # ---------------------------
    # Summary
    # ---------------------------
    print("\n" + "=" * 70)
    print("📊 INGESTION SUMMARY")
    print("=" * 70)
    print(f"Chunking Strategy:     PAGE-WISE (1 page = 1 chunk)")
    print(f"Total Pages:           {len(all_pages)}")
    print(f"Total Chunks:          {len(all_pages)} (same as pages)")
    print(f"Average Page Size:     {avg_size:.0f} characters")
    print(f"Min Page Size:         {min_size} characters")
    print(f"Max Page Size:         {max_size} characters")
    print(f"Embedding Model:       sentence-transformers/all-MiniLM-L6-v2")
    print(f"Vector Dimensions:     384")
    print(f"Database Location:     {PERSIST_DIR}")
    print("=" * 70)
    
    print("\n✅ Ingestion complete!")
    print("   Next step: python 3_answer_generation.py\n")

if __name__ == "__main__":
    try:
        ingest_documents()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise