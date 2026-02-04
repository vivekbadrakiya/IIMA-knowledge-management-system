"""
FINAL INGESTION - Rock Solid, Simple, Optimized

Hardware: 8GB RAM, 4GB GPU, Intel i5-10300H
Purpose: Load documents → Create searchable database
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd

# Paths
PDF_DIR = "data"
EXCEL_DIR = "data"
PERSIST_DIR = "db/chroma_db"

# Best model for 8GB RAM: Fast, small (80MB), good quality
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_pdf_documents():
    """Load PDFs page-by-page"""
    documents = []
    
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR, exist_ok=True)
        return documents
    
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        return documents
    
    print(f" Loading {len(pdf_files)} PDF file(s)...")
    
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(os.path.join(PDF_DIR, pdf_file))
            pages = loader.load()
            
            for page in pages:
                page.page_content = " ".join(page.page_content.split())
                page.metadata['source'] = pdf_file
            
            documents.extend(pages)
            print(f"    {pdf_file}: {len(pages)} pages")
        except Exception as e:
            print(f"    {pdf_file}: {str(e)}")
    
    return documents

def load_excel_documents():
    """Load Excel files row-by-row"""
    documents = []
    
    if not os.path.exists(EXCEL_DIR):
        os.makedirs(EXCEL_DIR, exist_ok=True)
        return documents
    
    excel_files = [f for f in os.listdir(EXCEL_DIR) 
                   if f.endswith(('.xlsx', '.xls', '.csv'))]
    if not excel_files:
        return documents
    
    print(f"\n Loading {len(excel_files)} Excel file(s)...")
    
    for excel_file in excel_files:
        try:
            path = os.path.join(EXCEL_DIR, excel_file)
            df = pd.read_csv(path) if excel_file.endswith('.csv') else pd.read_excel(path)
            
            for idx, row in df.iterrows():
                text = " | ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
                
                doc = Document(
                    page_content=text,
                    metadata={'source': excel_file, 'row': idx + 2}
                )
                documents.append(doc)
            
            print(f"    {excel_file}: {len(df)} rows")
        except Exception as e:
            print(f"    {excel_file}: {str(e)}")
    
    return documents

def create_database():
    """Build vector database"""
    print("\n" + "=" * 60)
    print("CREATING VECTOR DATABASE")
    print("=" * 60 + "\n")
    
    # Load documents
    pdf_docs = load_pdf_documents()
    excel_docs = load_excel_documents()
    all_docs = pdf_docs + excel_docs
    
    if not all_docs:
        print(" No documents found!\n")
        print(f"Add files to:")
        print(f"   {PDF_DIR}/ (for PDFs)")
        print(f"   {EXCEL_DIR}/ (for Excel)\n")
        return False
    
    print(f"\n Total: {len(all_docs)} chunks ({len(pdf_docs)} PDF + {len(excel_docs)} Excel)")
    
    # Create embeddings
    print(f"\n🔹 Creating embeddings ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    # Build database
    print(f"🔹 Building database...")
    
    if os.path.exists(PERSIST_DIR):
        import shutil
        shutil.rmtree(PERSIST_DIR)
    
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    
    print(f"\n" + "=" * 60)
    print(f" SUCCESS - Database created at: {PERSIST_DIR}")
    print("=" * 60 + "\n")
    
    return True

if __name__ == "__main__":
    success = create_database()
    if success:
        print("Next: python chatbot.py\n")