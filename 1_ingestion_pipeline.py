"""
FIXED Ingestion Pipeline

Key Fixes:
1. Better Excel chunking - Each row becomes a separate chunk with full context
2. More descriptive Excel content for better retrieval
3. Proper metadata for accurate page/sheet references
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd

# ---------------------------
# Configuration
# ---------------------------
PDF_DIR = "data"        
EXCEL_DIR = "data"    
PERSIST_DIR = "db/chroma_db"

def process_excel_file(file_path, file_name):
    """
    Process Excel file with BETTER chunking strategy
    
    Strategy: Each row = 1 chunk with descriptive content
    """
    documents = []
    
    try:
        # Read Excel file
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_path)
            sheet_name = 'Sheet1'
        else:
            excel_file = pd.ExcelFile(file_path)
            sheet_name = excel_file.sheet_names[0]  # Use first sheet
            df = excel_file.parse(sheet_name)
        
        # Create one chunk per row with full context
        for idx, row in df.iterrows():
            # Build descriptive text for this row
            row_text = f"FILE: {file_name}\n"
            row_text += f"SHEET: {sheet_name}\n"
            row_text += f"ROW: {idx + 2}\n\n"  # +2 because Excel rows start at 1 and row 1 is header
            
            # Add each column with clear labels
            for col in df.columns:
                value = row[col]
                # Handle NaN values
                if pd.isna(value):
                    value = "N/A"
                row_text += f"{col}: {value}\n"
            
            # Add searchable summary for better retrieval
            row_text += f"\n[SEARCHABLE SUMMARY]\n"
            
            # Make key fields more searchable
            if 'Name' in df.columns:
                row_text += f"Employee Name: {row.get('Name', 'N/A')}\n"
            if 'Employee_Name' in df.columns:
                row_text += f"Employee Name: {row.get('Employee_Name', 'N/A')}\n"
            if 'Department' in df.columns:
                row_text += f"Department: {row.get('Department', 'N/A')}\n"
            if 'Program_Name' in df.columns:
                row_text += f"Program: {row.get('Program_Name', 'N/A')}\n"
            if 'Designation' in df.columns:
                row_text += f"Job Title: {row.get('Designation', 'N/A')}\n"
            if 'Salary' in df.columns:
                row_text += f"Monthly Salary: {row.get('Salary', 'N/A')}\n"
            if 'Gross_Salary' in df.columns:
                row_text += f"Gross Salary: {row.get('Gross_Salary', 'N/A')}\n"
            if 'Net_Salary' in df.columns:
                row_text += f"Net Salary: {row.get('Net_Salary', 'N/A')}\n"
            
            # Create Document
            doc = Document(
                page_content=row_text,
                metadata={
                    'source': file_name,
                    'file_type': 'excel',
                    'source_type': 'Excel',
                    'sheet_name': sheet_name,
                    'row_number': idx + 2,  # Excel row number
                    'page': f"Row {idx + 2}"  # For display
                }
            )
            
            documents.append(doc)
        
        return documents, len(df)
        
    except Exception as e:
        print(f"    ❌ Error processing {file_name}: {str(e)}")
        return [], 0

def ingest_documents():
    """
    Ingest PDF and Excel documents
    """
    print("=" * 70)
    print("📚 FIXED INGESTION PIPELINE")
    print("=" * 70)
    print("Improvements:")
    print("  • Better Excel chunking (row-level with full context)")
    print("  • Enhanced searchability")
    print("  • Accurate page/row references")
    print("=" * 70 + "\n")
    
    all_documents = []

    # ---------------------------
    # Load PDFs - Page by Page
    # ---------------------------
    if os.path.exists(PDF_DIR):
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
        
        if pdf_files:
            print(f"📄 Processing PDF files...\n")
            
            for pdf in pdf_files:
                pdf_path = os.path.join(PDF_DIR, pdf)
                
                try:
                    loader = PyPDFLoader(pdf_path)
                    pages = loader.load()
                    
                    # Add metadata
                    for page in pages:
                        page.metadata['source'] = pdf
                        page.metadata['file_type'] = 'pdf'
                        page.metadata['source_type'] = 'PDF'
                        # Ensure page is 1-indexed for display
                        actual_page = page.metadata.get('page', 0)
                        page.metadata['page_display'] = actual_page + 1
                    
                    all_documents.extend(pages)
                    
                    print(f"  ✅ {pdf}: {len(pages)} pages loaded")
                    
                except Exception as e:
                    print(f"  ❌ Error loading {pdf}: {str(e)}")
                    continue
        else:
            print("⚠️  No PDF files found in", PDF_DIR)
    else:
        print(f"⚠️  PDF directory not found: {PDF_DIR}")

    # ---------------------------
    # Load Excel Files - Row by Row
    # ---------------------------
    if os.path.exists(EXCEL_DIR):
        excel_files = [
            f for f in os.listdir(EXCEL_DIR)
            if f.endswith(".xlsx") or f.endswith(".xls") or f.endswith(".csv")
        ]
        
        if excel_files:
            print(f"\n📊 Processing Excel files...\n")
            
            for excel_file in excel_files:
                excel_path = os.path.join(EXCEL_DIR, excel_file)
                
                print(f"  Processing: {excel_file}")
                docs, row_count = process_excel_file(excel_path, excel_file)
                
                if docs:
                    all_documents.extend(docs)
                    print(f"    ✅ {row_count} rows → {len(docs)} chunks")
        else:
            print(f"⚠️  No Excel files found in {EXCEL_DIR}")
    else:
        print(f"⚠️  Excel directory not found: {EXCEL_DIR}")

    # ---------------------------
    # Validation
    # ---------------------------
    if not all_documents:
        raise FileNotFoundError(
            "❌ No documents found.\n"
            f"Please add files to:\n  - {PDF_DIR}/ (for PDFs)\n  - {EXCEL_DIR}/ (for Excel/CSV)"
        )

    # Count by type
    pdf_docs = [d for d in all_documents if d.metadata.get('file_type') == 'pdf']
    excel_docs = [d for d in all_documents if d.metadata.get('file_type') == 'excel']
    
    print(f"\n📊 Total documents loaded:")
    print(f"   PDF pages: {len(pdf_docs)}")
    print(f"   Excel rows: {len(excel_docs)}")
    print(f"   Total chunks: {len(all_documents)}")
    
    # ---------------------------
    # Generate Embeddings
    # ---------------------------
    print("\n🔹 Generating embeddings...")
    print("   This may take a few minutes...\n")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("✅ Embedding model loaded")

    # ---------------------------
    # Create Vector Store
    # ---------------------------
    print("\n🔹 Creating vector database...")
    
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    vector_store = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    print(f"✅ Vector store created at: {PERSIST_DIR}")

    # ---------------------------
    # Summary
    # ---------------------------
    print("\n" + "=" * 70)
    print("📊 INGESTION SUMMARY")
    print("=" * 70)
    print(f"PDF Documents:         {len(pdf_docs)} pages")
    print(f"Excel Documents:       {len(excel_docs)} rows")
    print(f"Total Chunks:          {len(all_documents)}")
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