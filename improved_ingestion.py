"""
UNIFIED INGESTION PIPELINE
===========================

Single workflow to ingest all data sources:
- PDFs (page-wise)
- Excel (row-wise)
- MySQL (row-wise)

All data processed together and stored in one vector database.

Usage:
    python unified_ingestion.py
"""

import os
import mysql.connector
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd


# ===========================================
# CONFIGURATION
# ===========================================

# File locations
PDF_DIR = "data"
EXCEL_DIR = "data"
PERSIST_DIR = "db/chroma_db"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# MySQL Configuration
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Vivek@2611',
    'database': 'company_hr'
}

MYSQL_TABLES = {
    'employees': ['employee_id', 'first_name', 'last_name', 'email', 'phone', 'department', 'designation', 'salary', 'status'],
    'attendance': ['attendance_id', 'employee_id', 'date', 'check_in', 'check_out', 'hours_worked', 'status'],
    'leave_applications': ['leave_id', 'employee_id', 'leave_type', 'start_date', 'end_date', 'days', 'reason', 'status'],
    'performance_reviews': ['review_id', 'employee_id', 'review_period', 'technical_skills', 'communication', 'teamwork', 'leadership', 'overall_rating', 'comments'],
    'training_programs': ['program_id', 'program_name', 'description', 'trainer', 'start_date', 'end_date', 'status'],
    'training_enrollments': ['enrollment_id', 'program_id', 'employee_id', 'enrollment_date', 'completion_status', 'score'],
    'projects': ['project_id', 'project_name', 'description', 'client', 'status', 'budget', 'department'],
    'project_assignments': ['assignment_id', 'project_id', 'employee_id', 'role', 'allocation_percentage', 'start_date', 'end_date'],
}


# ===========================================
# LOAD PDFS - PAGE-WISE CHUNKING
# ===========================================

def load_pdfs():
    """Load PDFs page by page"""
    documents = []
    
    if not os.path.exists(PDF_DIR):
        return documents
    
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    
    if not pdf_files:
        return documents
    
    print(f"\n📄 Loading {len(pdf_files)} PDF(s)...")
    
    for pdf_file in pdf_files:
        try:
            pdf_path = os.path.join(PDF_DIR, pdf_file)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # Clean and add metadata
            for page in pages:
                page.page_content = " ".join(page.page_content.split())
                page.metadata['source'] = pdf_file
                page.metadata['file_type'] = 'pdf'
            
            documents.extend(pages)
            print(f"  ✓ {pdf_file}: {len(pages)} pages")
            
        except Exception as e:
            print(f"  ✗ {pdf_file}: {e}")
    
    return documents


# ===========================================
# LOAD EXCEL - ROW-WISE CHUNKING
# ===========================================

def load_excel():
    """Load Excel files row by row"""
    documents = []
    
    if not os.path.exists(EXCEL_DIR):
        return documents
    
    excel_files = [f for f in os.listdir(EXCEL_DIR) 
                   if f.endswith(('.xlsx', '.xls', '.csv'))]
    
    if not excel_files:
        return documents
    
    print(f"\n📊 Loading {len(excel_files)} Excel file(s)...")
    
    for excel_file in excel_files:
        try:
            excel_path = os.path.join(EXCEL_DIR, excel_file)
            
            # Read Excel or CSV
            if excel_file.endswith('.csv'):
                df = pd.read_csv(excel_path)
            else:
                df = pd.read_excel(excel_path)
            
            # Convert each row to a document
            for idx, row in df.iterrows():
                text_parts = []
                for col in df.columns:
                    value = row[col]
                    if pd.notna(value):
                        text_parts.append(f"{col}: {value}")
                
                text = " | ".join(text_parts)
                
                doc = Document(
                    page_content=text,
                    metadata={
                        'source': excel_file,
                        'file_type': 'excel',
                        'row': idx + 2
                    }
                )
                documents.append(doc)
            
            print(f"  ✓ {excel_file}: {len(df)} rows")
            
        except Exception as e:
            print(f"  ✗ {excel_file}: {e}")
    
    return documents


# ===========================================
# LOAD MYSQL - ROW-WISE CHUNKING
# ===========================================

def get_mysql_connection():
    """Create MySQL connection"""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        return conn
    except mysql.connector.Error as e:
        print(f"  ✗ MySQL connection error: {e}")
        return None


def load_mysql():
    """Load MySQL data row by row"""
    documents = []
    
    if not MYSQL_TABLES:
        return documents
    
    # Test connection
    conn = get_mysql_connection()
    if not conn:
        print("\n🗄️ Skipping MySQL (connection failed)")
        return documents
    
    conn.close()
    
    print(f"\n🗄️ Loading {len(MYSQL_TABLES)} MySQL table(s)...")
    
    for table_name, columns in MYSQL_TABLES.items():
        try:
            # Connect
            conn = get_mysql_connection()
            if not conn:
                print(f"  ✗ {table_name}: Connection failed")
                continue
            
            cursor = conn.cursor(dictionary=True)
            
            # Build query
            col_list = ", ".join(columns)
            query = f"SELECT {col_list} FROM {table_name} LIMIT 10000"
            
            # Execute
            cursor.execute(query)
            rows = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            if not rows:
                print(f"  ✓ {table_name}: 0 rows")
                continue
            
            # Convert each row to document
            for row in rows:
                text_parts = []
                for col in columns:
                    value = row.get(col)
                    if value is not None:
                        text_parts.append(f"{col}: {value}")
                
                text = " | ".join(text_parts)
                
                doc = Document(
                    page_content=text,
                    metadata={
                        'source': f"MySQL:{table_name}",
                        'file_type': 'mysql',
                        'table': table_name,
                        'row_id': str(row.get('id', 'unknown'))
                    }
                )
                documents.append(doc)
            
            print(f"  ✓ {table_name}: {len(rows)} rows")
            
        except mysql.connector.Error as e:
            print(f"  ✗ {table_name}: {e}")
        except Exception as e:
            print(f"  ✗ {table_name}: {e}")
    
    return documents


# ===========================================
# UNIFIED INGESTION PIPELINE
# ===========================================

def ingest():
    """
    Main unified ingestion pipeline
    
    Process:
    1. Load PDFs (page-wise)
    2. Load Excel (row-wise)
    3. Load MySQL (row-wise)
    4. Combine all documents
    5. Create embeddings
    6. Store in vector database
    """
    
    print("=" * 70)
    print("UNIFIED INGESTION PIPELINE")
    print("=" * 70)
    
    # Load all data sources
    print("\nLoading all data sources...")
    
    pdf_docs = load_pdfs()
    excel_docs = load_excel()
    mysql_docs = load_mysql()
    
    # Combine all documents
    all_docs = pdf_docs + excel_docs + mysql_docs
    
    if not all_docs:
        print("\n✗ No documents loaded!")
        print(f"Add PDFs to: {PDF_DIR}/")
        print(f"Add Excel to: {EXCEL_DIR}/")
        print(f"Configure MySQL in MYSQL_TABLES")
        return False
    
    # Summary
    print(f"\n{'=' * 70}")
    print("DOCUMENTS LOADED")
    print(f"{'=' * 70}")
    print(f"PDF pages:      {len(pdf_docs)}")
    print(f"Excel rows:     {len(excel_docs)}")
    print(f"MySQL rows:     {len(mysql_docs)}")
    print(f"Total chunks:   {len(all_docs)}")
    print(f"{'=' * 70}")
    
    # Create embeddings
    print("\nCreating embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        print("✓ Embeddings ready")
    except Exception as e:
        print(f"✗ Error creating embeddings: {e}")
        return False
    
    # Clear old database
    if os.path.exists(PERSIST_DIR):
        import shutil
        shutil.rmtree(PERSIST_DIR)
        print(f"\nCleared old database at: {PERSIST_DIR}")
    
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    # Create vector database
    print("\nBuilding vector database...")
    try:
        Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        print("✓ Vector database created")
    except Exception as e:
        print(f"✗ Error creating database: {e}")
        return False
    
    # Final summary
    print(f"\n{'=' * 70}")
    print("INGESTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"PDF documents:      {len(pdf_docs)}")
    print(f"Excel documents:    {len(excel_docs)}")
    print(f"MySQL documents:    {len(mysql_docs)}")
    print(f"Total chunks:       {len(all_docs)}")
    print(f"Database location:  {PERSIST_DIR}")
    print(f"Embedding model:    {EMBEDDING_MODEL}")
    print(f"{'=' * 70}\n")
    
    return True


# ===========================================
# CONFIGURATION VALIDATION
# ===========================================

def validate_config():
    """Check if configuration is set up"""
    
    if MYSQL_CONFIG['password'] == 'password':
        print("\n⚠️  MySQL password not configured!")
        print("Edit MYSQL_CONFIG in unified_ingestion.py")
        print("Set your actual MySQL password\n")
        return False
    
    if MYSQL_CONFIG['database'] == 'your_database':
        print("\n⚠️  MySQL database not configured!")
        print("Edit MYSQL_CONFIG in unified_ingestion.py")
        print("Set your actual database name\n")
        return False
    
    if not MYSQL_TABLES:
        print("\n⚠️  No MySQL tables configured!")
        print("Edit MYSQL_TABLES in unified_ingestion.py")
        print("Add your table names and columns\n")
        return False
    
    return True


# ===========================================
# ENTRY POINT
# ===========================================

if __name__ == "__main__":
    try:
        # Validate config
        if not validate_config():
            print("⚠️  Continuing without MySQL support...\n")
        
        # Run ingestion
        success = ingest()
        
        if success:
            print("✓ All data ingested successfully!")
            print("Next: python simple_chatbot.py\n")
        else:
            print("✗ Ingestion failed\n")
            exit(1)
            
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
        exit(1)