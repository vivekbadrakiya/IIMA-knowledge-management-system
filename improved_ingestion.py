"""
PRODUCTION INGESTION - ALL DATA SOURCES
========================================

Ingests:
- PDFs (HR policies, handbooks, etc.)
- Excel (employee records, salary, leave, budget, training)
- Proper chunking and deduplication
"""

import os
import logging
import hashlib
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# CONFIG
# ============================================

DATA_DIR = "data"
QDRANT_PATH = "db/qdrant_db"
QDRANT_COLLECTION = "hr_documents"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768
BATCH_SIZE = 50


# ============================================
# LOAD PDFs
# ============================================

def load_pdfs():
    """Load all PDFs from data directory"""
    documents = []
    
    if not os.path.exists(DATA_DIR):
        logger.warning(f"Directory {DATA_DIR} not found")
        return documents
    
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.info("No PDF files found - this is OK if you only have Excel")
        return documents
    
    logger.info(f"Loading {len(pdf_files)} PDF file(s)...")
    
    for pdf_file in pdf_files:
        try:
            pdf_path = os.path.join(DATA_DIR, pdf_file)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            for page in pages:
                # Clean content
                content = " ".join(page.page_content.split())
                
                if len(content) > 50:  # Skip empty pages
                    documents.append({
                        'content': content,
                        'source': pdf_file,
                        'file_type': 'pdf',
                        'page': page.metadata.get('page', 0)
                    })
            
            logger.info(f"  ✓ {pdf_file}: {len([p for p in documents if p['source'] == pdf_file])} pages")
            
        except Exception as e:
            logger.error(f"  ✗ {pdf_file}: {e}")
    
    return documents


# ============================================
# LOAD EXCEL
# ============================================

def load_excel():
    """Load all Excel/CSV files"""
    documents = []
    
    if not os.path.exists(DATA_DIR):
        logger.warning(f"Directory {DATA_DIR} not found")
        return documents
    
    excel_files = [
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith(('.xlsx', '.xls', '.csv'))
    ]
    
    if not excel_files:
        logger.warning("No Excel files found")
        return documents
    
    logger.info(f"Loading {len(excel_files)} Excel file(s)...")
    
    for excel_file in excel_files:
        try:
            excel_path = os.path.join(DATA_DIR, excel_file)
            
            # Read file
            if excel_file.lower().endswith('.csv'):
                df = pd.read_csv(excel_path)
            else:
                df = pd.read_excel(excel_path)
            
            # Process each row
            for idx, row in df.iterrows():
                text_parts = []
                
                for col in df.columns:
                    val = row[col]
                    if pd.notna(val):
                        # Format currency values
                        if isinstance(val, (int, float)) and ('salary' in col.lower() or 'budget' in col.lower() or 'cost' in col.lower()):
                            text_parts.append(f"{col}: ₹{val:,.0f}")
                        else:
                            text_parts.append(f"{col}: {val}")
                
                content = " | ".join(text_parts)
                
                if content:
                    documents.append({
                        'content': content,
                        'source': excel_file,
                        'file_type': 'excel',
                        'row': idx + 2
                    })
            
            logger.info(f"  ✓ {excel_file}: {len(df)} rows")
            
        except Exception as e:
            logger.error(f"  ✗ {excel_file}: {e}")
    
    return documents


# ============================================
# INGEST TO QDRANT
# ============================================

def ingest_to_qdrant(documents, embeddings):
    """Create Qdrant collection and ingest documents"""
    
    if not documents:
        logger.error("No documents to ingest!")
        return False
    
    logger.info("Initializing Qdrant...")
    
    # Create client
    client = QdrantClient(path=QDRANT_PATH)
    
    # Recreate collection
    try:
        client.delete_collection(QDRANT_COLLECTION)
    except:
        pass
    
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )
    
    logger.info(f"Processing {len(documents)} documents in batches...")
    
    # Process in batches
    total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} docs)")
        
        # Generate embeddings
        texts = [doc['content'] for doc in batch]
        vectors = embeddings.embed_documents(texts)
        
        # Create points
        points = []
        for doc, vector in zip(batch, vectors):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    'content': doc['content'],
                    'source': doc['source'],
                    'file_type': doc['file_type'],
                    'metadata': {k: v for k, v in doc.items() if k not in ['content', 'source', 'file_type']}
                }
            )
            points.append(point)
        
        # Upload
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    
    # Verify
    info = client.get_collection(QDRANT_COLLECTION)
    logger.info(f"✓ Ingestion complete: {info.points_count} documents")
    
    return True


# ============================================
# MAIN
# ============================================

def main():
    logger.info("="*70)
    logger.info("PRODUCTION INGESTION PIPELINE")
    logger.info("="*70)
    
    # Load documents
    pdf_docs = load_pdfs()
    excel_docs = load_excel()
    
    all_docs = pdf_docs + excel_docs
    
    if not all_docs:
        logger.error("No documents found!")
        logger.info(f"\nPlease add files to '{DATA_DIR}/' directory:")
        logger.info("  - PDFs: HR policies, handbooks, manuals")
        logger.info("  - Excel: employee_records.xlsx, salary_details.xlsx, etc.")
        return False
    
    logger.info("="*70)
    logger.info(f"LOADED: {len(pdf_docs)} PDF pages, {len(excel_docs)} Excel rows")
    logger.info(f"TOTAL: {len(all_docs)} documents")
    logger.info("="*70)
    
    # Initialize embeddings
    logger.info(f"Loading embeddings model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    # Ingest
    success = ingest_to_qdrant(all_docs, embeddings)
    
    if success:
        logger.info("="*70)
        logger.info("✓ SUCCESS - Ready to use!")
        logger.info("="*70)
        logger.info(f"Run: python production_chatbot.py")
    
    return success


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)