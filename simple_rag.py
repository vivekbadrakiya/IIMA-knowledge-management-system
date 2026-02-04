import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
import pandas as pd

PDF_DIR = "data"
EXCEL_DIR = "data"
PERSIST_DIR = "db/chroma_db"

# Lightweight models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
LLM_MODEL = "llama3.2:3b"  

# STEP 1: INGESTION

def load_pdfs():
    """Load PDF files page by page"""
    documents = []
    
    if not os.path.exists(PDF_DIR):
        return documents
    
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    
    print(f" Loading {len(pdf_files)} PDF file(s)...")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            documents.extend(pages)
            print(f"    {pdf_file}: {len(pages)} pages")
        except Exception as e:
            print(f"    Error loading {pdf_file}: {e}")
    
    return documents

def load_excel_files():
    """Load Excel files - each row becomes a searchable chunk"""
    documents = []
    
    if not os.path.exists(EXCEL_DIR):
        return documents
    
    excel_files = [f for f in os.listdir(EXCEL_DIR) 
                   if f.endswith((".xlsx", ".xls", ".csv"))]
    
    print(f"\n Loading {len(excel_files)} Excel file(s)...")
    
    for excel_file in excel_files:
        excel_path = os.path.join(EXCEL_DIR, excel_file)
        
        try:
            # Read Excel/CSV
            if excel_file.endswith('.csv'):
                df = pd.read_csv(excel_path)
            else:
                df = pd.read_excel(excel_path)
            
            # Create one chunk per row with clear formatting
            for idx, row in df.iterrows():
                # Build clear text for this row
                text_parts = [f"From file: {excel_file}\n"]
                
                for col in df.columns:
                    value = row[col]
                    if pd.notna(value):
                        text_parts.append(f"{col}: {value}")
                
                full_text = "\n".join(text_parts)
                
                doc = Document(
                    page_content=full_text,
                    metadata={"source": excel_file, "type": "excel"}
                )
                documents.append(doc)
            
            print(f"    {excel_file}: {len(df)} rows")
            
        except Exception as e:
            print(f"    Error loading {excel_file}: {e}")
    
    return documents

def create_database():
    print("CREATING VECTOR DATABASE")
    
    # Load all documents
    pdf_docs = load_pdfs()
    excel_docs = load_excel_files()
    
    all_docs = pdf_docs + excel_docs
    
    if not all_docs:
        print(" No documents found!")
        return False
    
    print(f"\n Total: {len(all_docs)} chunks")
    print(f"   PDF: {len(pdf_docs)} pages")
    print(f"   Excel: {len(excel_docs)} rows")
    
    # Create embeddings
    print(f"\n🔹 Creating embeddings using {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Create vector database
    print("🔹 Building vector database...")
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    
    print("\n Database created successfully!")
    return True

# STEP 2: CHATBOT

def init_chatbot():
    """Initialize chatbot components"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    
    llm = OllamaLLM(
        model=LLM_MODEL,
        temperature=0.0  # Deterministic answers
    )
    
    return db, llm

def get_answer(question, db, llm):
    """Get answer for a question"""
    
    # Retrieve relevant documents
    docs = db.similarity_search(question, k=5)
    
    if not docs:
        return "I cannot find any relevant information in the documents."
    
    # Combine context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create strict prompt
    prompt = f"""Answer this question using ONLY the information provided below.

RULES:
- Use ONLY the information given
- Do NOT make up or guess information
- If the answer is not in the provided information, say "I cannot find this information in the documents"
- Be specific and detailed in your answer

Information:
{context}

Question: {question}

Answer:"""
    
    # Get answer from LLM
    answer = llm.invoke(prompt)
    
    return answer.strip()

def run_chatbot():
    """Run the chatbot"""
    print("\n" + "=" * 60)
    print("🤖 SIMPLE RAG CHATBOT")
    print("=" * 60)
    print("Type 'exit' to quit\n")
    
    # Initialize
    print("Loading chatbot...")
    db, llm = init_chatbot()
    print("✅ Ready!\n")
    
    # Chat loop
    while True:
        question = input("You: ").strip()
        
        if not question:
            continue
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!\n")
            break
        
        print("\nBot: ", end="", flush=True)
        answer = get_answer(question, db, llm)
        print(f"{answer}\n")

# MAIN

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        # Run ingestion
        create_database()
    else:
        # Run chatbot
        if not os.path.exists(PERSIST_DIR):
            print("❌ Database not found!")
            print("Run: python simple_rag.py ingest")
        else:
            run_chatbot()