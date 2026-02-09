"""
UNIFIED CHATBOT
===============

Chat interface for unified RAG system.

Features:
- Query all data sources (PDFs, Excel, MySQL)
- Generate answers with LLM
- Show sources with indicators
- Simple command interface

Usage:
    python simple_chatbot.py

Note: Use unified_ingestion.py to prepare all data sources
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM


# Configuration
PERSIST_DIR = "db/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"
K_CHUNKS = 8  # Number of documents to retrieve


# ===========================================
# INITIALIZATION
# ===========================================

def init():
    """Initialize chatbot components"""
    
    if not os.path.exists(PERSIST_DIR):
        print("✗ Database not found!")
        print(f"Run: python simple_ingestion.py\n")
        return None, None
    
    print("Loading chatbot...\n")
    
    try:
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        # Load database
        db = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
        
        count = db._collection.count()
        print(f"✓ Database loaded: {count} chunks")
        
        # Load LLM
        llm = OllamaLLM(model=LLM_MODEL, temperature=0.1)
        print(f"✓ LLM ready: {LLM_MODEL}\n")
        
        return db, llm
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return None, None


# ===========================================
# ANSWER GENERATION
# ===========================================

def get_answer(question, db, llm):
    """Generate answer from question"""
    
    # Search database
    try:
        docs = db.similarity_search(question, k=K_CHUNKS)
    except Exception as e:
        print(f"✗ Search error: {e}\n")
        return None, []
    
    if not docs:
        return "I don't have information about that.", []
    
    # Build context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Generate answer
    prompt = f"""Answer based on this information:

{context}

Question: {question}

Answer:"""
    
    try:
        answer = llm.invoke(prompt).strip()
    except Exception as e:
        print(f"✗ LLM error: {e}\n")
        answer = "Error generating answer."
    
    # Get sources with file type indicators
    sources = {}
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        file_type = doc.metadata.get('file_type', 'unknown').upper()
        
        # Create source indicator
        if file_type == 'PDF':
            source_label = f"📄 {source}"
        elif file_type == 'EXCEL':
            source_label = f"📊 {source}"
        elif file_type == 'MYSQL':
            table = doc.metadata.get('table', 'Unknown')
            source_label = f"🗄️ MySQL:{table}"
        else:
            source_label = source
        
        if source_label not in sources:
            sources[source_label] = True
    
    return answer, list(sources.keys())


# ===========================================
# CHAT INTERFACE
# ===========================================

def chat():
    """Interactive chat"""
    
    db, llm = init()
    if not db or not llm:
        return
    
    print("=" * 60)
    print("SIMPLE CHATBOT")
    print("=" * 60)
    print("Ask questions about your data")
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit']:
                print("\nGoodbye!\n")
                break
            
            # Get answer
            answer, sources = get_answer(question, db, llm)
            
            print(f"\nBot: {answer}")
            
            if sources:
                print(f"Sources: {', '.join(sources)}")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!\n")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}\n")


if __name__ == "__main__":
    chat()