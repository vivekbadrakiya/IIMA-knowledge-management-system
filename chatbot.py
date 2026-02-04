"""
FINAL CHATBOT - ChatGPT-like Question Answering

Hardware: 8GB RAM, 4GB GPU, Intel i5-10300H
Purpose: Answer questions from documents with ChatGPT-quality responses
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

# Configuration
PERSIST_DIR = "db/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"  # Best for 8GB RAM

# Retrieval settings
K_CHUNKS = 8  # Retrieve 8 most relevant chunks
TEMPERATURE = 0.1  # Slight creativity for natural responses

def init_chatbot():
    """Initialize chatbot"""
    
    if not os.path.exists(PERSIST_DIR):
        print(" Database not found!")
        print("Run: python ingestion.py\n")
        return None, None
    
    print(" Loading chatbot...")
    
    # Load database
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    
    # Load LLM
    llm = OllamaLLM(
        model=LLM_MODEL,
        temperature=TEMPERATURE
    )
    
    print("Ready!\n")
    return db, llm

def get_answer(question, db, llm):
    """Get ChatGPT-like answer"""
    
    # Normalize question (handle uppercase, lowercase, etc.)
    question = question.strip()
    
    # Retrieve relevant documents
    docs = db.similarity_search(question, k=K_CHUNKS)
    
    if not docs:
        return "I don't have information about that in the documents."
    
    # Build context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # ChatGPT-like prompt
    prompt = f"""You are a helpful AI assistant. Answer the question based on the information provided.

Be natural, conversational, and helpful - like ChatGPT.
If the information is in the context, give a complete, well-structured answer.
If not in the context, say you don't have that information.

Context from documents:
{context}

Question: {question}

Answer:"""
    
    # Generate answer
    answer = llm.invoke(prompt)
    return answer.strip()

def run_chat():
    """Run chatbot"""
    
    db, llm = init_chatbot()
    if not db or not llm:
        return
    
    print("=" * 60)
    print("🤖 DOCUMENT CHATBOT")
    print("=" * 60)
    print("Ask anything about your documents!")
    print("Type 'exit' to quit\n")
    
    while True:
        question = input("You: ").strip()
        
        if not question:
            continue
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!\n")
            break
        
        try:
            answer = get_answer(question, db, llm)
            print(f"\nBot: {answer}\n")
        except Exception as e:
            print(f"\n Error: {str(e)}\n")

if __name__ == "__main__":
    run_chat()