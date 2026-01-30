"""
Answer Generation Pipeline - PRODUCTION VERSION

Replace your current 3_answer_generation.py with this file.

This file:
1. Retrieves relevant documents from vector database
2. Generates answers using LLM
3. Handles conversation
4. Main chatbot interface
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# -----------------------------
# Config
# -----------------------------
PERSIST_DIR = "db/chroma_db"
MODEL_NAME = "llama3.2:3b"
K_CHUNKS = 10  # Number of document chunks to retrieve
TEMPERATURE = 0.0  # 0 = factual, 1 = creative

# -----------------------------
# Initialize Components
# -----------------------------
# print("Loading chatbot...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

llm = OllamaLLM(
    model=MODEL_NAME,
    temperature=TEMPERATURE
)

# print("Chatbot ready!\n")

# -----------------------------
# Query Expansion (helps find synonyms)
# -----------------------------
def expand_query(query):
    """
    Expands query with synonyms for better retrieval
    Example: "penalties" → "penalties disciplinary sanctions punishment"
    """
    expansions = {
        'penalty': ['penalties', 'disciplinary', 'sanctions', 'punishment', 'fine'],
        'leave': ['leave', 'absence', 'time off', 'vacation', 'holiday'],
        'salary': ['salary', 'compensation', 'pay', 'wages', 'remuneration'],
        'hours': ['hours', 'working time', 'shift', 'schedule'],
        'termination': ['termination', 'dismissal', 'firing', 'separation'],
        'benefit': ['benefits', 'perks', 'allowance', 'compensation'],
        'policy': ['policy', 'rule', 'regulation', 'guideline', 'procedure'],
    }
    
    query_lower = query.lower()
    expanded = [query]
    
    for key, synonyms in expansions.items():
        if key in query_lower:
            expanded.extend([s for s in synonyms if s not in query_lower])
            break
    
    return ' '.join(expanded[:5])

# -----------------------------
# Prompts
# -----------------------------
standalone_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Extract the core topic from this question as a search query.

Question: {question}

Search Query (5-10 words):"""
)

answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Answer the question using ONLY the context below.

RULES:
1. Use ONLY information from the context
2. Be direct and concise
3. If not in context, say: "I cannot find this information in the provided documents."

Context:
{context}

Question: {question}

Answer:"""
)

# -----------------------------
# Main Chat Function
# -----------------------------
def chat(user_question):
    """
    Main chat function
    
    Args:
        user_question: User's question
        
    Returns:
        dict with 'answer' and 'source'
    """
    # Generate search query
    search_query = llm.invoke(
        standalone_prompt.format(question=user_question)
    ).strip()
    
    # Clean search query
    search_query = search_query.replace('"', '').strip()
    if len(search_query) < 3:
        search_query = user_question
    
    # Expand with synonyms
    expanded_query = expand_query(search_query)
    
    # Retrieve documents
    docs = db.similarity_search(expanded_query, k=K_CHUNKS)
    
    if not docs:
        return {
            'answer': "I cannot find relevant information in the documents.",
            'source': None
        }
    
    # Build context
    context_parts = []
    sources = set()
    
    for doc in docs:
        context_parts.append(doc.page_content)
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        sources.add(f"{source} (Page {page})")
    
    context = "\n\n".join(context_parts)
    
    # Generate answer
    answer = llm.invoke(
        answer_prompt.format(
            context=context,
            question=user_question
        )
    ).strip()
    
    # Check if answer is valid
    if "cannot find" in answer.lower():
        return {
            'answer': answer,
            'source': None
        }
    
    # Format sources
    source_list = list(sources)[:3]
    source_info = f"Source: {', '.join(source_list)}"
    if len(sources) > 3:
        source_info += f" +{len(sources)-3} more"
    
    return {
        'answer': answer,
        'source': source_info
    }

# -----------------------------
# Chat Interface
# -----------------------------
print("RAG Chatbot")
print("Commands: 'exit' to quit\n")

while True:
    user_input = input("You: ").strip()
    
    if not user_input:
        continue
    
    if user_input.lower() in ['exit', 'quit']:
        print("\n👋 Goodbye!\n")
        break
    
    try:
        result = chat(user_input)
        
        print(f"\nBot: {result['answer']}")
        
        if result['source']:
            print(f"     ({result['source']})")
        
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")