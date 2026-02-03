"""
FIXED Answer Generation

Key Fixes:
1. Strict anti-hallucination prompt
2. Forces LLM to only use provided context
3. Better page number extraction (1-indexed)
4. Filters to prioritize Excel results for Excel queries
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
K_CHUNKS = 10  # Retrieve more chunks to ensure we get Excel data
TEMPERATURE = 0.0

# Initialize
print("🔧 Loading chatbot...")

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

print("✅ Chatbot ready!\n")

# -----------------------------
# Query Expansion
# -----------------------------
def expand_query(query):
    """Expand query with synonyms"""
    expansions = {
        'designation': ['designation', 'job title', 'position', 'role'],
        'salary': ['salary', 'compensation', 'pay', 'wages', 'gross salary', 'net salary'],
        'name': ['name', 'employee name', 'person'],
        'department': ['department', 'division', 'team'],
        'training': ['training', 'program', 'course'],
        'leave': ['leave', 'absence', 'vacation'],
        'budget': ['budget', 'allocation', 'spending'],
    }
    
    query_lower = query.lower()
    expanded = [query]
    
    for key, synonyms in expansions.items():
        if key in query_lower:
            expanded.extend([s for s in synonyms if s not in query_lower])
            break
    
    return ' '.join(expanded[:5])

# -----------------------------
# Prompts with STRICT Anti-Hallucination
# -----------------------------

# CRITICAL: Very strict prompt to prevent hallucination
answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a document assistant. Answer the question using ONLY the exact information from the context below.

CRITICAL RULES - YOU MUST FOLLOW THESE:
1. Use ONLY information that is EXPLICITLY stated in the context
2. DO NOT make up information, infer, or guess
3. DO NOT use your general knowledge
4. If the answer is NOT in the context, you MUST say: "I cannot find this information in the provided documents."
5. DO NOT say "according to the manual" or "based on the policy" unless that exact text is in the context
6. Copy relevant details DIRECTLY from the context
7. Provide a COMPLETE and DETAILED answer (4-5 sentences) using ONLY context information

Context:
{context}

Question: {question}

Answer (use ONLY the context above):"""
)

# -----------------------------
# Smart retrieval with source filtering
# -----------------------------
def detect_query_type(query):
    """Detect if query is asking about Excel data"""
    query_lower = query.lower()
    
    # Employee/person queries
    person_indicators = ['rajesh', 'priya', 'amit', 'sneha', 'vikram', 'anjali', 
                        'rohit', 'kavita', 'arjun', 'pooja', 'employee', 'person']
    
    # Excel-specific queries
    excel_indicators = ['salary', 'designation', 'department', 'training', 
                       'budget', 'leave application', 'participant', 'program']
    
    is_person_query = any(name in query_lower for name in person_indicators)
    is_excel_query = any(indicator in query_lower for indicator in excel_indicators)
    
    return 'excel' if (is_person_query or is_excel_query) else 'general'

# -----------------------------
# Main Chat Function
# -----------------------------
def chat(user_question):
    """
    Main chat function with anti-hallucination
    """
    # Detect query type
    query_type = detect_query_type(user_question)
    
    # Expand query
    expanded_query = expand_query(user_question)
    
    # Retrieve documents
    docs = db.similarity_search(expanded_query, k=K_CHUNKS)
    
    if not docs:
        return {
            'answer': "I cannot find relevant information in the documents.",
            'source': None,
            'page': None
        }
    
    # If Excel query, prioritize Excel documents
    if query_type == 'excel':
        excel_docs = [d for d in docs if d.metadata.get('file_type') == 'excel']
        if excel_docs:
            # Use only Excel docs for Excel queries
            docs = excel_docs[:5]  # Top 5 Excel results
    
    # Build context from retrieved docs
    context_parts = []
    
    for doc in docs:
        context_parts.append(doc.page_content)
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Get metadata from FIRST (most relevant) document
    first_doc = docs[0]
    source_file = first_doc.metadata.get('source', 'Unknown')
    source_type = first_doc.metadata.get('source_type', 'Document')
    
    # Get page/row reference
    if source_type == 'PDF':
        # For PDF, get the actual page number (1-indexed)
        page_num = first_doc.metadata.get('page', 0)
        page_display = f"Page {page_num + 1}"  # +1 for 1-indexing
    elif source_type == 'Excel':
        # For Excel, show sheet name
        sheet_name = first_doc.metadata.get('sheet_name', 'Sheet1')
        page_display = f"Sheet: {sheet_name}"
    else:
        page_display = None
    
    # Generate answer with STRICT anti-hallucination prompt
    answer = llm.invoke(
        answer_prompt.format(
            context=context,
            question=user_question
        )
    ).strip()
    
    # Check if answer is valid
    if "cannot find" in answer.lower() or "not mentioned" in answer.lower():
        return {
            'answer': "I cannot find this information in the provided documents.",
            'source': None,
            'page': None
        }
    
    return {
        'answer': answer,
        'source': source_file,
        'page': page_display,
        'source_type': source_type
    }

# -----------------------------
# Chat Interface
# -----------------------------
print("=" * 70)
print("🤖 FIXED RAG Chatbot (Anti-Hallucination)")
print("=" * 70)
print("Improvements:")
print("  • Strict context-only responses")
print("  • Better Excel data retrieval")
print("  • Accurate page numbers")
print("\nCommands: 'exit' to quit\n")

while True:
    user_input = input("You: ").strip()
    
    if not user_input:
        continue
    
    if user_input.lower() in ['exit', 'quit']:
        print("\n👋 Goodbye!\n")
        break
    
    try:
        result = chat(user_input)
        
        # Display answer
        print(f"\nBot: {result['answer']}")
        
        # Display source and page
        if result['source']:
            source_line = f"     (Source: {result['source']}"
            if result['page']:
                source_line += f", {result['page']}"
            source_line += ")"
            print(source_line)
        
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")