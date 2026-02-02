"""
Diagnostic Tool - NEW FILE

Create this as a NEW file called: diagnostic_tool.py

This file:
1. Shows what's in your vector database
2. Tests search queries
3. Helps troubleshoot why chatbot fails
4. Run this when chatbot says "cannot find"
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "db/chroma_db"

print("🔍 Loading vector database...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

collection = db._collection
total_chunks = collection.count()

print(f"✅ Loaded: {total_chunks} chunks\n")

# -----------------------------
# Show Database Info
# -----------------------------
print("=" * 70)
print("📊 DATABASE INFO")
print("=" * 70)

all_docs = db.get()
sources = {}
for metadata in all_docs['metadatas']:
    source = metadata.get('source', 'Unknown')
    sources[source] = sources.get(source, 0) + 1

print(f"Total chunks: {total_chunks}\n")
print("Documents:")
for source, count in sources.items():
    print(f"  • {source}: {count} chunks")

print("\n" + "=" * 70)

# -----------------------------
# Interactive Testing
# -----------------------------
print("\n🔍 DIAGNOSTIC MODE")
print("=" * 70)
print("Commands:")
print("  • Type a question to test search")
print("  • 'keyword:word' - Check if word exists in database")
print("  • 'exit' - Quit")
print("=" * 70 + "\n")

def test_search(query):
    """Test search and show results"""
    print(f"\n🔎 Searching: '{query}'")
    print("-" * 70)
    
    results = db.similarity_search(query, k=10)
    
    if not results:
        print("❌ No results found!\n")
        return
    
    print(f"✅ Found {len(results)} chunks:\n")
    
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        
        print(f"[{i}] {source} (Page {page})")
        content = doc.page_content[:200]
        print(content + "...\n")
    
    print("=" * 70)

def keyword_search(keyword):
    """Search for exact keyword"""
    print(f"\n🔍 Looking for keyword: '{keyword}'")
    print("-" * 70)
    
    all_docs = db.get()
    matches = []
    
    for content, metadata in zip(all_docs['documents'], all_docs['metadatas']):
        if keyword.lower() in content.lower():
            matches.append((content, metadata))
    
    if not matches:
        print(f"❌ '{keyword}' NOT FOUND in any chunks!")
        print("\nThis means:")
        print("  1. The word doesn't exist in your PDF, OR")
        print("  2. The PDF uses different terminology\n")
        print("Try these:")
        print(f"  keyword:disciplinary")
        print(f"  keyword:punishment")
        print(f"  keyword:sanctions\n")
        return
    
    print(f"✅ Found '{keyword}' in {len(matches)} chunks:\n")
    
    for i, (content, metadata) in enumerate(matches[:5], 1):
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page', 'Unknown')
        
        print(f"[{i}] {source} (Page {page})")
        
        lines = content.split('\n')
        for line in lines:
            if keyword.lower() in line.lower():
                print(f"  → {line.strip()}")
        print()
    
    if len(matches) > 5:
        print(f"... and {len(matches) - 5} more matches\n")
    
    print("=" * 70)

# MAIN LOOP
while True:
    command = input("\nEnter command: ").strip()
    
    if not command:
        continue
    
    if command.lower() in ['exit', 'quit']:
        print("\n👋 Goodbye!\n")
        break
    
    if command.startswith('keyword:'):
        keyword = command.replace('keyword:', '').strip()
        keyword_search(keyword)
    else:
        test_search(command)