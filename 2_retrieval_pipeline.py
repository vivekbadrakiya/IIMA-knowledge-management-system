from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = "db/chroma_db"

def get_retriever(k=5):
    print("🔹 Loading vector store...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": k})

    print("✅ Retriever ready. You can now ask questions.")
    return retriever


if __name__ == "__main__":
    retriever = get_retriever()

    while True:
        query = input("\nAsk a question (type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        docs = retriever.invoke(query)

        print("\n--- Retrieved Context ---")
        for i, doc in enumerate(docs, 1):
            print(f"\nChunk {i}:\n{doc.page_content[:500]}...")
