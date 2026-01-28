from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# -----------------------------
# Config
# -----------------------------
PERSIST_DIR = "db/chroma_db"
MODEL_NAME = "llama3.2:3b"

# -----------------------------
# Load Embeddings & Vector DB
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 4})

# -----------------------------
# Load LLM
# -----------------------------
llm = OllamaLLM(model=MODEL_NAME)

# -----------------------------
# Prompt for History-Aware Question Rewriting
# -----------------------------
rewrite_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
You are a helpful assistant.

Given the chat history and a follow-up question, rewrite the question so that it is a standalone question.

Chat History:
{chat_history}

Follow-up Question:
{question}

Standalone Question:
"""
)

# -----------------------------
# Prompt for Answer Generation
# -----------------------------
answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer the question using ONLY the information from the context below.
If the answer is not present in the context, say:
"I don't have enough information from the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
)

# -----------------------------
# Chat Memory (Simple & Clean)
# -----------------------------
chat_history = []

print("\n🧠 RAG Chat Started")
print("Type your question below.")
print("Type 'exit' to end the chat.\n")

# -----------------------------
# Chat Loop
# -----------------------------
while True:
    user_question = input("Ask a question: ").strip()

    if user_question.lower() in ["exit", "quit"]:
        print("\n👋 Chat ended. Goodbye!")
        break

    # ---- Step 1: Rewrite Question (History-Aware)
    history_text = "\n".join(chat_history)

    standalone_question = llm.invoke(
        rewrite_prompt.format(
            chat_history=history_text,
            question=user_question
        )
    )

    # ---- Step 2: Retrieve Relevant Chunks
    docs = retriever.invoke(standalone_question)

    if not docs:
        print("\n⚠️ No relevant information found.\n")
        continue

    context = "\n\n".join([doc.page_content for doc in docs])

    # ---- Step 3: Generate Answer
    answer = llm.invoke(
        answer_prompt.format(
            context=context,
            question=standalone_question
        )
    )

    # ---- Step 4: Display Answer
    print("\n🧠 Answer:")
    print(answer.strip(), "\n")

    # ---- Step 5: Save History
    chat_history.append(f"User: {user_question}")
    chat_history.append(f"Assistant: {answer.strip()}")