from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from llama_cpp import Llama

# ----------------- Load FAISS once ------------------
model_name = "nomic-ai/nomic-embed-text-v1"
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu', 'trust_remote_code': True},
    encode_kwargs={'normalize_embeddings': False}
)

vectorstore = FAISS.load_local(
    "faiss_nomic_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# ----------------- Load TinyLLaMA once ------------------
llm = Llama(
    model_path="C:\\Users\\hp\\models\\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_threads=6,
    n_ctx=2048,
    verbose=False
)

# ----------------- RAG-based response ------------------
def ask_llama_rag(query, chat_history, k=5, max_tokens=256):
    relevant_docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    full_chat = "\n".join(chat_history)
    prompt = f"""<|user|>
You are a helpful assistant. Answer based on the context below.

Context:
{context}

Conversation so far:
{full_chat}
<|user|>
{query}
<|assistant|>"""

    response = llm(prompt, max_tokens=max_tokens, temperature=0.7, top_p=0.9)
    answer = response["choices"][0]["text"].strip()
    return answer

# ----------------- Chat Loop ------------------
chat_history = []

print("💬 Chat with TinyLLaMA + RAG (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Ending chat.")
        break

    answer = ask_llama_rag(user_input, chat_history)
    print("Assistant:", answer)

    # Append to chat history for continuity
    chat_history.append(f"<|user|>\n{user_input}")
    chat_history.append(f"<|assistant|>\n{answer}")
