"""
This script provides a Retrieval-Augmented Generation (RAG)-based chatbot using the TinyLLaMA model and FAISS vector store. The chatbot performs the following operations:

1. **Embedding and Vector Store**: 
   - Loads precomputed document embeddings from a FAISS index.
   - Uses `HuggingFaceEmbeddings` with a model (`nomic-ai/nomic-embed-text-v1`) to convert documents into vector embeddings.
   
2. **TinyLLaMA Model**:
   - Loads the TinyLLaMA model (`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`) for generating natural language responses.
   - The model is loaded once and used to generate answers based on retrieved context from the FAISS vector store.

3. **RAG-based Query Answering**:
   - A function `ask_llama_rag(query, chat_history)` that takes a user query, searches for relevant documents in the vector store, and generates an answer using TinyLLaMA.
   
4. **Interactive Chat Loop**:
   - An interactive chat loop allows the user to input questions, receive answers, and continue the conversation with context.
   - The conversation history is appended for continuity, and the assistant answers based on the context of the query and previous chat history.
   
The model supports a variety of document types, such as PDF, DOCX, XLSX, and CSV, stored in a FAISS vector store for efficient retrieval. 

Dependencies:
- `langchain`
- `faiss-cpu` or `faiss-gpu`
- `llama-cpp-python`
- `sentence-transformers`
- `huggingface-hub`
- `PyMuPDF`
- `unstructured`
- `pandas`
"""


from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from llama_cpp import Llama
import streamlit as st
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
import streamlit as st
st.set_page_config(
    page_title="🧠 TinyLLaMA Chat",
    page_icon="🤖",  # You can use an emoji or link to a .ico/.png
)

chat_history = []

st.title("💬 Chat with TinyLLaMA + RAG")

# Input box
user_input = st.text_input("You:")

if user_input:
    if user_input.lower() in ["exit", "quit"]:
        st.write("👋 Ending chat.")
    else:
        # Show spinner while processing
        with st.spinner("🔍 Searching the documents..."):
            answer = ask_llama_rag(user_input, chat_history)

        # Show result
        st.write("Assistant:", answer)

        # Append to local history
        chat_history.append(f"<|user|>\n{user_input}")
        chat_history.append(f"<|assistant|>\n{answer}")

