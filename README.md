# RAG-based Chatbot with TinyLLaMA and FAISS

## Objective

The goal of this project is to build an interactive chatbot that leverages a Retrieval-Augmented Generation (RAG) approach to answer user queries based on a pre-loaded document corpus. By using the FAISS vector store and TinyLLaMA model, this chatbot can provide accurate, context-aware responses while ensuring efficient retrieval and generation.

Key features:
- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval and language generation for more accurate and informed responses.
- **Efficient Memory Usage**: The vector store and model are loaded only once to optimize performance.
- **Interactive Chat Interface**: Users can chat with the assistant, and it will remember the conversation history for context-aware responses.

## Approach

1. **Document Loading and Preprocessing**:
   - The first step is to load various types of documents, such as PDFs, DOCX, CSV, and XLSX, using `langchain`'s document loaders. 
   - The documents are split into manageable chunks using `RecursiveCharacterTextSplitter` to prepare them for embedding.

2. **Embedding and Vector Store**:
   - The documents are converted into embeddings using a HuggingFace embedding model (`nomic-ai/nomic-embed-text-v1`), and the embeddings are stored in a FAISS vector store for efficient similarity searches.
   - The vector store is saved locally and can be reloaded for future queries, ensuring that the system doesn’t need to rebuild the store each time.

3. **Model Loading (TinyLLaMA)**:
   - The `llama_cpp` package is used to load the TinyLLaMA model. This model is responsible for generating responses based on the retrieved context from the vector store.
   - The model is loaded once at the start to improve efficiency and avoid reloading during each query.

4. **RAG-based Query Answering**:
   - When a user submits a query, the system retrieves the top `k` relevant documents from the FAISS vector store and constructs a prompt that includes both the user’s question and the retrieved context.
   - TinyLLaMA then generates a response based on the prompt, making the conversation context-aware.

5. **Interactive Chat Loop**:
   - The user can interact with the assistant through a simple terminal-based chat interface.
   - The system maintains chat history, ensuring that the assistant’s responses are contextually relevant to prior interactions.

## Features

- **Efficiency**: The vector store and model are loaded once, improving query response time and reducing resource usage.
- **Context Awareness**: The assistant maintains conversation history, providing more natural and coherent responses.
- **Multi-document Handling**: Supports multiple document formats (PDF, DOCX, CSV, XLSX) for versatility in data sources.
- **Local Deployment**: Everything is designed to run locally, with no external dependencies required once set up.

## Technical Details

### Why TinyLLaMA?

TinyLLaMA is a smaller, optimized version of the LLaMA model, designed to provide fast and efficient inference while maintaining impressive language generation capabilities. Here’s why TinyLLaMA was chosen for this project:

1. **Efficiency and Speed**: TinyLLaMA is optimized for low-latency responses, making it ideal for real-time interactions like chatbots. It offers a good trade-off between performance and computational efficiency compared to larger models.
2. **Contextual Understanding**: Despite being smaller in size, TinyLLaMA is capable of understanding and generating contextually relevant responses, making it suitable for the Retrieval-Augmented Generation (RAG) approach in this project.
3. **Multilingual Capabilities**: TinyLLaMA is trained on diverse multilingual datasets, enabling it to handle questions in multiple languages. This makes the chatbot adaptable to a global user base and provides support for various languages.
4. **Customization and Fine-Tuning**: TinyLLaMA can be easily fine-tuned with specific datasets, making it highly customizable for different domains or applications.

### Parameters of TinyLLaMA

- `n_threads=6`: This parameter controls the number of threads used for model inference. More threads can help speed up inference on multi-core CPUs.
- `n_ctx=2048`: This defines the context window size, which determines how much of the conversation history and context the model can consider when generating a response. A larger context size allows the model to better remember prior conversation and provide more coherent responses.
- `verbose=False`: Controls whether detailed logs are shown during the execution. It is set to `False` to reduce unnecessary output and keep the terminal clean.

### FAISS Vector Store

FAISS (Facebook AI Similarity Search) is a highly efficient library for similarity search and clustering of dense vectors. It is designed to handle large-scale vector searches, which is a critical component of the RAG-based approach. Here’s why FAISS is used in this project:

1. **Efficient Similarity Search**: FAISS allows for fast and scalable similarity searches within the high-dimensional vector space. This makes it highly suitable for applications where quick retrieval of relevant documents is needed, like in this chatbot system.
2. **Handling Large Datasets**: FAISS is optimized for handling large collections of document embeddings, making it suitable for systems with a large corpus of documents.
3. **Local Deployment**: FAISS supports both CPU and GPU-based searches. In this project, it is configured to run on CPU for simplicity, but it can be switched to GPU for faster processing if needed.
4. **Memory Efficient**: By storing embeddings in a vector store, FAISS allows for fast retrieval of relevant documents without needing to load the entire document corpus into memory.

### Multilingual Capabilities

The `nomic-ai/nomic-embed-text-v1` embedding model used in this project is capable of handling multilingual text, which means that the documents processed and stored in the FAISS vector store can include content in various languages. TinyLLaMA, being a multilingual model, can then generate responses in the same language as the input query. This is especially useful in global applications where users may interact with the system in different languages.

## Requirements

- Python 3.x
- Required libraries:
  - `langchain`
  - `faiss-cpu` or `faiss-gpu`
  - `llama-cpp-python`
  - `sentence-transformers`
  - `huggingface-hub`
  - `PyMuPDF`
  - `unstructured`
  - `pandas`

Install the required libraries using:

```bash
pip install -r requirements.txt

```

## Repo Files Description
- ** Docs - This folder containing all the documents provided for this assignment
- ** faiss_nomic_index - The vector store stored locally to input the model.
- ** Data_preparation - Its the ipython notebook file contains the preprocessing stages invloved to create the vector database.
- ** chat.py - Is the python file to run the chatbot system
