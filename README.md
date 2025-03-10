# AI Powered PDF Document Q&A

## Overview
This project implements an AI-powered Q&A system for PDF documents using LangChain, FAISS, OpenAI, and Streamlit. Users can upload a PDF file, and the system processes the document into chunks, generates embeddings, stores them in a FAISS vector database, and retrieves relevant information to answer user queries.

## Features
- Upload and process PDF documents
- Split documents into chunks for efficient retrieval
- Embed text using Hugging Face's `all-MiniLM-L6-v2` model
- Store and retrieve document embeddings with FAISS
- Answer user queries using OpenAI's GPT-4o-mini model
- Interactive web interface using Streamlit

## Installation

### Prerequisites
Ensure you have Python installed (>=3.8). Install the required dependencies:

```bash
pip install langchain langchain_community langchain_openai langchain_huggingface streamlit faiss-cpu
```

## Usage
Run the application using Streamlit:

```bash
streamlit run app.py
```

### Steps
1. Upload a PDF file.
2. The system processes and stores document embeddings.
3. Type a question related to the uploaded document.
4. The AI retrieves relevant information and provides an answer.

## Code Overview
### Key Components
- **Document Loader**: Loads and splits the PDF into chunks.
- **Embedding Generation**: Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
- **Vector Storage**: FAISS is used to store and retrieve document embeddings.
- **Retrieval Chain**: Combines the retriever with an OpenAI-powered question-answering system.
- **Streamlit UI**: Provides a simple interface for user interaction.

## Future Enhancements
- Support for multiple file types (e.g., Word, TXT)
- Improved chunking strategies for better retrieval
- Option to select different LLM models

## License
This project is licensed under the MIT License.

## Acknowledgments
- [LangChain](https://python.langchain.com/)
- [FAISS](https://faiss.ai/)
- [OpenAI](https://openai.com/)

