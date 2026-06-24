# DocuChat
# PDF RAG Application

A Retrieval-Augmented Generation (RAG) application built with:

- Streamlit
- ChromaDB
- Ollama
- LangChain
- Sentence Transformers

## Features

- Upload PDF documents
- Store embeddings in ChromaDB
- Semantic search
- Cross-encoder reranking
- LLM-powered answers using Ollama

## Installation

```bash
git clone <repo-url>
cd rag-app

pip install -r requirements.txt
```

## Run Ollama

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama serve
```

## Run Application

```bash
streamlit run app.py
```
