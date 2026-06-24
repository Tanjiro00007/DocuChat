import os
import tempfile

import chromadb
import ollama
import streamlit as st

from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile


system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context.

Context will be passed as "Context:"
User question will be passed as "Question:"

Important:
- Answer only from the provided context.
- If the answer is not available in the context, clearly state that.
"""


def process_document(uploaded_file: UploadedFile) -> list[Document]:
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )

    try:
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()

        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )

        splits = text_splitter.split_documents(documents)
        return splits

    finally:
        try:
            os.remove(temp_file.name)
        except Exception:
            pass


def get_vector_collection() -> chromadb.Collection:

    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(
        path="./demo-rag-chroma"
    )

    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(
    all_splits: list[Document],
    file_name: str,
):
    collection = get_vector_collection()

    documents = []
    metadatas = []
    ids = []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    st.success("PDF processed successfully!")


def query_collection(
    prompt: str,
    n_results: int = 10,
):
    collection = get_vector_collection()

    results = collection.query(
        query_texts=[prompt],
        n_results=n_results,
    )

    return results


def call_llm(context: str, prompt: str):

    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {prompt}",
            },
        ],
    )

    for chunk in response:
        if not chunk["done"]:
            yield chunk["message"]["content"]


def re_rank_cross_encoders(
    query: str,
    documents: list[str],
):
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    ranks = encoder_model.rank(
        query,
        documents,
        top_k=min(3, len(documents)),
    )

    for rank in ranks:
        corpus_id = rank["corpus_id"]

        relevant_text += (
            documents[corpus_id] + "\n\n"
        )

        relevant_text_ids.append(corpus_id)

    return relevant_text, relevant_text_ids


# ---------------- STREAMLIT APP ---------------- #

st.set_page_config(
    page_title="RAG Question Answer",
    layout="wide",
)

with st.sidebar:

    st.title("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
    )

    process = st.button("⚡ Process")

    if uploaded_file and process:

        normalized_file_name = (
            uploaded_file.name.translate(
                str.maketrans(
                    {
                        "-": "_",
                        ".": "_",
                        " ": "_",
                    }
                )
            )
        )

        all_splits = process_document(
            uploaded_file
        )

        add_to_vector_collection(
            all_splits,
            normalized_file_name,
        )

st.header("🗣️ RAG Question Answer")

prompt = st.text_area(
    "Ask a question about your PDF:"
)

ask = st.button("🔥 Ask")

if ask and prompt:

    try:
        results = query_collection(prompt)

        documents = results["documents"][0]

        relevant_text, relevant_text_ids = (
            re_rank_cross_encoders(
                prompt,
                documents,
            )
        )

        response = call_llm(
            context=relevant_text,
            prompt=prompt,
        )

        st.write_stream(response)

        with st.expander(
            "Retrieved Documents"
        ):
            st.write(results)

        with st.expander(
            "Most Relevant Chunks"
        ):
            st.write(relevant_text_ids)
            st.write(relevant_text)

    except Exception as e:
        st.error(f"Error: {str(e)}")