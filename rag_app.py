"""
RAG CHATBOT - STREAMLIT WEB APP
================================
Same RAG pipeline as main.py, but with a simple browser-based chat UI.

SETUP (run once in terminal):
    pip install streamlit groq sentence-transformers scikit-learn numpy pypdf

Put your Groq key in .streamlit/secrets.toml as:
    GROQ_API_KEY = "your_key_here"
(This file already exists in your project folder — just add the line above.)

RUN:
    streamlit run rag_app.py
"""

import os
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from groq import Groq
from pypdf import PdfReader

PDF_PATH = "Talent_English_4_Keybook_compressed.pdf"  # <-- change this to your PDF's filename/path


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Reading and indexing PDF...")
def build_index(pdf_path, chunk_size=500, chunk_overlap=50):
    reader = PdfReader(pdf_path, strict=False)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text

    words = full_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    embedder = get_embedder()
    doc_embeddings = embedder.encode(chunks)
    return chunks, doc_embeddings


def retrieve(query, documents, doc_embeddings, top_k=4):
    embedder = get_embedder()
    query_embedding = embedder.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [documents[i] for i in top_indices]


def generate_answer(query, context_chunks):
    client = Groq(api_key=st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", "")))
    context_text = "\n".join(f"- {c}" for c in context_chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know.

Context:
{context_text}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="📚")
st.title("📚 RAG Chatbot")
st.caption(f"Answering questions from: {PDF_PATH}")

if "messages" not in st.session_state:
    st.session_state.messages = []

documents, doc_embeddings = build_index(PDF_PATH)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about the PDF...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context_chunks = retrieve(question, documents, doc_embeddings)
            answer = generate_answer(question, context_chunks)
            st.markdown(answer)
            with st.expander("Show retrieved context"):
                for c in context_chunks:
                    st.write(c)

    st.session_state.messages.append({"role": "assistant", "content": answer})
