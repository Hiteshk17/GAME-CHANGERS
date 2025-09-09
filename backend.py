# backend.py

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
import os

# ✅ Gemini SDK
import google.generativeai as genai


def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def chunk_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


def build_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(chunks, embeddings)


def load_llm():
    """Load Gemini model and cache in session_state."""
    if "llm" not in st.session_state:
        # 🔹 Get API key from environment variable, fallback to hardcoded
        api_key = os.getenv(
            "GEMINI_API_KEY",
            "your API key"  # fallback if not set
        )
        genai.configure(api_key=api_key)
        st.session_state.llm = genai.GenerativeModel("gemini-1.5-flash-latest")
    return st.session_state.llm


def generate_answer(query, vectorstore):
    llm = load_llm()

    # 🔹 Retrieve context from PDF
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    # 🔹 Call Gemini model
    response = llm.generate_content(prompt)
    return response.text
