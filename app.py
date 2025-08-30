# app.py

import streamlit as st
from backend import extract_text_from_pdf, chunk_text, build_faiss_index, generate_answer

# ----------------- PAGE SETUP -----------------
st.set_page_config(page_title="📘 StudyMate PDF Q&A", layout="wide")
st.title("📘 StudyMate - PDF Q&A")

# ----------------- PDF UPLOAD -----------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    # -------- Extract text --------
    text = extract_text_from_pdf(uploaded_file)

    # -------- Split into chunks --------
    chunks = chunk_text(text)

    # -------- Build / reuse FAISS index --------
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = build_faiss_index(chunks)

    vectorstore = st.session_state.vectorstore

    # -------- Ask a Question --------
    query = st.text_input("❓ Ask a question about the PDF:")

    if query:
        response = generate_answer(query, vectorstore)
        st.subheader("✅ Answer:")
        st.write(response)
