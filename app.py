# app.py

import streamlit as st
from backend import extract_text_from_pdf, chunk_text, build_faiss_index, generate_answer

st.set_page_config(page_title="📘 StudyMate", layout="wide")

st.title("📘 StudyMate: AI-Powered PDF Q&A Assistant")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Step 1: Extract text
        text = extract_text_from_pdf(uploaded_file)

        # Step 2: Chunk text
        chunks = chunk_text(text)

        # Step 3: Build FAISS index
        vectorstore = build_faiss_index(chunks)

        st.success("✅ PDF processed successfully!")

        # User query input
        query = st.text_input("Ask a question from your PDF:")

        if query:
            with st.spinner("Generating answer..."):
                answer = generate_answer(query, vectorstore)

            st.subheader("📖 Answer")
            st.write(answer)
