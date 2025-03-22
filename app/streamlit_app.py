import os
import streamlit as st
import pdfplumber
import numpy as np
import faiss
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain.chains import RetrievalQA

from predict_neet_tab import show_predict_neet_tab
from mcq_generator_tab import show_mcq_generator_tab

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Utility Functions
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def split_text(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
    return splitter.split_text(text)

def build_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return LangchainFAISS.from_texts(text_chunks, embedding=embeddings), embeddings

def answer_question(query, vector_store, embeddings):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa.invoke(query)
    return result['result'] if isinstance(result, dict) and 'result' in result else result

# Main PDF Q&A UI
def show_pdf_qa_tab():
    st.header("📘 Ask Your PDF")
    uploaded_file = st.file_uploader("📤 Upload a PDF file", type="pdf")

    if uploaded_file:
        with st.spinner("Reading and indexing your PDF..."):
            text = extract_text_from_pdf(uploaded_file)
            chunks = split_text(text)
            vector_store, embeddings = build_vector_store(chunks)
            st.success("PDF processed successfully!")

        question = st.text_input("❓ Ask a question from your document")

        if question:
            with st.spinner("Thinking..."):
                answer = answer_question(question, vector_store, embeddings)
                st.markdown("### 🧠 Answer:")
                st.write({"query": question, "result": answer})

# Streamlit App Tabs
st.set_page_config(page_title="📄 AI-Powered PDF App", layout="centered")
st.title("📄 AI-Powered Document Assistant")

tab1, tab2, tab3 = st.tabs(["📘 Ask Your PDF", "🧠 Predict NEET Questions", "📝 Generate MCQs"])

with tab1:
    show_pdf_qa_tab()

with tab2:
    show_predict_neet_tab()

with tab3:
    show_mcq_generator_tab()
