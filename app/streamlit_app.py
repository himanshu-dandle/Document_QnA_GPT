import os
import streamlit as st
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain.chains import RetrievalQA

from predict_from_past_tab import show_predict_from_past_tab
from predict_neet_tab import show_predict_neet_tab
from mcq_generator_tab import show_mcq_generator_tab

# ✅ MUST be first Streamlit command
st.set_page_config(page_title="📄 AI-Powered NEET Assistant", layout="centered")

# ✅ Get OpenAI API Key from secrets or input
if "OPENAI_API_KEY" not in st.session_state:
    if "OPENAI_API_KEY" in st.secrets:
        st.session_state["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    else:
        st.session_state["OPENAI_API_KEY"] = st.text_input("🔑 Enter your OpenAI API Key", type="password")

OPENAI_API_KEY = st.session_state.get("OPENAI_API_KEY", "")

# Stop execution if key not provided
if not OPENAI_API_KEY:
    st.warning("Please enter your OpenAI API key to use the app.")
    st.stop()

# Tab Layout
st.title("📄 AI-Powered NEET Assistant")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Predict from Past Papers",
    "📘 Ask Your PDF",
    "🧠 Predict NEET Questions",
    "📝 Generate MCQs"
])

with tab1:
    show_predict_from_past_tab(OPENAI_API_KEY)

with tab2:
    # You already had this tab
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

with tab3:
    show_predict_neet_tab(OPENAI_API_KEY)

with tab4:
    show_mcq_generator_tab(OPENAI_API_KEY)
