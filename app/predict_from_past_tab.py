import streamlit as st
from langchain_openai import ChatOpenAI
from main import extract_text_from_pdf
import io
from fpdf import FPDF
import random
import hashlib

PAST_LIMIT = 4000

FEW_SHOT_EXAMPLES = """
Q1. A ball is thrown vertically upwards with a velocity of 20 m/s. What is its maximum height?
A. 10 m
B. 15 m
C. 20 m
D. 25 m
Answer: C
Difficulty: Medium

Q2. Which law states that energy can neither be created nor destroyed?
A. Newton's First Law
B. Law of Conservation of Energy
C. Coulomb's Law
D. Ohm's Law
Answer: B
Difficulty: Easy
"""

seen_question_hashes_past = set()


def hash_question_block(mcq_block):
    return hashlib.md5(mcq_block.strip().encode()).hexdigest()


def generate_mcqs_from_past_only(
    past_questions_text, openai_key, num_questions=25
):
    prompt = f"""
You are a very expert senior NEET UG examination paper setter who can predict the questions  for year 2025 which will held on 4-may-2025

Generate {num_questions} **high-quality NEET-style MCQs only** based **only on past NEET papers** provided below.

Guidelines:
- Predict future-style questions smartly based only on past papers.
- Ensure diversity across different topics.
- Include a mix of Easy, Medium, and Hard questions however focus more on hard and medium 
- Add Difficulty level tag for each question.
- Avoid copying questions exactly; rephrase and innovate.

Format:
{FEW_SHOT_EXAMPLES}

### Past NEET Questions:
{past_questions_text[:PAST_LIMIT]}
"""

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_key)
    response = llm.invoke(prompt)
    result = response.content if hasattr(response, "content") else str(response)

    all_mcqs_list = []
    question_counter = 1
    for block in result.strip().split("\n\n"):
        h = hash_question_block(block)
        if h not in seen_question_hashes_past:
            seen_question_hashes_past.add(h)
            numbered = f"Q{question_counter}. " + block.strip()
            all_mcqs_list.append(numbered)
            question_counter += 1

    return "\n\n".join(all_mcqs_list)


def create_pdf_download(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in content.splitlines():
        pdf.multi_cell(0, 10, line.encode("latin-1", errors="ignore").decode("latin-1"))
    buffer = io.BytesIO()
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    buffer.write(pdf_bytes)
    buffer.seek(0)
    return buffer


def show_predict_from_past_tab(openai_key):
    st.header("📚 Predict NEET MCQs from Past Papers Only")

    num_questions = st.selectbox(
        "🔹 Number of MCQs to Generate",
        [5, 10, 20, 25, 30, 50],
        index=3,
        key="past_tab_num_questions"
    )
    past_papers_pdfs = st.file_uploader(
        "📄 Upload Past NEET Question Papers (1 or more)",
        type="pdf",
        key="predict_past_papers_upload",
        accept_multiple_files=True
    )

    if past_papers_pdfs:
        if st.button("🔮 Generate MCQs from Past Papers", key="past_tab_generate_button"):
            with st.spinner("Analyzing past papers and predicting questions..."):
                all_past_texts = [extract_text_from_pdf(pdf) for pdf in past_papers_pdfs]
                past_questions_text = "\n".join(all_past_texts)

                mcqs = generate_mcqs_from_past_only(
                    past_questions_text,
                    openai_key,
                    num_questions=num_questions
                )

                st.success(f"Here are {num_questions} NEET-style MCQs predicted!")
                st.markdown(f"""```text\n{mcqs}```""")
                st.download_button(
                    "⬇️ Download MCQs PDF",
                    create_pdf_download(mcqs),
                    file_name="mcqs_from_past.pdf",
                    key="past_tab_download_button"
                )
    else:
        st.info("📥 Please upload at least one NEET past paper PDF.")
