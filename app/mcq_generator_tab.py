import streamlit as st
from langchain_openai import OpenAI
from main import extract_text_from_pdf
import io
import pandas as pd
from fpdf import FPDF

def generate_mcqs_from_text(chapter_text, openai_key, num_questions=5):
    prompt = f"""
You are an AI tutor helping NEET UG aspirants.
Generate {num_questions} NEET-style MCQs from this chapter:
{chapter_text[:2000]}

Each must have:
- 4 options (A to D)
- Correct answer
- Difficulty: Easy, Medium, or Hard

Format:
Q1. <question>
A. ...
B. ...
C. ...
D. ...
Answer: <A/B/C/D>
Difficulty: <Easy/Medium/Hard>
"""
    llm = OpenAI(openai_api_key=openai_key, temperature=0.5)
    return llm.invoke(prompt)

def create_pdf_download(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in content.splitlines():
        pdf.multi_cell(0, 10, line)
    output = io.BytesIO()
    output.write(pdf.output(dest='S').encode('latin-1'))
    output.seek(0)
    return output

def create_csv_download(content):
    questions = []
    current = {}
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("Q") and "." in line:
            if current:
                questions.append(current)
            current = {"question": line}
        elif line.startswith("A."):
            current["option_a"] = line[2:].strip()
        elif line.startswith("B."):
            current["option_b"] = line[2:].strip()
        elif line.startswith("C."):
            current["option_c"] = line[2:].strip()
        elif line.startswith("D."):
            current["option_d"] = line[2:].strip()
        elif line.startswith("Answer:"):
            current["answer"] = line.replace("Answer:", "").strip()
        elif line.startswith("Difficulty:"):
            current["difficulty"] = line.replace("Difficulty:", "").strip()
    if current:
        questions.append(current)
    df = pd.DataFrame(questions)
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return output

def show_mcq_generator_tab(openai_key):
    st.header("📝 Generate NEET-style MCQs")

    chapter_pdf = st.file_uploader("📄 Upload Chapter PDF", type="pdf", key="mcq_chapter_pdf")
    num_questions = st.selectbox("📌 Number of MCQs to generate", options=[3, 5, 10], index=1)

    if chapter_pdf:
        if st.button("🧠 Generate MCQs"):
            with st.spinner("Generating questions..."):
                chapter_text = extract_text_from_pdf(chapter_pdf)
                result = generate_mcqs_from_text(chapter_text, openai_key, num_questions)

                st.success(f"Here are {num_questions} NEET-style MCQs with difficulty tags:")
                st.markdown(f"""```text\n{result}```""")

                st.download_button("⬇️ Download as PDF", create_pdf_download(result), file_name="mcqs.pdf")
                st.download_button("⬇️ Download as CSV", create_csv_download(result), file_name="mcqs.csv")
    else:
        st.info("Please upload a chapter PDF to generate MCQs.")
