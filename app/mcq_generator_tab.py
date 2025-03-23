import streamlit as st
from langchain_openai import OpenAI
from main import extract_text_from_pdf
import io
import pandas as pd
from fpdf import FPDF

# Function to generate MCQs using GPT with difficulty levels
def generate_mcqs_from_text(chapter_text, num_questions, api_key):
    prompt = f"""
You are an AI tutor helping NEET UG aspirants.
Based on the following chapter content, generate {num_questions} NEET-style multiple-choice questions.
Each question must have:
- 4 options (A, B, C, D)
- The correct answer
- A difficulty tag: Easy, Medium, or Hard

### Chapter Content:
{chapter_text[:2000]}

Return in this format:
Q1. <question>
A. <option1>
B. <option2>
C. <option3>
D. <option4>
Answer: <Correct Option Letter>
Difficulty: <Easy/Medium/Hard>

Repeat for all questions.
"""
    llm = OpenAI(openai_api_key=api_key, temperature=0.5)
    return llm.invoke(prompt)

# Utility: Generate PDF file in memory
def create_pdf_download(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in content.splitlines():
        pdf.multi_cell(0, 10, line)
    pdf_output = io.BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)
    return pdf_output

# Utility: Generate CSV file in memory
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
    csv_output = io.BytesIO()
    df.to_csv(csv_output, index=False)
    csv_output.seek(0)
    return csv_output

# Streamlit Tab UI
def show_mcq_generator_tab(api_key):
    st.header("📝 Generate NEET-style MCQs")

    chapter_pdf = st.file_uploader("📄 Upload Chapter PDF (e.g., Laws of Motion)", type="pdf", key="mcq_chapter_pdf")
    num_questions = st.selectbox("📌 Number of MCQs to generate", options=[3, 5, 10], index=1)

    if chapter_pdf:
        if st.button("🧠 Generate MCQs"):
            with st.spinner("Generating multiple choice questions with difficulty levels..."):
                chapter_text = extract_text_from_pdf(chapter_pdf)
                result = generate_mcqs_from_text(chapter_text, num_questions, api_key)

                st.success(f"Here are {num_questions} NEET-style MCQs with difficulty tags:")
                st.markdown(f"""```text\n{result}```""")

                # Export buttons
                st.download_button("⬇️ Download as PDF", create_pdf_download(result), file_name="mcqs.pdf")
                st.download_button("⬇️ Download as CSV", create_csv_download(result), file_name="mcqs.csv")
    else:
        st.info("Please upload a chapter PDF to generate MCQs.")
