import streamlit as st
from langchain_openai import OpenAI
from main import extract_text_from_pdf
import os
from dotenv import load_dotenv
import io
from fpdf import FPDF

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to generate predicted descriptive questions
def generate_predicted_questions(chapter_text, past_questions_text):
    prompt = f"""
You are an expert NEET question paper analyst.

Based on the chapter content below (from Laws of Motion) and the past NEET Physics questions:

### Chapter Content:
{chapter_text[:1500]}

### Past Questions:
{past_questions_text[:1500]}

Now, generate 5 high-probability NEET questions for this year on this chapter.
Focus on trends, concepts asked repeatedly, derivations, and common question types.
"""

    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.3)
    return llm.invoke(prompt)

# Function to generate MCQs
def generate_mcqs_from_combined_text(chapter_text, past_questions_text, num_questions=5):
    prompt = f"""
You are an AI tutor generating NEET-style multiple choice questions.

Using the content below:

### Chapter:
{chapter_text[:1500]}

### Past Questions:
{past_questions_text[:1500]}

Generate {num_questions} NEET-style multiple choice questions.
Each should include a question, 4 options labeled A–D, and the correct answer.

Format:
Q1. Question?
A. Option 1
B. Option 2
C. Option 3
D. Option 4
Answer: A
"""

    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.5)
    return llm.invoke(prompt)

# Export Utility: PDF

def create_pdf_download(content, filename="predicted_questions.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in content.splitlines():
        pdf.multi_cell(0, 10, line)
    pdf_output = io.BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin-1')  # get PDF as bytes string
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)
    return pdf_output

# Streamlit Tab UI
def show_predict_neet_tab():
    st.header("🧠 Predict NEET Questions")

    chapter_pdf = st.file_uploader("📄 Upload Chapter PDF (e.g., Laws of Motion)", type="pdf", key="predict_chapter")
    past_papers_pdf = st.file_uploader("📄 Upload Past NEET Questions PDF", type="pdf", key="predict_papers")

    if chapter_pdf and past_papers_pdf:
        if st.button("🔮 Generate Predicted Questions + MCQs"):
            with st.spinner("Analyzing and generating questions..."):
                chapter_text = extract_text_from_pdf(chapter_pdf)
                past_questions_text = extract_text_from_pdf(past_papers_pdf)

                result = generate_predicted_questions(chapter_text, past_questions_text)
                mcqs = generate_mcqs_from_combined_text(chapter_text, past_questions_text)

                st.success("Here are 5 high-probability NEET UG questions:")
                st.markdown(f"""```text\n{result}```""")
                st.download_button("⬇️ Download Predictions as PDF", create_pdf_download(result), file_name="predicted_questions.pdf")

                st.success("Here are 5 NEET-style MCQs:")
                st.markdown(f"""```text\n{mcqs}```""")
                st.download_button("⬇️ Download MCQs as PDF", create_pdf_download(mcqs), file_name="mcqs.pdf")
    else:
        st.info("Please upload both the chapter and past question paper PDFs.")
