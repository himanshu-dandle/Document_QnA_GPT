import streamlit as st
from langchain_openai import OpenAI
from main import extract_text_from_pdf
import os
from dotenv import load_dotenv
import io
from fpdf import FPDF
import random

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Token-safe character limits
CHAPTER_LIMIT = 2500  # ~625 tokens
PAST_LIMIT = 1200     # ~300 tokens

# Function to generate predicted descriptive questions
def generate_predicted_questions(chapter_text, past_questions_text):
    chapter_trimmed = chapter_text[:CHAPTER_LIMIT].strip()
    past_trimmed = past_questions_text[:PAST_LIMIT].strip()

    variation = random.choice([
        "Give conceptual and numerical mix.",
        "Focus on derivations and tricky MCQs.",
        "Emphasize frequently repeated patterns."
    ])

    prompt = f"""
You are a senior NEET Physics paper setter with access to past trends and syllabus knowledge.

Using the chapter content and past NEET questions provided below, create **5 diverse and high-probability NEET UG Physics questions** for this year. Prioritize conceptual clarity, frequently repeated topics, derivations, real-world applications, and numericals.

### Chapter Content:
{chapter_trimmed}

### Past NEET Questions:
{past_trimmed}

{variation}

🧠 Format:
1. Descriptive question 1
2. Descriptive question 2
...

Ensure no duplication and do not repeat past questions verbatim.
"""

    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7, max_tokens=700)
    return llm.invoke(prompt)


# Function to generate MCQs
def generate_mcqs_from_combined_text(chapter_text, past_questions_text, num_questions=5):
    chapter_trimmed = chapter_text[:CHAPTER_LIMIT].strip()
    past_trimmed = past_questions_text[:PAST_LIMIT].strip()

    prompt = f"""
You're an AI tutor generating NEET-style multiple choice questions from the given chapter and past NEET papers.

Using the inputs below, generate **{num_questions} NEET-style MCQs**. Each should include:
- A conceptual or numerical question
- 4 answer options labeled A–D
- The correct answer
- A difficulty tag (Easy, Medium, or Hard)

### Chapter Summary:
{chapter_trimmed}

### Past NEET Physics Questions:
{past_trimmed}

📘 Format:
Q1. Question text?
A. Option A
B. Option B
C. Option C
D. Option D
Answer: B
Difficulty: Medium
"""

    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7, max_tokens=700)
    return llm.invoke(prompt)


# Export Utility: PDF
def create_pdf_download(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in content.splitlines():
        clean_line = line.encode("latin-1", errors="ignore").decode("latin-1")
        pdf.multi_cell(0, 10, clean_line)
    pdf_output = io.BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)
    return pdf_output


# Streamlit Tab UI
def show_predict_neet_tab():
    st.header("🤑 Predict NEET Questions")

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
