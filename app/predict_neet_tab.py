import streamlit as st
from langchain_openai import ChatOpenAI
from main import extract_text_from_pdf
import io
from fpdf import FPDF
import random

CHAPTER_LIMIT = 6000
PAST_LIMIT = 4000

FEW_SHOT_EXAMPLES = """
Q1. A block of mass 2kg is placed on a frictionless surface. If a force of 10N is applied, what is its acceleration?
A. 5 m/s^2
B. 10 m/s^2
C. 2 m/s^2
D. 20 m/s^2
Answer: A
Difficulty: Medium

Q2. Which law of motion defines the relationship F = ma?
A. First Law
B. Second Law
C. Third Law
D. Newton’s Universal Law
Answer: B
Difficulty: Easy
"""

def generate_predicted_questions(chapter_text, past_questions_text, subject, openai_key):
    chapter_trimmed = chapter_text[:CHAPTER_LIMIT].strip()
    past_trimmed = past_questions_text[:PAST_LIMIT].strip()
    variation = random.choice([
        "Give conceptual and numerical mix.",
        "Focus on derivations and tricky MCQs.",
        "Emphasize frequently repeated patterns."
    ])
    prompt = f"""
You are a senior NEET {subject} paper setter with deep insight into past trends and syllabus.

Using the chapter content and past NEET questions provided below, generate **5 varied and high-probability NEET UG questions**.

### Chapter Content:
{chapter_trimmed}

### Past NEET Questions:
{past_trimmed}

{variation}

🧠 Format:
1. Descriptive question 1
2. Descriptive question 2
... (and so on)

Avoid copying old questions exactly.
"""
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.5, openai_api_key=openai_key)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)

def generate_mcqs_from_combined_text(chapter_text, past_questions_text, openai_key, num_questions=5):
    chapter_trimmed = chapter_text[:CHAPTER_LIMIT].strip()
    past_trimmed = past_questions_text[:PAST_LIMIT].strip()
    prompt = f"""
You're an AI tutor generating NEET-style MCQs from the given chapter and past NEET papers.

Use this format:
{FEW_SHOT_EXAMPLES}

Now generate {num_questions} new questions.

### Chapter:
{chapter_trimmed}

### Past NEET Questions:
{past_trimmed}
"""
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7, openai_api_key=openai_key)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)

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

def show_predict_neet_tab(openai_key):
    st.header("🤑 Predict NEET Questions")

    subject = st.selectbox("🧪 Select Subject", ["Physics", "Chemistry", "Biology"])
    chapter_pdf = st.file_uploader("📄 Upload Chapter PDF", type="pdf", key="predict_chapter")
    past_papers_pdf = st.file_uploader("📄 Upload Past NEET Papers", type="pdf", key="predict_papers")

    if chapter_pdf and past_papers_pdf:
        if st.button("🔮 Generate Predicted Questions + MCQs"):
            with st.spinner("Analyzing and generating questions..."):
                chapter_text = extract_text_from_pdf(chapter_pdf)
                past_questions_text = extract_text_from_pdf(past_papers_pdf)

                result = generate_predicted_questions(chapter_text, past_questions_text, subject, openai_key)
                mcqs = generate_mcqs_from_combined_text(chapter_text, past_questions_text, openai_key)

                st.success("Here are 5 high-probability NEET UG questions:")
                st.markdown(f"""```text\n{result}```""")
                st.download_button("⬇️ Download Predictions as PDF", create_pdf_download(result), file_name="predicted_questions.pdf")

                st.success("Here are 5 NEET-style MCQs:")
                st.markdown(f"""```text\n{mcqs}```""")
                st.download_button("⬇️ Download MCQs as PDF", create_pdf_download(mcqs), file_name="mcqs.pdf")
    else:
        st.info("Please upload both the chapter and past question paper PDFs.")
