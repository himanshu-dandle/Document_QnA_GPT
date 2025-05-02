import streamlit as st
from langchain_openai import ChatOpenAI
from main import extract_text_from_pdf
import io
from fpdf import FPDF
import random
import hashlib

PAST_LIMIT = 4000

FEW_SHOT_EXAMPLES = """
Q1. A capacitor of 5 μF is charged to a potential difference of 200 V. What is the energy stored?
A. 0.1 J
B. 0.05 J
C. 0.25 J
D. 0.5 J
Answer: A
Difficulty: Medium

Q2. The phenomenon responsible for the twinkling of stars is:
A. Reflection
B. Refraction
C. Scattering
D. Total internal reflection
Answer: B
Difficulty: Easy
"""

seen_question_hashes_past = set()

def hash_question_block(mcq_block):
    return hashlib.md5(mcq_block.strip().encode()).hexdigest()

def generate_mcqs_from_past_only(
    past_questions_text, openai_key, num_questions=25, difficulty_filter="All"
):
    difficulty_instruction = ""
    if difficulty_filter != "All":
        difficulty_instruction = f"\n- Focus ONLY on {difficulty_filter.upper()} difficulty questions."

    prompt = f"""
📝 **ROLE:** You are part of a **top NEET UG 2025 question paper committee** (Physics/Chemistry/Biology).

📅 **Exam:** NEET UG 2025 will be held on **4 May 2025.**

🚩 **Objective:** Predict {num_questions} **NEET-style MCQs** based STRICTLY on past NEET exam trends (provided below).

✅ **Strict Rules:**
- Study the question patterns, topics & complexity from past papers.
- Rephrase and innovate to predict fresh but realistic questions.
- NO direct repetition; each question must look fresh yet aligned with NEET's standards.
- Include:  
   - Options (A/B/C/D)  
   - Correct answer at the end  
   - Difficulty tag (Easy, Medium, Hard)  
- Ensure excellent topic coverage (no bias).  
- Aim for more **Medium and Hard** questions to make it realistic.{difficulty_instruction}

💡 **Example format:**
{FEW_SHOT_EXAMPLES}

### 🗂 Past NEET Questions:
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
        key="past_num_questions_select"
    )

    difficulty_filter = st.selectbox(
        "🎯 Focus on Difficulty Level",
        ["All", "Easy", "Medium", "Hard"],
        index=0,
        key="past_difficulty_select"
    )

    past_papers_pdfs = st.file_uploader(
        "📄 Upload Past NEET Question Papers (1 or more)",
        type="pdf",
        key="predict_past_papers_upload",
        accept_multiple_files=True
    )

    if past_papers_pdfs:
        if st.button("🔮 Generate MCQs from Past Papers", key="past_generate_button"):
            with st.spinner("Analyzing past papers and predicting questions..."):
                all_past_texts = [extract_text_from_pdf(pdf) for pdf in past_papers_pdfs]
                past_questions_text = "\n".join(all_past_texts)

                mcqs = generate_mcqs_from_past_only(
                    past_questions_text,
                    openai_key,
                    num_questions=num_questions,
                    difficulty_filter=difficulty_filter
                )

                st.success(f"Here are {num_questions} NEET-style MCQs predicted!")
                st.markdown(f"""```text\n{mcqs}```""")
                st.download_button(
                    "⬇️ Download MCQs PDF",
                    create_pdf_download(mcqs),
                    file_name="mcqs_from_past.pdf",
                    key="past_download_button"
                )
    else:
        st.info("📥 Please upload at least one NEET past paper PDF.")
