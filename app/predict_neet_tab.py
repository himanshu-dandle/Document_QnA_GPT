import streamlit as st
from langchain_openai import ChatOpenAI
from main import extract_text_from_pdf
import io
from fpdf import FPDF
import random
import hashlib

CHAPTER_LIMIT = 4000
PAST_LIMIT = 3000

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

# ✅ Initialize session variables safely
if "chapter_text_map" not in st.session_state:
    st.session_state.chapter_text_map = {}

if "used_chapter_names" not in st.session_state:
    st.session_state.used_chapter_names = set()

if "past_questions_text" not in st.session_state:
    st.session_state.past_questions_text = ""

seen_question_hashes = set()

def hash_question_block(mcq_block):
    return hashlib.md5(mcq_block.strip().encode()).hexdigest()

def generate_mcqs_from_combined_text(
    chapter_chunks, past_questions_text, openai_key,
    num_questions=25, exclude_logic=False, question_type="Mixed", difficulty_filter="All"
):
    if len(chapter_chunks) == 0:
        return "❌ No chapters left or selected. Please upload new PDFs or pick different chapters."

    questions_per_chunk = max(1, num_questions // len(chapter_chunks))
    all_mcqs_list = []
    question_counter = 1
    random.shuffle(chapter_chunks)

    for chapter_name, chunk in chapter_chunks:
        chunk_trimmed = chunk[:CHAPTER_LIMIT].strip()
        past_trimmed = past_questions_text[:PAST_LIMIT].strip()

        logic_instruction = "\n- Do not include questions related to logic gates or digital electronics." if exclude_logic else ""
        type_instruction = f"\n- Only include {question_type.lower()} questions." if question_type != "Mixed" else ""
        difficulty_instruction = f"\n- Only include {difficulty_filter.lower()} difficulty questions." if difficulty_filter != "All" else ""

        prompt = f"""
You are a senior NEET Physics paper setter with expertise in recent NEET trends (2023-24).

Generate {questions_per_chunk} NEET-style MCQs only (no descriptive), with:
- Conceptual and numerical balance (~50%)
- Follow latest NEET patterns{logic_instruction}{type_instruction}{difficulty_instruction}
- Add difficulty tag (Easy/Medium/Hard)
- Tag chapter: {chapter_name}
- Do not repeat common old questions.

Format:
{FEW_SHOT_EXAMPLES}

### Chapter:
{chunk_trimmed}

### Past NEET Questions:
{past_trimmed}
"""
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_key)
        response = llm.invoke(prompt)
        result = response.content if hasattr(response, "content") else str(response)

        for block in result.strip().split("\n\n"):
            h = hash_question_block(block)
            if h not in seen_question_hashes:
                seen_question_hashes.add(h)
                numbered = f"Q{question_counter}. " + block.strip()
                all_mcqs_list.append(f"[Chapter: {chapter_name}]\n{numbered}")
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

def show_predict_neet_tab(openai_key):
    st.header("🚙 Predict NEET MCQs Only")

    subject = st.selectbox("🧪 Select Subject", ["Physics", "Chemistry", "Biology"])
    question_type = st.radio("⚙️ Question Type", ["Mixed", "Conceptual Only", "Numerical Only"])
    difficulty = st.selectbox("🎯 Focus on Difficulty Level", ["All", "Easy", "Medium", "Hard"])
    num_questions = st.selectbox("🔹 Number of MCQs to Generate", [5, 10, 20, 25, 30, 50], index=3)
    exclude_logic = st.toggle("🚫 Exclude Logic/Digital Electronics Questions", value=False)

    chapter_pdfs = st.file_uploader("📄 Upload Chapter PDFs", type="pdf", key="predict_chapter", accept_multiple_files=True)
    past_papers_pdfs = st.file_uploader("📄 Upload Past NEET Papers", type="pdf", key="predict_papers", accept_multiple_files=True)

    # ✅ Read chapters into session
    if chapter_pdfs:
        for pdf in chapter_pdfs:
            name = pdf.name.replace(".pdf", "")
            if name not in st.session_state.chapter_text_map:
                st.session_state.chapter_text_map[name] = extract_text_from_pdf(pdf)

    # ✅ Read past questions once
    if past_papers_pdfs and not st.session_state.past_questions_text:
        all_past_texts = [extract_text_from_pdf(pdf) for pdf in past_papers_pdfs]
        st.session_state.past_questions_text = "\n".join(all_past_texts)

    available_chapters = [c for c in st.session_state.chapter_text_map if c not in st.session_state.used_chapter_names]

    if available_chapters:
        selected_chapters = st.multiselect("📌 Select Chapter(s) to focus", available_chapters, default=available_chapters)
    else:
        selected_chapters = []

    if st.button("🔮 Generate NEET MCQs"):
        if not selected_chapters:
            st.error("⚠️ No chapters selected or available. Upload more or reset.")
            return

        selected_chunks = [(name, st.session_state.chapter_text_map[name]) for name in selected_chapters]
        for name in selected_chapters:
            st.session_state.used_chapter_names.add(name)

        mcqs = generate_mcqs_from_combined_text(
            selected_chunks,
            st.session_state.past_questions_text,
            openai_key,
            num_questions=num_questions,
            exclude_logic=exclude_logic,
            question_type=question_type,
            difficulty_filter=difficulty
        )

        st.success(f"✅ {num_questions} NEET MCQs generated!")
        st.markdown(f"""```text\n{mcqs}```""")
        st.download_button("⬇️ Download MCQs PDF", create_pdf_download(mcqs), file_name="mcqs.pdf")

    if st.button("🔁 Reset Used Chapters"):
        st.session_state.used_chapter_names.clear()
        st.success("✅ Chapters reset! You can reuse all uploaded chapters.")
