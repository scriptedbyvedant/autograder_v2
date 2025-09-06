import streamlit as st
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import fitz  # PyMuPDF
import re
import json
from typing import Dict, List

# 🔗 RAG store
from grader_engine.rag_integration import VS

# --- AUTH CHECK ---
if "logged_in_prof" not in st.session_state:
    st.warning("Please login first to access this page.", icon="🔒")
    st.stop()
prof = st.session_state["logged_in_prof"]

# -------------- PDF UTILITIES --------------
def extract_text_from_pdf(pdf_file) -> str:
    pdf_file.seek(0)
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        return "\n".join(page.get_text("text") for page in doc)

def parse_professor_pdf(text: str) -> Dict:
    prof_info = {}
    patterns = {
        "professor":      r"(Professor(?:in)?):\s*(.+)",
        "course":         r"(Course|Kurs):\s*(.+)",
        "session":        r"(Session|Sitzung):\s*(.+)",
        "assignment_no":  r"(Assignment(?: No)?|Aufgabe(?:n)?(?: Nr)?):\s*([^\n]+)"
    }
    for key, pat in patterns.items():
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            prof_info[key] = match.group(2).strip()

    # Split by Q1: / Aufgabe 1: etc.
    blocks = re.split(r'(?m)^(Q\d+:|Aufgabe\s*\d+:)', text)
    questions: List[Dict] = []
    for i in range(1, len(blocks), 2):
        qid = blocks[i].rstrip(':').strip().replace("Aufgabe", "Q")
        block = blocks[i+1]
        q_match = re.search(
            r'(Question|Frage):\s*(.*?)(?=(Ideal Answer|Ideale Antwort):)',
            block, re.S | re.IGNORECASE
        )
        question = q_match.group(2).strip() if q_match else ""
        ia_match = re.search(
            r'(Ideal Answer|Ideale Antwort):\s*(.*?)(?=(Rubric|Bewertungskriterien):)',
            block, re.S | re.IGNORECASE
        )
        ideal_answer = ia_match.group(2).strip() if ia_match else ""
        rubric_list: List[Dict[str, int]] = []
        rubric_text = ""
        r_match = re.search(
            r'(Rubric|Bewertungskriterien):\s*(.*)', block, re.S | re.IGNORECASE
        )
        if r_match:
            rubric_text = r_match.group(2).strip()
            for line in rubric_text.splitlines():
                line = line.strip()
                if not line.startswith('-'):
                    continue
                m_pts = re.match(
                    r'-\s*(?P<crit>.+?)\s*\(\s*(?P<pts>\d+)\s*(?:points?|pts?|pt|Punkte?)\s*\)',
                    line, re.IGNORECASE
                )
                if m_pts:
                    rubric_list.append({
                        "criteria": m_pts.group('crit').strip(),
                        "points":   int(m_pts.group('pts'))
                    })
                else:
                    crit_only = line.lstrip('-').strip()
                    rubric_list.append({"criteria": crit_only, "points": 0})
        questions.append({
            "id":           qid,
            "question":     question,
            "ideal_answer": ideal_answer,
            "rubric":       rubric_list,
            "rubric_text":  rubric_text,
        })
    prof_info["questions"] = questions
    return prof_info

def parse_student_pdf(text: str) -> Dict[str, Dict[str, str]]:
    students: Dict[str, Dict[str, str]] = {}
    parts = re.split(r'(?m)^(Student\s*\d+:|Studierende[rn]?\s*\d+:)', text)
    for i in range(1, len(parts), 2):
        student_label = parts[i].rstrip(':').strip()
        block = parts[i+1]
        answers: Dict[str, str] = {}
        for m in re.finditer(
            r'(A\d+|F\d+):\s*(.*?)(?=\nA\d+:|\nF\d+:|\nStudent\s*\d+:|\nStudierende[rn]?\s*\d+:|$)',
            block, re.S
        ):
            answers[m.group(1)] = m.group(2).strip()
        students[student_label] = answers
    return students

# --- RAG seeding helper ---
def seed_rag_from_professor(prof_info: Dict):
    """
    Push rubric & ideal answer per question into the in-memory RAG store.
    This makes rubric/ideal discoverable during grading via retrieve_context().
    """
    for q in prof_info.get("questions", []):
        qid = q.get("id")
        if not qid:
            continue
        # Add rubric JSON
        rubric_obj = q.get("rubric") or []
        if rubric_obj:
            VS.add(f"{qid}-rubric", json.dumps(rubric_obj, ensure_ascii=False), {"type": "rubric", "q_id": qid})
        # Add ideal answer
        ideal = (q.get("ideal_answer") or "").strip()
        if ideal:
            VS.add(f"{qid}-ideal", ideal, {"type": "ideal", "q_id": qid})
        # (Optional) also add the question text to help fallback lexical search
        qt = (q.get("question") or "").strip()
        if qt:
            VS.add(f"{qid}-question", qt, {"type": "question", "q_id": qid})

# --- Sample PDF Generators (unchanged) ---
def make_professor_sample_pdf():
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = 770; lh = 18
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Professor Assignment PDF Format Example")
    c.setFont("Helvetica", 11); y -= lh*2
    lines = [
        "Professor: Dr. Smith",
        "Course: AI Fundamentals",
        "Session: Summer 2025",
        "Assignment No: 2",
        "",
        "Q1:",
        "Question: Explain supervised vs. unsupervised learning.",
        "Ideal Answer: Supervised learning uses labeled data without labels.",
        "Rubric:",
        "- Correct definition of supervised learning (2 points)",
        "- Correct definition of unsupervised learning (2 points)",
        "- Example for each (2 points)",
        "",
        "Q2:",
        "Question: What is a neural network?",
        "Ideal Answer: A neural network is ...",
        "Rubric:",
        "- Defines a neural network (2 points)",
        "- Gives a real-world application (1 point)"
    ]
    for line in lines:
        c.drawString(50, y, line); y -= lh
    c.save(); buf.seek(0)
    return buf

def make_student_sample_pdf():
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = 770; lh = 18
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Student Answers PDF Format Example")
    c.setFont("Helvetica", 11); y -= lh*2
    lines = [
        "Student 1:",
        "A1: Supervised learning uses labeled data...",
        "A2: A neural network is a system of algorithms...",
        "",
        "Student 2:",
        "A1: Supervised learning is where...",
        "A2: ..."
    ]
    for line in lines:
        c.drawString(50, y, line); y -= lh
    c.save(); buf.seek(0)
    return buf

# ----------- STREAMLIT PAGE LOGIC ------------
st.set_page_config(page_title="Upload PDFs for Grading", layout="wide")
st.title("📄 Upload Assignment PDFs for Grading")
st.markdown("""
Upload your assignment PDFs and ensure they follow the required structure below.  
Select grading language, then start grading.
""")

# 1. Upload Section
st.header("1. Upload Files")
col1, col2 = st.columns(2)
with col1:
    prof_pdf = st.file_uploader(
        "📋 Professor PDF (Questions & Rubric)", 
        type=["pdf"], key="prof_pdf")
    st.download_button(
        "⬇️ Download Sample Professor PDF",
        data=make_professor_sample_pdf(),
        file_name="professor_sample.pdf",
        mime="application/pdf"
    )
with col2:
    student_pdf = st.file_uploader(
        "📝 Student Answers PDF", 
        type=["pdf"], key="student_pdf")
    st.download_button(
        "⬇️ Download Sample Student PDF",
        data=make_student_sample_pdf(),
        file_name="student_sample.pdf",
        mime="application/pdf"
    )

# 2. Language and Grading Button
st.header("2. Grading Options")
language = st.selectbox("Grading Language", ["English", "German", "Spanish"])
st.session_state["answer_language"] = language

can_grade = prof_pdf and student_pdf
grade_btn = st.button("🚦 Start Grading", disabled=not can_grade)
if not can_grade:
    st.info("Please upload both files to enable grading.")

if grade_btn and can_grade:
    try:
        # extract & parse
        prof_text    = extract_text_from_pdf(prof_pdf)
        student_text = extract_text_from_pdf(student_pdf)
        prof_info    = parse_professor_pdf(prof_text)
        students_data= parse_student_pdf(student_text)

        # --- VALIDATION ---
        errors = []

        # required prof fields
        for fld in ("professor","course","session","assignment_no"):
            if not prof_info.get(fld):
                errors.append(f"Missing `{fld}` in professor PDF.")

        # each question block must have question, ideal_answer, rubric
        for q in prof_info.get("questions", []):
            if not q["question"]:
                errors.append(f"{q['id']}: missing Question text.")
            if not q["ideal_answer"]:
                errors.append(f"{q['id']}: missing Ideal Answer.")
            if not q["rubric"]:
                errors.append(f"{q['id']}: missing or empty Rubric items.")

        # ensure student answers cover all questions
        qids = [q["id"] for q in prof_info.get("questions",[])]
        for student_label, ans in students_data.items():
            for qid in qids:
                akey = qid.replace("Q","A")
                if akey not in ans or not ans[akey].strip():
                    errors.append(f"{student_label}: missing answer `{akey}` for {qid}.")

        if errors:
            st.error("❌ PDF format validation failed:")
            for e in errors:
                st.write(f"• {e}")
            st.stop()  # halt before grading

        # --- ✅ RAG SEEDING (rubric + ideal per question) ---
        seed_rag_from_professor(prof_info)

        # --- proceed on success ---
        st.session_state["prof_data"]      = prof_info
        st.session_state["students_data"]  = students_data
        st.session_state["answer_language"]= language
        st.success("✅ Files validated & parsed! Redirecting…")
        st.switch_page("pages/2_grading_result.py")

    except Exception as e:
        st.error(f"❌ Error parsing PDF: {e}")

# 3. PDF Format Guide
st.markdown("---")
st.header("Required PDF Format & Fields")
with st.expander("📘 Click to see required format for PDF uploads (Professor & Student)", expanded=False):
    st.markdown("""
    <style>
    .format-table td, .format-table th { padding: 7px 18px; border-bottom: 1px solid #e4e8f0; font-size: 15px; }
    .format-table th { background-color: #f4f7fb; font-weight: 700; color: #133467; }
    .format-table { border-collapse: collapse; min-width: 440px; margin-bottom: 22px; margin-top: 0px; }
    .sample-block { background: #f9fafb; border-radius: 8px; padding: 18px; font-family: monospace; font-size: 15px;
                    color: #234; margin-bottom: 18px; margin-top: 2px; border-left: 4px solid #477ddb; box-shadow: 0 1.5px 7px #253d6a12;
                    overflow-x: auto; white-space: pre-line; }
    </style>

    <h5 style="color:#133467; font-size:1.08em;">Professor PDF – Required Fields</h5>
    <table class="format-table">
      <tr><th>Field</th><th>Example</th><th>Required</th></tr>
      <tr><td>Professor</td><td>Dr. Smith</td><td>Yes</td></tr>
      <tr><td>Course</td><td>AI Fundamentals</td><td>Yes</td></tr>
      <tr><td>Session</td><td>Summer 2025</td><td>Yes</td></tr>
      <tr><td>Assignment No</td><td>2</td><td>Yes</td></tr>
      <tr><td>Q1, Q2…</td><td>See below</td><td>Yes</td></tr>
      <tr><td>Question</td><td>Explain supervised vs. unsupervised...</td><td>Yes</td></tr>
      <tr><td>Ideal Answer</td><td>Supervised learning uses labeled...</td><td>Yes</td></tr>
      <tr><td>Rubric</td><td>- Correct definition (2 pts)<br>- Example (1 pt)</td><td>Yes</td></tr>
    </table>

    <div class="sample-block">
    Professor: Dr. Smith
    Course: AI Fundamentals
    Session: Summer 2025
    Assignment No: 2

    Q1:
    Question: Explain supervised vs. unsupervised learning.
    Ideal Answer: Supervised learning uses labeled data...
    Rubric:
    - Correct definition (2 pts)
    - Example (1 pt)

    Q2:
    Question: What is a neural network?
    Ideal Answer: A neural network is ...
    Rubric:
    - Defines a neural network (2 pts)
    - Gives a real-world application (1 pt)
    </div>

    <h5 style="color:#197044; font-size:1.08em;">Student PDF – Required Fields</h5>
    <table class="format-table">
      <tr><th>Field</th><th>Example</th><th>Required</th></tr>
      <tr><td>Student 1:</td><td>Header for each student</td><td>Yes</td></tr>
      <tr><td>A1, A2…</td><td>Answers per question</td><td>Yes</td></tr>
    </table>

    <div class="sample-block">
    Student 1:
    A1: Supervised learning uses labeled data...
    A2: A neural network is a system of algorithms...

    Student 2:
    A1: Supervised learning is where...
    A2: ...
    </div>
    """, unsafe_allow_html=True)

st.info("""
**Important**  
– All headers and fields must match (case‐insensitive).  
– Missing any required element halts grading.  
""", icon="📘")
