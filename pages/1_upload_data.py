
import streamlit as st
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import re
import json
from typing import Dict, List, Any

# --- Custom Package Imports ---
from grader_engine.pdf_parser_multimodal import extract_multimodal_content_from_pdf
from grader_engine.multimodal_rag import MultimodalVectorStore
from ilias_utils.zip_parser import parse_ilias_zip, IngestResult, StudentFolder
from ilias_utils.feedback_generator import FeedbackZipGenerator

# --- AUTH CHECK ---
if "logged_in_prof" not in st.session_state:
    st.warning("Please login first to access this page.", icon="🔒")
    st.stop()
prof = st.session_state["logged_in_prof"]

# --- PROFESSOR PDF PARSING ---
def parse_professor_pdf(text: str) -> Dict:
    prof_info = {}
    patterns = {
        "professor": r"(Professor(?:in)?):\s*(.+)",
        "course": r"(Course|Kurs):\s*(.+)",
        "session": r"(Session|Sitzung):\s*(.+)",
        "assignment_no": r"(Assignment(?: No)?|Aufgabe(?:n)?(?: Nr)?):\s*([^\n]+)"
    }
    for key, pat in patterns.items():
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            prof_info[key] = match.group(2).strip()

    blocks = re.split(r'(?m)^(Q\d+:|Aufgabe\s*\d+:)', text)
    questions: List[Dict] = []
    for i in range(1, len(blocks), 2):
        qid = blocks[i].rstrip(':').strip().replace("Aufgabe", "Q")
        block = blocks[i+1]
        q_match = re.search(r'(Question|Frage):\s*(.*?)(?=(Ideal Answer|Ideale Antwort):)', block, re.S | re.IGNORECASE)
        question = q_match.group(2).strip() if q_match else ""
        ia_match = re.search(r'(Ideal Answer|Ideale Antwort):\s*(.*?)(?=(Rubric|Bewertungskriterien):)', block, re.S | re.IGNORECASE)
        ideal_answer = ia_match.group(2).strip() if ia_match else ""
        rubric_list: List[Dict[str, int]] = []
        r_match = re.search(r'(Rubric|Bewertungskriterien):\s*(.*)', block, re.S | re.IGNORECASE)
        if r_match:
            rubric_text = r_match.group(2).strip()
            for line in rubric_text.splitlines():
                if not line.strip().startswith('-'): continue
                m_pts = re.match(r'-\s*(?P<crit>.+?)\s*\(\s*(?P<pts>\d+)\s*(?:points?|pts?|pt|Punkte?)\s*\)', line, re.IGNORECASE)
                if m_pts:
                    rubric_list.append({"criteria": m_pts.group('crit').strip(), "points": int(m_pts.group('pts'))})
                else:
                    rubric_list.append({"criteria": line.lstrip('-').strip(), "points": 0})
        questions.append({"id": qid, "question": question, "ideal_answer": ideal_answer, "rubric": rubric_list})
    prof_info["questions"] = questions
    return prof_info

# --- MULTIMODAL STUDENT DATA PROCESSING ---
def process_student_data(content_blocks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parses student answers from a list of multimodal content blocks.
    This function is designed to be robust against various answer formats.
    """
    full_text = "\n".join(block.get("content", "") for block in content_blocks if block.get("type") == "text").strip()
    if not full_text:
        return {}

    # Regex to find all possible answer markers (e.g., Q1, A1, Answer 1, Aufgabe 1) at the start of a line.
    answer_marker_regex = re.compile(r"(?im)^\s*(?:Q|A|Answer|Aufgabe)\s*(\d+)\s*[:.)]?")

    # Find all markers and their positions in the text.
    markers = list(answer_marker_regex.finditer(full_text))
    
    answers: Dict[str, List[Dict[str, Any]]] = {}
    
    # Iterate through the found markers to split the text into answers.
    for i, match in enumerate(markers):
        q_num = match.group(1)
        answer_key = f"A{q_num}"  # Standardize the key to "A<number>"

        # Determine the start and end of the answer content.
        start_pos = match.end()
        end_pos = markers[i + 1].start() if i + 1 < len(markers) else len(full_text)
        
        answer_content = full_text[start_pos:end_pos].strip()

        if answer_content:
            # Wrap the extracted text in the expected multimodal format.
            answers[answer_key] = [{"type": "text", "content": answer_content}]
            
    return answers

# --- RAG SEEDING HELPER ---
def seed_multimodal_rag_from_professor(prof_info: Dict, prof_content_blocks: List[Dict], vs_instance: MultimodalVectorStore):
    vs_instance.index.reset()
    vs_instance.items.clear()
    vs_instance.by_q.clear()
    for i, block in enumerate(prof_content_blocks):
        doc_id = f"prof-block-{i}"
        for q in prof_info.get("questions", []):
            if qid := q.get("id"):
                vs_instance.add(doc_id, block['content'], block['type'], {"q_id": qid, "source": "professor"})
    for q in prof_info.get("questions", []):
        if not (qid := q.get("id")): 
            continue
        if rubric := q.get("rubric"):
            vs_instance.add(f"{qid}-rubric", json.dumps(rubric, ensure_ascii=False), "text", {"type": "rubric", "q_id": qid})
        if ideal := (q.get("ideal_answer") or "").strip():
            vs_instance.add(f"{qid}-ideal", ideal, "text", {"type": "ideal", "q_id": qid})

# --- UI & MAIN LOGIC ---
st.set_page_config(page_title="Upload Data for Grading", layout="wide")
st.title("📄 Upload Assignment Data for Grading")

prof_pdf = st.file_uploader("📋 Professor PDF", type=["pdf"])
submission_file = st.file_uploader("📝 Student Submissions (PDF or ILIAS ZIP)", type=["pdf", "zip"])
language = st.selectbox("Grading Language", ["English", "German", "Spanish"])
st.session_state["answer_language"] = language

if st.button("🚦 Start Grading", disabled=not (prof_pdf and submission_file), type="primary"):
    multimodal_vs_instance = None
    try:
        if 'multimodal_vs' not in st.session_state:
            with st.spinner("Initializing Multimodal Vector Store..."):
                st.session_state['multimodal_vs'] = MultimodalVectorStore()
        multimodal_vs_instance = st.session_state['multimodal_vs']
    except (IOError, ValueError) as e:
        st.warning(f"⚠️ Could not initialize embedding model: {e}. The RAG context retrieval feature will be disabled. Grading will proceed without it.", icon="⚠️")
        st.session_state['multimodal_vs'] = None # Ensure it's None in session state

    try:
        with st.spinner("Processing Professor PDF..."):
            pdf_bytes = prof_pdf.getvalue()
            prof_content_blocks = extract_multimodal_content_from_pdf(BytesIO(pdf_bytes))
            prof_text = "\n".join([b['content'] for b in prof_content_blocks if b['type'] == 'text'])
            prof_info = parse_professor_pdf(prof_text)
            if not prof_info.get("questions"): st.error("Professor PDF missing Q1:, etc."); st.stop()
            st.session_state["prof_data"] = prof_info
            # Only seed RAG if the vector store was successfully initialized
            if multimodal_vs_instance:
                seed_multimodal_rag_from_professor(prof_info, prof_content_blocks, multimodal_vs_instance)
                st.write("✅ Professor PDF processed and RAG seeded.")
            else:
                st.write("✅ Professor PDF processed (RAG feature disabled).")

    except Exception as e: st.error(f"❌ Error processing Professor PDF: {e}"); st.stop()

    students_data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    errors = []
    try:
        with st.spinner(f"Processing student submissions..."):
            if submission_file.name.lower().endswith(".zip"):
                st.session_state["upload_type"] = "ilias_zip"
                ingest_result: IngestResult = parse_ilias_zip(
                    BytesIO(submission_file.getvalue()),
                    multimodal_extractor=extract_multimodal_content_from_pdf
                )
                st.session_state["ilias_ingest_result"] = ingest_result
                for student_folder in ingest_result.student_folders:
                    student_id = student_folder.email or student_folder.raw_folder
                    all_student_blocks = [block for f in student_folder.files for block in f.multimodal_content]
                    students_data[student_id] = process_student_data(all_student_blocks)
            else: # PDF
                st.session_state["upload_type"] = "pdf"
                all_content_blocks = extract_multimodal_content_from_pdf(BytesIO(submission_file.getvalue()))
                students_data["Student 1"] = process_student_data(all_content_blocks)

        qids = [q["id"] for q in prof_info.get("questions", [])]
        for student_id, answers in students_data.items():
            for qid in qids:
                if (akey := qid.replace("Q", "A")) not in answers or not answers[akey]:
                    errors.append(f"**{student_id}** missing content for `{qid}` (`{akey}`).")

        if errors: st.error("❌ Validation failed:"); st.write("\n".join(f"• {e}" for e in errors)); st.stop()
        if not students_data: st.error("No student data extracted."); st.stop()

        st.session_state["students_data"] = students_data
        st.success(f"✅ Processed {len(students_data)} submissions. Redirecting...")
        st.switch_page("pages/2_grading_result.py")

    except Exception as e: st.error(f"❌ Error processing submissions: {e}"); st.exception(e)
