
# File: pages/2_grading_result.py
import streamlit as st
st.set_page_config(page_title="⚖️ Grading Results", layout="wide")

import pandas as pd
import json, re, difflib, hashlib, io, zipfile, time
from typing import List, Dict, Any
from PIL import Image

from database.postgres_handler import PostgresHandler
from grader_engine.multimodal_grader import grade_answer_multimodal as grade_answer
from grader_engine.code_grader import grade_code
from grader_engine.multimodal_rag import retrieve_multimodal_context
from ilias_utils.zip_parser import IngestResult
from ilias_utils.pdf_feedback import FeedbackPDFGenerator

# --- TRANSLATIONS ---
TRANSLATIONS = {
    "English": {
        "page_title": "⚖️ Grading Results", "question": "Question", "ideal_answer": "Ideal Answer",
        "student_answer": "Student Answer", "rubric_breakdown": "🧮 Rubric Breakdown",
        "feedback": "📝 Feedback", "save_changes": "💾 Save Changes", "results_summary": "Results Summary",
        "detailed_view": "Detailed Grading & Editing",
        "no_data": "Please upload data on the Upload page first.", "no_answer": "No answer provided.",
        "debug_title": "🧪 LLM Debug", "retrieved_context_title": "Retrieved Context",
        "export_button": "📦 Download All Feedback", "share_expander": "🔗 Share Result with a Colleague",
        "share_email_input": "Enter colleague\'s email address:", "share_button": "Share",
        "share_success": "Result shared successfully!", "share_error": "Failed to share result.",
        "invalid_email": "Please enter a valid email address.",
        "code_feedback_tests": "Passed {passed} of {total} test cases.",
        "code_feedback_failures_header": "\n**Failed Tests:**",
        "code_feedback_failure_item": "- Input: `{input}`\n  - Expected: `{expected}`\n  - Got: `{got}`",
        "code_feedback_blank": "The submission was empty.",
        "code_feedback_invalid": "The submission was not valid Python code and could not be executed.",
        "code_feedback_runtime_error": "The code was syntactically correct but failed to run. Error: `{error}`",
        "code_feedback_generic_fail": "Code evaluation failed. Reason: {reason}"
    },
    "German": {
        "page_title": "⚖️ Benotungsergebnisse", "question": "Frage", "ideal_answer": "Ideale Antwort",
        "student_answer": "Antwort des Studierenden", "rubric_breakdown": "🧮 Notenschlüssel-Aufschlüsselung",
        "feedback": "📝 Feedback", "save_changes": "💾 Änderungen speichern", "results_summary": "Ergebnisübersicht",
        "detailed_view": "Detaillierte Benotung & Bearbeitung",
        "no_data": "Bitte laden Sie zuerst Daten auf der Upload-Seite hoch.", "no_answer": "Keine Antwort abgegeben.",
        "debug_title": "🧪 LLM Debug", "retrieved_context_title": "Abgerufener Kontext",
        "export_button": "📦 Alle Feedbacks herunterladen", "share_expander": "🔗 Ergebnis mit einem Kollegen teilen",
        "share_email_input": "E-Mail-Adresse des Kollegen eingeben:", "share_button": "Teilen",
        "share_success": "Ergebnis erfolgreich geteilt!", "share_error": "Fehler beim Teilen des Ergebnisses.",
        "invalid_email": "Bitte geben Sie eine gültige E-Mail-Adresse ein.",
        "code_feedback_tests": "{passed} von {total} Testfällen bestanden.",
        "code_feedback_failures_header": "\n**Fehlgeschlagene Tests:**",
        "code_feedback_failure_item": "- Eingabe: `{input}`\n  - Erwartet: `{expected}`\n  - Erhalten: `{got}`",
        "code_feedback_blank": "Die Einreichung war leer.",
        "code_feedback_invalid": "Die Einreichung war kein gültiger Python-Code und konnte nicht ausgeführt werden.",
        "code_feedback_runtime_error": "Der Code war syntaktisch korrekt, konnte aber nicht ausgeführt werden. Fehler: `{error}`",
        "code_feedback_generic_fail": "Code-Bewertung fehlgeschlagen. Grund: {reason}"
    },
    "Spanish": {
        "page_title": "⚖️ Resultados de Calificación", "question": "Pregunta", "ideal_answer": "Respuesta Ideal",
        "student_answer": "Respuesta del Estudiante", "rubric_breakdown": "🧮 Desglose de la Rúbrica",
        "feedback": "📝 Comentarios", "save_changes": "💾 Guardar Cambios", "results_summary": "Resumen de Resultados",
        "detailed_view": "Calificación y Edición Detallada",
        "no_data": "Por favor, suba los datos en la página de Carga primero.", "no_answer": "No se proporcionó respuesta.",
        "debug_title": "🧪 Depuración del LLM", "retrieved_context_title": "Contexto Recuperado",
        "export_button": "📦 Descargar Todos los Comentarios", "share_expander": "🔗 Compartir Resultado con un Colega",
        "share_email_input": "Ingrese el correo electrónico del colega:", "share_button": "Compartir",
        "share_success": "¡Resultado compartido con éxito!", "share_error": "Error al compartir el resultado.",
        "invalid_email": "Por favor, ingrese una dirección de correo electrónico válida.",
        "code_feedback_tests": "{passed} de {total} casos de prueba superados.",
        "code_feedback_failures_header": "\n**Pruebas Fallidas:**",
        "code_feedback_failure_item": "- Entrada: `{input}`\n  - Esperado: `{expected}`\n  - Obtenido: `{got}`",
        "code_feedback_blank": "La entrega estaba vacía.",
        "code_feedback_invalid": "La entrega no era código Python válido y no pudo ser ejecutada.",
        "code_feedback_runtime_error": "El código era sintácticamente correcto pero falló al ejecutarse. Error: `{error}`",
        "code_feedback_generic_fail": "La evaluación del código falló. Razón: {reason}"
    }
}

# ---- HELPERS ----
def render_content_blocks(title: str, content_blocks: List[Dict[str, Any]]):
    st.markdown(f"**{title}**")
    if not content_blocks:
        st.info("No content provided for this section.")
        return
    for item in content_blocks:
        content_type = item.get('type') or item.get('content_type')
        content = item.get('content')
        if content_type == 'text':
            st.text_area("Text", value=content, height=max(100, len(content)//3), disabled=True, label_visibility="collapsed")
        elif content_type == 'image':
            try: st.image(Image.open(io.BytesIO(content)), use_column_width=True)
            except Exception as e: st.warning(f"Could not display image: {e}")

def _normalize_criteria(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _align_to_rubric(rubric_list: List[Dict], model_breakdown: List[Dict], model_total: int, fuzzy_cutoff: float=0.6) -> List[Dict]:
    if not rubric_list: return []
    mm, keys = {}, []
    for it in (model_breakdown or []):
        k = _normalize_criteria(it.get("criteria", ""))
        try: sc = int(round(float(it.get("score", 0))))
        except (ValueError, TypeError): sc = 0
        if k: mm[k]=sc; keys.append(k)
    aligned = []
    for r in rubric_list:
        crit, pts = r.get("criteria", ""), int(r.get("points", 0)); norm = _normalize_criteria(crit)
        if norm in mm: sc = mm[norm]
        else:
            close = difflib.get_close_matches(norm, keys, n=1, cutoff=fuzzy_cutoff)
            sc = mm.get(close[0], 0) if close else 0
        aligned.append({"criteria": crit, "score": max(0, min(sc, pts))})
    return aligned

def _total_possible(rubric_list: List[Dict]) -> int:
    return sum(int(r.get("points", 0)) for r in rubric_list)

def _dedupe_feedback(text: str) -> str:
    lines = [l.strip() for l in re.split(r'[\n]+', text or "") if l.strip()]
    seen, out = set(), []
    for ln in lines:
        key = re.sub(r'\W+', ' ', ln.lower()).strip()
        if key not in seen: out.append(ln); seen.add(key)
    return "\n".join(out)

def _make_serializable(data: Any) -> Any:
    if isinstance(data, dict): return {k: _make_serializable(v) for k, v in data.items()}
    if isinstance(data, list): return [_make_serializable(v) for v in data]
    if isinstance(data, bytes): return f"<bytes_hash:{hashlib.sha256(data).hexdigest()}>"
    return data

def _signature(prof_data: Dict, students_data: Dict, language: str) -> str:
    payload = json.dumps({"prof": prof_data, "students": _make_serializable(students_data), "lang": language}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def update_score_callback(student, q_id, rubric_idx, slider_key):
    if slider_key in st.session_state:
        st.session_state["grading_cache"]['results'][f"{student}_{q_id}"]['rubric_scores'][rubric_idx]['score'] = st.session_state[slider_key]

# ---- MAIN PAGE LOGIC ----
def grading_result_page():
    if "logged_in_prof" not in st.session_state: st.warning("Please login first."); st.stop()
    prof = st.session_state["logged_in_prof"]; my_email = prof.get("university_email","")

    try: db = PostgresHandler()
    except Exception as e: st.error(f"Database connection failed: {e}"); st.stop()

    prof_data = st.session_state.get("prof_data"); students_data = st.session_state.get("students_data")
    language = st.session_state.get("answer_language", "English")
    T = TRANSLATIONS.get(language, TRANSLATIONS["English"])
    if not prof_data or not students_data: st.warning(T["no_data"]); return

    st.title(T["page_title"])

    sig = _signature(prof_data, students_data, language)
    if st.session_state.get("grading_cache", {}).get("signature") != sig:
        grading_results = {}
        total_tasks = len(students_data) * len(prof_data["questions"])
        progress_bar = st.progress(0, text=f"Preparing to grade {total_tasks} answers...")
        tasks_done = 0

        with st.status("Running AI grading...", expanded=True) as status:
            for i, (student_id, student_answers) in enumerate(students_data.items()):
                status.write(f"---**Processing student: {student_id}** ({i+1} of {len(students_data)})")

                for j, q in enumerate(prof_data["questions"]):
                    tasks_done += 1
                    progress_bar.progress(tasks_done / total_tasks, text=f"Overall Progress: {tasks_done} of {total_tasks}")
                    
                    q_status_placeholder = status.empty()
                    key = f"{student_id}_{q['id']}"; ans_key = q["id"].replace("Q", "A")
                    stud_ans_content = student_answers.get(ans_key, [])
                    rubric_list = q.get("rubric", [])
                    
                    aligned, feedback_txt, llm_debug, ctx_items = [], T["no_answer"], None, []

                    is_code_question = "tests" in q or any(kw in q['question'].lower() for kw in ['python', 'program', 'code', 'function'])

                    if not stud_ans_content:
                        feedback_txt = T["no_answer"]
                        aligned = [{"criteria": r["criteria"], "score": 0} for r in rubric_list]
                    
                    elif is_code_question:
                        q_status_placeholder.text(f"  - Q{j+1}: 💻 Using specialized code grader...")
                        student_code = next((block.get('content', '') for block in stud_ans_content if block.get('type') == 'text'), '')
                        tests = q.get('tests', [])
                        
                        total_award, rubric_breakdown, details = grade_code(student_code=student_code, tests=tests, rubric=rubric_list)
                        aligned = rubric_breakdown
                        
                        reason = details.get("reason", "N/A")
                        if reason == "tests":
                            feedback_parts = [T["code_feedback_tests"].format(passed=details.get('passed', 0), total=details.get('total', 0))]
                            if details.get("failures"):
                                feedback_parts.append(T["code_feedback_failures_header"])
                                for failure in details["failures"][:3]: # Limit to 3 failures
                                    feedback_parts.append(T["code_feedback_failure_item"].format(
                                        input=failure.get('input','N/A'),
                                        expected=failure.get('expected','N/A'),
                                        got=failure.get('got','N/A')))
                            feedback_txt = "\n".join(feedback_parts)
                        elif reason == "blank":
                            feedback_txt = T["code_feedback_blank"]
                        elif reason == "no_tests_and_bad_code":
                            feedback_txt = T["code_feedback_invalid"]
                        elif "syntax_ok" in reason:
                            feedback_txt = T["code_feedback_runtime_error"].format(error=details.get('stderr', 'Unknown issue'))
                        else:
                            feedback_txt = T["code_feedback_generic_fail"].format(reason=reason)

                        llm_debug = {"grader": "code_grader", "details": details}
                    
                    else: # Is NOT a code question -> use LLM
                        q_status_placeholder.text(f"  - Q{j+1}: 🤖 Using language model grader...")
                        ctx_items = retrieve_multimodal_context(q_id=q['id'], question=q['question'])['context']
                        out = grade_answer(question=q["question"], ideal_answer=q["ideal_answer"], rubric=rubric_list, student_answer_blocks=stud_ans_content, multimodal_context=ctx_items, language=language, return_debug=True)
                        aligned = _align_to_rubric(rubric_list, out.get("rubric_scores",[]), out.get("total_score"))
                        feedback_txt = _dedupe_feedback(out.get("feedback", ""))
                        llm_debug = out.get("debug")
                    
                    q_status_placeholder.text(f"  - Q{j+1}: 💾 Saving result...")
                    student_answer_str = json.dumps(_make_serializable(stud_ans_content))
                    total_score = sum(s['score'] for s in aligned)

                    db_id = db.insert_or_update_grading_result(student_id=student_id, professor_id=prof_data.get("professor", my_email), course=prof_data.get("course", ""), semester=prof_data.get("session", ""), assignment_no=prof_data.get("assignment_no", ""), question=q["question"], student_answer=student_answer_str, language=language, old_score=total_score, new_score=total_score, old_feedback=feedback_txt, new_feedback=feedback_txt)
                    
                    grading_results[key] = {
                        "db_id": db_id,
                        "rubric_scores": [{"criteria": a["criteria"], "score": int(a["score"]), "original_score": int(a["score"])} for a in aligned],
                        "feedback": {"text": feedback_txt, "original": feedback_txt},
                        "llm_debug": llm_debug, "question": q["question"], "ideal_answer": q["ideal_answer"],
                        "rubric_list": rubric_list, "student_answer_content": stud_ans_content,
                        "multimodal_context_items": ctx_items
                    }
                    q_status_placeholder.text(f"  - Q{j+1}: ✅ Grading complete.")
                    time.sleep(0.1)
            
            status.update(label="✅ AI Grading Complete!", state="complete", expanded=False)

        st.session_state["grading_cache"] = {"signature": sig, "results": grading_results}
        progress_bar.empty()
        st.success("All answers have been graded and saved!")

    grading_results = st.session_state["grading_cache"]["results"]

    st.subheader(T["results_summary"])
    rows = []
    for student in students_data:
        tot_sc = tot_ps = 0
        row = {"Student": student}
        for q in prof_data["questions"]:
            k = f"{student}_{q['id']}"
            sc = sum(int(it["score"]) for it in grading_results[k]["rubric_scores"])
            ps = _total_possible(q.get("rubric",[]))
            row[q["id"]] = f"{sc}/{ps}"
            tot_sc += sc
            tot_ps += ps
        row["Total"] = f"{tot_sc}/{tot_ps}"
        rows.append(row)
    st.table(pd.DataFrame(rows).set_index("Student"))

    st.subheader(T["detailed_view"])
    sel_st = st.selectbox("Select Student", list(students_data.keys()))
    sel_q = st.selectbox("Select Question", [q['id'] for q in prof_data["questions"]])
    detail_key = f"{sel_st}_{sel_q}"
    stored = grading_results[detail_key]

    st.markdown(f"**{T['question']}:** {stored['question']}")
    render_content_blocks(T['student_answer'], stored.get('student_answer_content', []))
    with st.expander(T["retrieved_context_title"]):
        render_content_blocks("Context", stored.get("multimodal_context_items", []))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(T["rubric_breakdown"])
        for idx, crit in enumerate(stored["rubric_list"]):
            init_score = int(stored["rubric_scores"][idx]["score"])
            max_pts = int(crit.get("points", 0)); slider_key = f"slider_{detail_key}_{idx}"
            st.slider(f'{crit["criteria"]} (Points: {max_pts})', 0, max_pts, init_score, key=slider_key, on_change=update_score_callback, args=(sel_st, sel_q, idx, slider_key))
        total_now = sum(int(x["score"]) for x in stored["rubric_scores"])
        st.info(f"**Total Score: {total_now} / {_total_possible(stored['rubric_list'])}**")

    with c2:
        st.markdown(T["feedback"])
        fb_key = f"fb_{detail_key}"
        fb_text = st.text_area("Feedback", value=stored["feedback"]["text"], key=fb_key, height=300, label_visibility="collapsed")
        stored["feedback"]["text"] = _dedupe_feedback(fb_text)

    if st.button(T["save_changes"], key=f"save_{detail_key}", type="primary"):
        new_total_score = sum(item["score"] for item in stored["rubric_scores"])
        db.update_grading_result_with_correction(grading_result_id=stored['db_id'], new_score=new_total_score, new_feedback=stored["feedback"]["text"], editor_id=my_email)
        stored["feedback"]["original"] = stored["feedback"]["text"]
        for item in stored["rubric_scores"]: item["original_score"] = item["score"]
        st.success("✅ Changes saved!")

    with st.expander(T["share_expander"]):
        share_email = st.text_input(T["share_email_input"], key=f"share_email_{detail_key}")
        if st.button(T["share_button"], key=f"share_button_{detail_key}"):
            if share_email and "@" in share_email:
                try:
                    db.share_result(owner_email=my_email, target_email=share_email, result_id=stored['db_id'])
                    st.success(T["share_success"])
                except Exception as e: st.error(f"{T['share_error']} Error: {e}")
            else: st.warning(T["invalid_email"])

    # --- ZIP EXPORT ---
    st.markdown("---")
    st.subheader("Download Feedback")
    ilias_ingest: IngestResult = st.session_state.get("ilias_ingest_result")
    assignment_name = (ilias_ingest.assignment_name if ilias_ingest else prof_data.get("assignment_no", "feedback")) or "feedback"

    if st.button(T["export_button"], type="primary"):
        with st.spinner("Generating PDF feedback for all students..."):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for student_id in students_data.keys():
                    student_grading_data = []
                    student_total_score = 0
                    student_total_possible = 0

                    for q in prof_data["questions"]:
                        key = f"{student_id}_{q['id']}"
                        result = grading_results.get(key, {})
                        student_grading_data.append(result)
                        student_total_score += sum(r.get("score", 0) for r in result.get("rubric_scores", []))
                        student_total_possible += _total_possible(q.get("rubric", []))

                    # Generate the PDF for the student
                    pdf_bytes = FeedbackPDFGenerator.create_pdf(
                        student_id=student_id,
                        assignment_name=assignment_name,
                        grading_data=student_grading_data,
                        total_score=student_total_score,
                        total_possible=student_total_possible,
                        T=T
                    )
                    
                    safe_student_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', student_id)
                    file_path_in_zip = f"{safe_student_id}/feedback_report.pdf"
                    zip_file.writestr(file_path_in_zip, pdf_bytes.getvalue())

            st.session_state["feedback_zip_buffer"] = zip_buffer.getvalue()
            st.success("Generated feedback zip with PDF reports.")

    if "feedback_zip_buffer" in st.session_state:
        st.download_button(
            label="Download Feedback.zip",
            data=st.session_state["feedback_zip_buffer"],
            file_name=f"{re.sub(r'[^a-zA-Z0-9_.-]', '_', assignment_name)}_feedback.zip",
            mime="application/zip",
            on_click=lambda: st.session_state.pop("feedback_zip_buffer", None)
        )

grading_result_page()
