# File: pages/2_grading_result.py
import streamlit as st
st.set_page_config(page_title="⚖️ Grading Results", layout="wide")

import pandas as pd
import json, re, difflib, hashlib
from typing import List, Dict, Any

from database.postgres_handler import PostgresHandler
from grader_engine.text_grader import grade_answer
from grader_engine.explainer import generate_explanation

TRANSLATIONS = {
    "English": {
        "page_title": "⚖️ Grading Results", "question": "Question", "ideal_answer": "Ideal Answer",
        "student_answer": "Student Answer", "rubric_matrix": "Rubric Matrix", "rubric_breakdown": "🧮 Rubric Breakdown",
        "feedback": "📝 Feedback", "generate_explanation": "🔍 Generate Explanation",
        "save_changes": "💾 Save Changes", "results_summary": "Results Summary",
        "detailed_view": "Detailed Grading & Editing", "share_label": "🔗 Share with Colleague",
        "share_button": "Share", "share_success": "Result shared successfully!", "share_error": "Failed to share: ",
        "no_data": "Please upload PDFs and parse them first on the Upload page.", "no_answer": "No answer provided.",
        "debug_title": "🧪 LLM Debug (Prompt & Output)", "debug_model": "Model",
        "debug_prompt": "Prompt sent to model", "debug_output": "Raw model output",
        "debug_expl_title": "🧪 Explanation LLM Debug (Prompt & Output)", "rerun": "♻️ Re-run grading"
    }
}
# ---- helpers ----
def _normalize_criteria(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _align_to_rubric(rubric_list: List[Dict[str, Any]], model_breakdown: List[Dict[str, Any]],
                     model_total: float | None = None, fuzzy_cutoff: float = 0.60) -> List[Dict[str, Any]]:
    if not rubric_list: return []
    mm, keys = {}, []
    for it in (model_breakdown or []):
        if not isinstance(it, dict): continue
        k = _normalize_criteria(it.get("criteria","")); 
        try: sc = float(it.get("score", 0))
        except: sc = 0.0
        if k: mm[k]=sc; keys.append(k)
    aligned=[]
    for r in rubric_list:
        crit=r.get("criteria",""); pts=int(r.get("points",0)); norm=_normalize_criteria(crit)
        if norm in mm: sc = mm[norm]
        else:
            close = difflib.get_close_matches(norm, keys, n=1, cutoff=fuzzy_cutoff)
            sc = mm.get(close[0], 0.0) if close else 0.0
        sc = int(max(0, min(int(round(sc)), pts)))
        aligned.append({"criteria": crit, "score": sc})
    if (not model_breakdown) and (model_total is not None):
        pts = [int(r.get("points",0)) for r in rubric_list]; tot = sum(pts) or 1
        raw = [model_total*(p/tot) for p in pts]; rounded=[int(round(x)) for x in raw]
        drift=int(round(model_total))-sum(rounded); i=0
        while drift and pts:
            if drift>0: rounded[i]=min(pts[i], rounded[i]+1); drift-=1
            else: rounded[i]=max(0, rounded[i]-1); drift+=1
            i=(i+1)%len(pts)
        aligned=[{"criteria":r.get("criteria",""), "score":int(max(0, min(sc, r.get("points",0))))}
                 for r,sc in zip(rubric_list, rounded)]
    return aligned

def _total_possible(rubric_list: List[Dict[str, Any]]) -> int:
    return sum(int(r.get("points", 0)) for r in rubric_list)

def _dedupe_feedback(text: str) -> str:
    """Remove repeated lines/sentences while preserving order."""
    lines = [l.strip() for l in re.split(r'[\n]+', text or "") if l.strip()]
    seen, out = set(), []
    for ln in lines:
        key = re.sub(r'\W+', ' ', ln.lower()).strip()
        if key not in seen:
            out.append(ln); seen.add(key)
    return "\n".join(out)

def _signature(prof_data: Dict[str, Any], students_data: Dict[str, Any], language: str) -> str:
    try:
        payload = json.dumps({"prof": prof_data, "students": students_data, "lang": language},
                             sort_keys=True, ensure_ascii=False)
    except Exception:
        payload = str((prof_data, students_data, language))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

# ---- page ----
def grading_result_page():
    if "logged_in_prof" not in st.session_state:
        st.warning("Please login first to access this page.", icon="🔒"); st.stop()
    prof = st.session_state["logged_in_prof"]; my_email = prof.get("university_email","")

    prof_data = st.session_state.get("prof_data")
    students_data = st.session_state.get("students_data")
    language = st.session_state.get("answer_language", "English")
    T = TRANSLATIONS["English"]
    if not prof_data or not students_data:
        st.warning(T["no_data"]); return

    st.title(T["page_title"])

    # cache
    if "grading_cache" not in st.session_state:
        st.session_state["grading_cache"] = {"signature": None, "results": None}
    sig = _signature(prof_data, students_data, language)

    col1, col2 = st.columns([1,1])
    with col1: st.caption(f"cache signature: `{sig[:12]}…`")
    with col2: force_rerun = st.button(T["rerun"], use_container_width=True)

    need_compute = force_rerun or (st.session_state["grading_cache"]["signature"] != sig) \
                   or (st.session_state["grading_cache"]["results"] is None)

    if need_compute:
        grading_results: Dict[str, Any] = {}
        with st.spinner(f"{T['results_summary']}..."):
            for student, answers in students_data.items():
                for q in prof_data["questions"]:
                    key = f"{student}_{q['id']}"; ans_key = q["id"].replace("Q", "A")
                    stud_ans = (answers.get(ans_key, "") or "").strip()

                    rubric_list = q.get("rubric", [])
                    if isinstance(rubric_list, str):
                        try: rubric_list = json.loads(rubric_list)
                        except: rubric_list = []

                    if not stud_ans:
                        aligned = [{"criteria": r["criteria"], "score": 0} for r in rubric_list]
                        feedback_txt = T["no_answer"]; llm_debug = None
                    else:
                        out = grade_answer(
                            question=q["question"], ideal_answer=q["ideal_answer"], rubric=rubric_list,
                            student_answer=stud_ans, language=language,
                            rag_context=None, return_debug=True,
                            include_header_in_feedback=False
                        )
                        aligned = _align_to_rubric(
                            rubric_list=rubric_list,
                            model_breakdown=out.get("rubric_scores", []),
                            model_total=out.get("total_score", None),
                            fuzzy_cutoff=0.60
                        )
                        feedback_txt = _dedupe_feedback(out.get("feedback", "") or "")
                        llm_debug = out.get("debug")

                    rubric_scores = [
                        {"criteria": r["criteria"], "score": float(aligned[i]["score"]),
                         "original_score": float(aligned[i]["score"])}
                        for i, r in enumerate(rubric_list)
                    ]
                    total_old = sum(item["original_score"] for item in rubric_scores)
                    total_new = sum(item["score"] for item in rubric_scores)

                    # ONE insert per compute only
                    result_id = PostgresHandler().insert_or_update_grading_result(
                        student_id=student,
                        professor_id=prof_data.get("professor",""),
                        course=prof_data.get("course",""),
                        semester=prof_data.get("session",""),
                        assignment_no=prof_data.get("assignment_no",""),
                        question=q["question"],
                        student_answer=stud_ans,
                        language=language,
                        old_score=total_old, new_score=total_new,
                        old_feedback=feedback_txt, new_feedback=feedback_txt
                    )

                    grading_results[key] = {
                        "rubric_scores": rubric_scores,
                        "feedback": {"language": language, "text": feedback_txt, "original": feedback_txt},
                        "result_id": result_id, "llm_debug": llm_debug,
                        "question": q["question"], "ideal_answer": q["ideal_answer"],
                        "rubric_list": rubric_list, "student_answer": stud_ans
                    }
        st.session_state["grading_cache"]["signature"] = sig
        st.session_state["grading_cache"]["results"] = grading_results
        st.success(f"✅ {T['results_summary']} complete!")

    grading_results = st.session_state["grading_cache"]["results"]

    # Summary
    st.subheader(T["results_summary"])
    rows=[]
    for student in students_data:
        tot_sc=tot_ps=0; row={"Student": student}
        for q in prof_data["questions"]:
            k=f"{student}_{q['id']}"
            sc=sum(int(it["score"]) for it in grading_results[k]["rubric_scores"])
            ps=sum(int(r.get("points",0)) for r in q.get("rubric",[]))
            row[q["id"]]=f"{sc}/{ps}"; tot_sc+=sc; tot_ps+=ps
        row["Total"]=f"{tot_sc}/{tot_ps}"; rows.append(row)
    st.table(pd.DataFrame(rows))

    # Detailed
    st.subheader(T["detailed_view"])
    students = list(students_data.keys())
    questions = [q["id"] for q in prof_data["questions"]]
    sel_st = st.selectbox("Student", students, key="detail_student")
    sel_q  = st.selectbox("Question", questions, key="detail_question")
    qobj = next(q for q in prof_data["questions"] if q["id"] == sel_q)
    detail_key = f"{sel_st}_{sel_q}"
    stored = grading_results[detail_key]

    st.markdown(f"**{T['rubric_matrix']}:**")
    rm = pd.DataFrame(qobj.get("rubric", []))
    if not rm.empty: rm = rm[["criteria","points"]]
    st.table(rm)

    stud_ans = students_data[sel_st].get(sel_q.replace("Q","A"), "")
    st.markdown(f"**{T['question']}:** {stored['question']}")
    st.markdown(f"**{T['ideal_answer']}:** {stored['ideal_answer']}")
    st.markdown(f"**{T['student_answer']} ({language}):**")
    if len(stud_ans) > 400: st.text_area("Student Answer", stud_ans, height=200)
    else: st.write(stud_ans)

    if stored.get("llm_debug"):
        with st.expander(T["debug_title"], expanded=False):
            dbg = stored["llm_debug"]
            st.markdown(f"**{T['debug_model']}:** `{dbg.get('model','')}`")

            # A) Post-validated result actually used by the UI
            post_total = sum(int(it["score"]) for it in stored["rubric_scores"])
            st.markdown("**Post-validated result used by the UI**")
            st.code(json.dumps({
                "total_score": post_total,
                "rubric_scores": stored["rubric_scores"],
                "feedback_preview": (stored["feedback"]["text"][:300] + "…") if stored["feedback"]["text"] else ""
            }, ensure_ascii=False, indent=2), language="json")

            # B) Sanity checks from the engine (mismatches, clamps, type coercions)
            st.markdown("**Sanity checks**")
            st.code(json.dumps(dbg.get("sanity", {}), ensure_ascii=False, indent=2), language="json")

            # C) Prompt and raw model output for reference
            st.markdown(f"**{T['debug_prompt']}:**")
            st.code(dbg.get("prompt",""), language="markdown")
            st.markdown(f"**{T['debug_output']}:**")
            st.code(dbg.get("raw_output",""), language="json")


    st.markdown(T["rubric_breakdown"])
    total_now = sum(int(x["score"]) for x in stored["rubric_scores"])
    total_possible = _total_possible(qobj.get("rubric", []))
    st.info(f"**Total: {total_now}/{total_possible}**")

    for idx, crit in enumerate(qobj.get("rubric", [])):
        init = int(stored["rubric_scores"][idx]["score"])
        max_pts = int(crit.get("points",0))
        val = st.slider(f"{crit['criteria']} (0–{max_pts})", 0, max_pts, init, key=f"slider_{detail_key}_{idx}")
        st.session_state["grading_cache"]["results"][detail_key]["rubric_scores"][idx]["score"] = int(val)

    st.markdown("📋 **Aligned table (details)**")
    aligned_table = pd.DataFrame([
        {"Criteria": r["criteria"],
         "Score": int(st.session_state["grading_cache"]["results"][detail_key]["rubric_scores"][i]["score"]),
         "Points": int(qobj["rubric"][i]["points"])}
        for i, r in enumerate(qobj["rubric"])
    ])
    st.dataframe(aligned_table, hide_index=True, use_container_width=True)

    fb = st.text_area(T["feedback"], value=stored["feedback"]["text"], key=f"fb_{detail_key}")
    st.session_state["grading_cache"]["results"][detail_key]["feedback"]["text"] = _dedupe_feedback(fb)

    if st.button(T["save_changes"], key=f"save_{detail_key}"):
        new_sc = sum(int(item["score"]) for item in st.session_state["grading_cache"]["results"][detail_key]["rubric_scores"])
        PostgresHandler().insert_or_update_grading_result(
            student_id=sel_st,
            professor_id=prof_data.get("professor",""),
            course=prof_data.get("course",""),
            semester=prof_data.get("session",""),
            assignment_no=prof_data.get("assignment_no",""),
            question=stored["question"],
            student_answer=stud_ans,
            language=language,
            old_score=sum(int(item["original_score"]) for item in stored["rubric_scores"]),
            new_score=new_sc,
            old_feedback=stored["feedback"]["original"],
            new_feedback=st.session_state["grading_cache"]["results"][detail_key]["feedback"]["text"]
        )
        st.success("✅ Saved!")

    st.markdown(f"---\n**{T['share_label']}:**")
    target = st.text_input("Professor email", key=f"share_email_{detail_key}")
    if st.button(T["share_button"], key=f"share_btn_{detail_key}"):
        if target:
            try:
                PostgresHandler().share_result(my_email, target, stored["result_id"])
                st.success(T["share_success"])
            except Exception as e:
                st.error(T["share_error"] + str(e))
        else:
            st.error("Enter a valid email to share.")

    if st.button(T["generate_explanation"], key=f"genexp_{detail_key}"):
        with st.spinner(f"{T['generate_explanation']}..."):
            expl, dbg = generate_explanation(
                question=stored["question"], ideal_answer=stored["ideal_answer"], rubric=stored["rubric_list"],
                student_answer=stud_ans, assigned_score=sum(int(item["score"]) for item in st.session_state["grading_cache"]["results"][detail_key]["rubric_scores"]),
                language=language, return_debug=True
            )
        st.subheader(T["generate_explanation"]); st.write(_dedupe_feedback(expl))
        with st.expander(T["debug_expl_title"], expanded=False):
            st.markdown(f"**{T['debug_model']}:** `{dbg.get('model','')}`")
            st.markdown(f"**{T['debug_prompt']}:**"); st.code(dbg.get("prompt",""), language="markdown")
            st.markdown(f"**{T['debug_output']}:**"); st.code(dbg.get("raw_output",""), language="json")

grading_result_page()
