# File: grader_engine/text_grader.py

import os
import json
import re
import difflib
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.chat_models import ChatOllama

load_dotenv()

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
# Ensure the Ollama service is running

# -----------------------------------------------------------------------------
# LLM OUTPUT SCHEMA (advisory; we still post-validate)
# -----------------------------------------------------------------------------
response_schemas = [
    ResponseSchema(name="total_score",   description="Sum of rubric points awarded (integer)"),
    ResponseSchema(name="rubric_scores", description="List of objects: {'criteria': string, 'score': integer}"),
    ResponseSchema(name="feedback",      description="Textual feedback aligned with rubric")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# -----------------------------------------------------------------------------
# PROMPT
# -----------------------------------------------------------------------------
BASE_TEMPLATE = """
You are a strict grader. Grade the student's answer strictly by the provided rubric.
Respond in {language}.

Question:
{question}

Ideal Answer:
{ideal_answer}

Rubric (JSON list of {{'criteria','points'}}):
{rubric_json}

Student Answer:
{student_answer}

{exemplar_block}

Instructions:
- For each rubric criterion, assign an INTEGER score between 0 and its 'points' (INCLUSIVE).
- Do NOT invent or add criteria; use EXACTLY the criteria names from the rubric list.
- The 'rubric_scores' array MUST have the SAME length and the SAME criteria (same order is preferred).
- 'total_score' MUST equal the sum of the rubric item scores.
- Provide concise feedback that justifies deductions in {language}.

Respond ONLY with valid JSON matching:
{format_instructions}
"""

def _make_prompt(
    language: str,
    question: str,
    ideal_answer: str,
    rubric_json: str,
    student_answer: str,
    rag_context: Optional[Dict[str, Any]] = None
) -> str:
    """Compose the final prompt; include exemplars if provided via RAG."""
    exemplars_txt = ""
    if rag_context:
        ex = rag_context.get("exemplars", []) or []
        if ex:
            snips = []
            for i, item in enumerate(ex[:3], 1):
                txt = str(item.get("text", ""))[:700]
                meta = item.get("meta", {})
                score = meta.get("score", "")
                snips.append(f"Exemplar {i} (score {score}):\n{txt}")
            exemplars_txt = "Context (consistency reference):\n" + "\n\n".join(snips)
        if not ideal_answer and rag_context.get("ideal"):
            ideal_answer = rag_context["ideal"]

    tmpl = PromptTemplate(
        template=BASE_TEMPLATE,
        input_variables=[
            "language", "question", "ideal_answer",
            "rubric_json", "student_answer", "exemplar_block"
        ],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    return tmpl.format(
        language=language,
        question=question,
        ideal_answer=ideal_answer,
        rubric_json=rubric_json,
        student_answer=student_answer,
        exemplar_block=exemplars_txt
    )

# -----------------------------------------------------------------------------
# ROBUST PARSING + ALIGNMENT HELPERS
# -----------------------------------------------------------------------------
def _extract_json(raw: str) -> str:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    return m.group(0) if m else cleaned

def _as_int(x, default: int = 0) -> int:
    try:
        if isinstance(x, (int, float)):
            return int(round(x))
        if isinstance(x, str):
            return int(round(float(x.strip())))
    except Exception:
        pass
    return int(default)

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _align_and_clamp(
    rubric: List[Dict[str, Any]],
    model_breakdown: List[Dict[str, Any]],
    fuzzy_cutoff: float = 0.60
) -> (List[Dict[str, Any]], Dict[str, Any]):
    """
    Align model breakdown to rubric order, clamp scores to [0, points].
    Returns (aligned_list, sanity_report).
    """
    sanity = {
        "unknown_criteria": [],
        "over_allocated": [],
        "coerced_types": False,
        "model_items_seen": 0
    }
    if not rubric:
        return [], sanity

    mm: Dict[str, int] = {}
    keys: List[str] = []
    for it in (model_breakdown or []):
        if not isinstance(it, dict):
            continue
        c = it.get("criteria", "")
        s_raw = it.get("score", 0)
        s = _as_int(s_raw, 0)
        if s != s_raw:
            sanity["coerced_types"] = True
        k = _normalize(c)
        if k:
            mm[k] = s
            keys.append(k)
            sanity["model_items_seen"] += 1

    aligned: List[Dict[str, Any]] = []
    for r in rubric:
        crit = r.get("criteria", "")
        pts  = _as_int(r.get("points", 0), 0)
        norm = _normalize(crit)
        if norm in mm:
            sc = mm[norm]
        else:
            match = difflib.get_close_matches(norm, keys, n=1, cutoff=fuzzy_cutoff)
            sc = mm.get(match[0], 0) if match else 0
            if not match:
                sanity["unknown_criteria"].append(crit)

        if sc > pts:
            sanity["over_allocated"].append({"criteria": crit, "score": sc, "max": pts})
        sc = max(0, min(sc, pts))
        aligned.append({"criteria": crit, "score": sc})

    return aligned, sanity

def _feedback_header(rubric: List[Dict[str, Any]], aligned: List[Dict[str, Any]], total: int) -> str:
    total_possible = sum(_as_int(r.get("points", 0)) for r in rubric)
    lines = [f"**Total: {int(total)}/{total_possible}**", "Rubric Breakdown:"]
    for r, a in zip(rubric, aligned):
        lines.append(f"- {r.get('criteria','')}: {int(a.get('score',0))}/{_as_int(r.get('points',0))}")
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------
def grade_answer(
    question: str,
    ideal_answer: str,
    rubric: List[Dict[str, Any]],
    student_answer: str,
    language: str = "English",
    model_name: Optional[str] = None,
    rag_context: Optional[Dict[str, Any]] = None,
    return_debug: bool = False
) -> Dict[str, Any]:
    """
    Grade a single answer via Ollama and post-validate the structure.
    Returns:
      - total_score: int (recomputed from aligned, clamped scores)
      - rubric_scores: [{'criteria','score'}] aligned to rubric
      - feedback: synchronized header + model feedback
      - debug: {model, prompt, raw_output, sanity} if return_debug=True
    """
    model_id = model_name or DEFAULT_MODEL

    # Normalize rubric
    if isinstance(rubric, str):
        try:
            rubric = json.loads(rubric)
        except Exception:
            rubric = []
    rubric = rubric if isinstance(rubric, list) else []
    rubric_json = json.dumps(rubric, ensure_ascii=False)

    # Build prompt
    prompt_str = _make_prompt(
        language=language,
        question=question,
        ideal_answer=ideal_answer,
        rubric_json=rubric_json,
        student_answer=student_answer,
        rag_context=rag_context
    )

    # Call Ollama
    llm = ChatOllama(model=model_id)
    raw = ""
    try:
        resp = llm.invoke(prompt_str)
        raw = resp.content
    except Exception as e:
        out = {
            "total_score":   0,
            "rubric_scores": [{"criteria": r.get("criteria",""), "score": 0} for r in rubric],
            "feedback":      f"Grading failed: {e}"
        }
        if return_debug:
            out["debug"] = {
                "model": model_id,
                "prompt": prompt_str,
                "raw_output": str(e),
                "sanity": {"error": "invoke_failed"}
            }
        return out

    # Parse JSON
    parsed: Dict[str, Any] = {}
    try:
        parsed = output_parser.parse(raw)
    except Exception:
        try:
            parsed = json.loads(_extract_json(raw))
        except Exception:
            parsed = {}

    # Extract model fields
    model_total = _as_int(parsed.get("total_score", 0), 0)
    model_breakdown = parsed.get("rubric_scores", []) or []
    model_feedback = parsed.get("feedback", "")
    if not isinstance(model_feedback, str):
        model_feedback = str(model_feedback)

    # Align + clamp + sanity report
    aligned, sanity = _align_and_clamp(rubric, model_breakdown)

    # Recompute total from aligned (source of truth)
    total_awarded = int(sum(int(it["score"]) for it in aligned))
    if model_total != total_awarded:
        sanity["model_total"] = model_total
        sanity["recomputed_total"] = total_awarded

    # Synchronized feedback
    header = _feedback_header(rubric, aligned, total_awarded)
    body = model_feedback.strip()
    full_feedback = header + ("\n\n" + body if body else "")

    result = {
        "total_score":   total_awarded,
        "rubric_scores": aligned,
        "feedback":      full_feedback
    }
    if return_debug:
        result["debug"] = {"model": model_id, "prompt": prompt_str, "raw_output": raw, "sanity": sanity}
    return result
