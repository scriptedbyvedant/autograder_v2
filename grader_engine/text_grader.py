# File: grader_engine/text_grader.py

import os
import json
import re
import difflib
from typing import List, Dict, Any, Optional

import torch
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.chat_models import ChatOllama
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

load_dotenv()

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

# --- Base Model Configuration ---
# This is the model that will be used if no fine-tuned model is found.
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# --- Fine-Tuned Model Configuration ---
# The base model used for fine-tuning. This MUST match the model from fine_tune.py
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
# The path to the trained LoRA adapters from the fine-tuning process.
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), '..', 'training', 'results', 'final_model')


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
# This template is used for BOTH fine-tuned and Ollama models.
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


def _get_raw_prediction_finetuned(prompt: str) -> (str, str):
    """Generates a prediction using the fine-tuned PEFT model.
       Requires a GPU with sufficient VRAM.
    """
    model_id = f"{BASE_MODEL_NAME} (PEFT Adapters)"
    print(f"Loading fine-tuned model from {ADAPTER_PATH}...")

    # Configure quantization to load the model in 4-bit, saving memory.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # Load the base model with quantization.
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    base_model.config.use_cache = False

    # Load the PEFT model by applying the adapters to the base model.
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Generate the prediction.
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_sequences = model.generate(input_ids=inputs["input_ids"], max_new_tokens=1024)
    
    # Decode the output and strip the prompt from the beginning.
    raw_output = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    # The raw output includes the prompt, so we remove it.
    clean_output = raw_output[len(prompt):].strip()
    
    return clean_output, model_id

def _get_raw_prediction_ollama(prompt: str) -> (str, str):
    """Generates a prediction using the fallback Ollama model.
    """
    model_id = OLLAMA_FALLBACK_MODEL
    print(f"Using fallback Ollama model: {model_id}")
    llm = ChatOllama(model=model_id)
    resp = llm.invoke(prompt)
    return resp.content, model_id


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------
def grade_answer(
    question: str,
    ideal_answer: str,
    rubric: List[Dict[str, Any]],
    student_answer: str,
    language: str = "English",
    model_name: Optional[str] = None, # model_name is now ignored, but kept for API compatibility
    rag_context: Optional[Dict[str, Any]] = None,
    return_debug: bool = False,
    include_header_in_feedback: bool = True
) -> Dict[str, Any]:
    """
    Grade a single answer. It will prioritize using a fine-tuned model if available,
    otherwise it will fall back to using an Ollama model.
    """
    use_finetuned = os.path.isdir(ADAPTER_PATH)
    
    # Normalize rubric
    rubric = json.loads(rubric) if isinstance(rubric, str) else (rubric or [])
    rubric_json = json.dumps(rubric, ensure_ascii=False)

    # Build prompt
    prompt_str = _make_prompt(
        language=language,
        question=question, ideal_answer=ideal_answer, rubric_json=rubric_json,
        student_answer=student_answer, rag_context=rag_context
    )

    # Get raw prediction from either the fine-tuned model or Ollama
    raw, model_id = "", ""
    try:
        if use_finetuned:
            raw, model_id = _get_raw_prediction_finetuned(prompt_str)
        else:
            raw, model_id = _get_raw_prediction_ollama(prompt_str)
    except Exception as e:
        # Handle potential errors during model inference (e.g., CUDA out of memory)
        out = {
            "total_score":   0,
            "rubric_scores": [{"criteria": r.get("criteria",""), "score": 0} for r in rubric],
            "feedback":      f"Grading failed: {e}"
        }
        if return_debug:
            out["debug"] = {"model": "N/A", "prompt": prompt_str, "raw_output": str(e), "sanity": {"error": "invoke_failed"}}
        return out

    # --- Post-processing (same for both models) ---

    # Parse JSON from the model's raw output
    parsed = {}
    try: parsed = output_parser.parse(raw)
    except Exception: 
        try: parsed = json.loads(_extract_json(raw))
        except Exception: parsed = {}

    # Extract fields
    model_total = _as_int(parsed.get("total_score", 0), 0)
    model_breakdown = parsed.get("rubric_scores", []) or []
    model_feedback = parsed.get("feedback", "")
    if not isinstance(model_feedback, str): model_feedback = str(model_feedback)

    # Align scores to the rubric and clamp values
    aligned, sanity = _align_and_clamp(rubric, model_breakdown)

    # Recompute total score from the aligned rubric (the source of truth)
    total_awarded = int(sum(int(it["score"]) for it in aligned))
    if model_total != total_awarded:
        sanity["model_total"] = model_total
        sanity["recomputed_total"] = total_awarded

    # Build the final feedback string
    body = model_feedback.strip()
    if include_header_in_feedback:
        header = _feedback_header(rubric, aligned, total_awarded)
        full_feedback = header + ("\n\n" + body if body else "")
    else:
        full_feedback = body

    # Final result object
    result = {
        "total_score":   total_awarded,
        "rubric_scores": aligned,
        "feedback":      full_feedback
    }
    if return_debug:
        result["debug"] = {"model": model_id, "prompt": prompt_str, "raw_output": raw, "sanity": sanity}
    return result
