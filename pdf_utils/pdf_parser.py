import fitz  # PyMuPDF
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional


# --- Data model for downstream pipeline --------------------------------------
@dataclass
class Block:
    q_id: str
    block_type: str           # 'question' | 'ideal' | 'rubric' | 'student'
    modality: List[str]       # ['text','math','code','table','image']
    text: str = ""
    latex: List[str] = None
    code: Optional[Dict[str, str]] = None     # {"lang": "...", "content": "..."}
    tables: Optional[List[Dict[str, str]]] = None
    images: Optional[List[str]] = None
    bbox: Optional[List[float]] = None
    page: int = 0


MATH_PATTERN = re.compile(r"\$[^$]+\$|\\\([^\)]+\\\)|\\\[[^\]]+\\\]", re.MULTILINE)
CODE_FENCE   = re.compile(r"```(\w+)?\n([\s\S]*?)```", re.MULTILINE)


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract raw text from uploaded PDF (file-like object).
    """
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join(page.get_text("text") for page in doc)


def extract_blocks_from_pdf(pdf_file) -> List[Block]:
    """
    Robust(ish) layout-aware extraction that keeps page segmentation,
    detects inline/display LaTeX, and fenced code blocks.
    """
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    blocks: List[Block] = []
    qn = 0

    for pno, page in enumerate(doc):
        text = page.get_text("text")
        if not text.strip():
            continue

        # Segment by common headers: Q1:, Question 1, Aufgabe 1, etc.
        chunks = re.split(
            r"(?mi)^(?:Q(?:uestion)?\s*\d+[:.)]|Aufgabe\s*\d+[:.)]|Frage\s*\d+[:.)])",
            text
        )

        # If no obvious markers, treat page as one question chunk
        if len(chunks) <= 1:
            chunks = [text]

        for chunk in chunks:
            c = chunk.strip()
            if not c:
                continue

            qn += 1
            qid = f"Q{qn}"

            latex = [m.group(0) for m in MATH_PATTERN.finditer(c)]
            code_m = list(CODE_FENCE.finditer(c))
            code = None
            modality = ["text"]
            if latex:
                modality.append("math")
            if code_m:
                lang = (code_m[0].group(1) or "text").lower()
                code = {"lang": lang, "content": code_m[0].group(2)}
                modality.append("code")

            blocks.append(
                Block(
                    q_id=qid,
                    block_type="question",
                    modality=list(dict.fromkeys(modality)),
                    text=c,
                    latex=latex or [],
                    code=code,
                    page=pno
                )
            )
    return blocks


def parse_professor_pdf(text: str) -> Dict[str, Any]:
    """
    Parse an instructor PDF (already text) into sections for questions/ideals/rubrics.
    Flexible: supports English/German keywords and numbered sections.

    Returns: {
      "questions": { "Q1": "...", ... },
      "ideals":    { "Q1": "...", ... },
      "rubrics":   { "Q1": { ... }, ... }
    }
    """
    # Extract "sequence" sections if present
    seq_sections = re.split(r"(?mi)^(?:Questions?|Aufgaben)\s*:\s*$", text)
    questions: Dict[str, str] = {}
    ideals: Dict[str, str] = {}
    rubrics: Dict[str, Any] = {}

    # Questions
    q_matches = re.finditer(
        r"(?mi)^(?:Q(?:uestion)?|Aufgabe)\s*(\d+)\s*[:.)]\s*(.*?)(?=^(?:Q(?:uestion)?|Aufgabe)\s*\d+|^Ideal\s*Answer|^Rubric|$)",
        text, re.S | re.M
    )
    for m in q_matches:
        questions[f"Q{m.group(1)}"] = m.group(2).strip()

    # Ideals
    i_matches = re.finditer(
        r"(?mi)^Ideal\s*Answer\s*(\d+)\s*[:.)]\s*(.*?)(?=^Ideal\s*Answer|^Rubric|$)",
        text, re.S | re.M
    )
    for m in i_matches:
        ideals[f"Q{m.group(1)}"] = m.group(2).strip()

    # Rubrics: allow inline JSON or bullet lists; keep raw text if not JSON
    r_matches = re.finditer(
        r"(?mi)^Rubric\s*(\d+)\s*[:.)]\s*(.*?)(?=^Rubric|$)",
        text, re.S | re.M
    )
    for m in r_matches:
        raw = m.group(2).strip()
        cleaned = raw
        try:
            import json
            rubrics[f"Q{m.group(1)}"] = json.loads(raw)
        except Exception:
            # Simple bullet -> JSON conversion (• item (n points))
            lines = [ln.strip("-• ").strip() for ln in cleaned.splitlines() if ln.strip()]
            criteria = []
            for ln in lines:
                pts = re.search(r"(\d+(?:\.\d+)?)\s*(?:pts?|points?)", ln, re.I)
                criteria.append({"id": ln, "points": float(pts.group(1)) if pts else 1.0})
            rubrics[f"Q{m.group(1)}"] = {"criteria": criteria}

    return {"questions": questions, "ideals": ideals, "rubrics": rubrics}


def parse_student_pdf(text: str) -> Dict[str, Dict[str, str]]:
    """
    Parse a student submissions PDF. Accepts blocks like:

      Student 1:
      A1: ...
      A2: ...

      Student 2:
      A1: ...
      ...

    Returns: { "Student 1": { "A1": "...", "A2": "..." }, ... }
    """
    students: Dict[str, Dict[str, str]] = {}
    parts = re.split(r'(?m)^(Student\s*\d+:|Studierende[rn]?\s*\d+:)', text)
    for i in range(1, len(parts), 2):
        label = parts[i].rstrip(':').strip()
        block = parts[i + 1]
        answers: Dict[str, str] = {}
        for m in re.finditer(r'(A\d+|F\d+):\s*(.*?)(?=\nA\d+:|\nF\d+:|\nStudent\s*\d+:|\nStudierende[rn]?\s*\d+:|$)', block, re.S):
            answers[m.group(1)] = m.group(2).strip()
        students[label] = answers
    return students


def blocks_to_json(blocks: List[Block]) -> List[Dict[str, Any]]:
    return [asdict(b) for b in blocks]
