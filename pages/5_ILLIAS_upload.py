# pages/5_ILIAS_upload.py
# ILIAS upload + preview: parse ZIP -> show student names, filenames, and PDF text content

import io
import os
import tempfile
from typing import Optional, Tuple, List
import zipfile
import streamlit as st

# Use our shared parser from ilias_utils (case-insensitive Submissions)
from ilias_utils.zip_parser import parse_ilias_assignment_zip_strict, IngestResult

st.set_page_config(page_title="ILIAS Upload & Preview", page_icon="🧩", layout="wide")
st.title("🧩 ILIAS — Upload & Preview (with PDF text)")

st.caption(
    "Upload the ILIAS assignment ZIP (root folder + **Submissions/**). "
    "We’ll parse all students and list their files. For **PDFs**, we’ll also show the extracted text."
)

uploaded = st.file_uploader("Upload ILIAS assignment ZIP", type=["zip"])
if not uploaded:
    st.info("Select an ILIAS **.zip** exported from the Exercise tool.")
    st.stop()

# ---- Optional PDF extractors ----
_HAS_PDF_UTILS = False
extract_text_from_pdf_pdfutils = None
try:
    # Adjust this import if your pdf_utils API differs
    from pdf_utils.pdf_parser import extract_text_from_pdf as extract_text_from_pdf_pdfutils
    _HAS_PDF_UTILS = True
except Exception:
    pass

_HAS_PYMUPDF = False
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    pass

def extract_pdf_text(data: bytes, max_chars: Optional[int] = None) -> str:
    """
    Extract text from PDF bytes using (in order):
    1) your pdf_utils.pdf_parser.extract_text_from_pdf (if available)
    2) PyMuPDF (fitz)
    Returns a unicode string (may be empty if nothing extracted).
    """
    # Try your own utils first
    if _HAS_PDF_UTILS and extract_text_from_pdf_pdfutils:
        try:
            txt = extract_text_from_pdf_pdfutils(data)
            if isinstance(txt, bytes):
                txt = txt.decode("utf-8", errors="ignore")
            if max_chars:
                return (txt or "")[:max_chars]
            return txt or ""
        except Exception as e:
            # fall through to PyMuPDF
            pass

    # Fallback: PyMuPDF
    if _HAS_PYMUPDF:
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            chunks: List[str] = []
            for page in doc:
                chunks.append(page.get_text("text"))
            doc.close()
            text = "\n".join(chunks)
            if max_chars:
                return text[:max_chars]
            return text
        except Exception:
            return ""
    # No extractor available
    return ""

# ---- Persist ZIP to a temp file for the shared parser
with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tf:
    tf.write(uploaded.read())
    zip_path = tf.name

# ---- Parse ZIP via strict, case-insensitive parser
try:
    ingest: IngestResult = parse_ilias_assignment_zip_strict(zip_path)
except Exception as e:
    st.error(f"❌ ZIP structure not as expected:\n\n{e}")
    st.stop()
finally:
    # Keep temp zip for this run (helps debugging)
    pass

# Cache manifest in session state for other pages (grading/export)
st.session_state["ilias_ingest"] = ingest.to_dict()

# ---- Summary
num_students = len(ingest.student_folders)
num_files = sum(len(sf.files) for sf in ingest.student_folders)
st.success(f"✅ Parsed **{num_students}** students, **{num_files}** files.")
st.write("**Detected Excel (optional):**", ingest.excel_path or "— none —")

# ---- Load the ZIP once more to read file bytes for PDFs
zip_bytes = open(zip_path, "rb").read()
zip_mem = io.BytesIO(zip_bytes)

# Helper to build a nice display name
def _display_name(sf):
    fn = (sf.firstname or "").strip() if sf.firstname else ""
    ln = (sf.lastname or "").strip() if sf.lastname else ""
    name = (fn + " " + ln).strip()
    return name if name else sf.raw_folder

st.markdown("---")
st.subheader("Preview")

if not ingest.student_folders:
    st.info("No student folders detected under Submissions.")
else:
    sfs = sorted(ingest.student_folders, key=lambda s: (_display_name(s).lower(), s.raw_folder.lower()))
    with st.spinner("Extracting PDF text (if any)…"):
        with st.expander("Extraction backends", expanded=False):
            st.write("pdf_utils:", "✅ found" if _HAS_PDF_UTILS else "⛔ not found")
            st.write("PyMuPDF (fitz):", "✅ found" if _HAS_PYMUPDF else "⛔ not found")

        with st.spinner("Rendering students…"):
            for sf in sfs:
                st.markdown(f"### 👤 {_display_name(sf)}")
                meta_bits = []
                if sf.email:  meta_bits.append(f"**Email:** {sf.email}")
                if sf.matric: meta_bits.append(f"**Matric:** {sf.matric}")
                if meta_bits:
                    st.markdown(" • ".join(meta_bits))

                if not sf.files:
                    st.markdown("_file not present_")
                    continue

                # unique filenames, sorted; and map arcname for extraction
                files = {}
                for f in sf.files:
                    files.setdefault(f.filename, f.arcname)  # prefer first occurrence
                for fname in sorted(files.keys(), key=lambda x: x.lower()):
                    arc = files[fname]
                    st.markdown(f"- **`{fname}`**")
                    # Show PDF content if it's a PDF; otherwise just list the name
                    if fname.lower().endswith(".pdf"):
                        try:
                            with zipfile.ZipFile(zip_mem, "r") as zf:
                                with zf.open(arc, "r") as fh:
                                    pdf_bytes = fh.read()
                        except Exception as ex:
                            st.warning(f"Could not open '{fname}' from ZIP: {ex}")
                            continue

                        # Extract text (preview + full)
                        preview = extract_pdf_text(pdf_bytes, max_chars=1500)  # short inline preview
                        full = extract_pdf_text(pdf_bytes, max_chars=None)

                        if not preview and not full:
                            st.write("_(no text extracted from PDF)_")
                        else:
                            st.code(preview or full, language="markdown")
                            if full and len(full) > len(preview):
                                with st.expander(f"Show full text of {fname}"):
                                    st.text(full)
                    else:
                        # For non-PDFs we just show the filename (extend here if you want text previews)
                        pass

st.caption("Read-only preview. No grading performed here. PDFs are extracted from the ZIP in-memory.")
