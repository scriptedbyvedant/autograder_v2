# ilias_utils/zip_parser.py
import os
import re
import json
import zipfile
import mimetypes
from typing import Optional, Tuple, List, Iterable, Dict
from .models import StudentFile, StudentFolder, IngestResult

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MATRIC_RE = re.compile(r"^[A-Za-z0-9._\-\/]+$")  # relaxed for some IDs


def _guess_mime(filename: str) -> str:
    mt, _ = mimetypes.guess_type(filename)
    return mt or "application/octet-stream"


def parse_student_folder_name(folder_name: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Supports:
      - Space-delimited: 'Lastname Firstname(with spaces) email@domain 123456'
      - Underscore-delimited: 'LAST_FIRST_email@domain_123456'
    Returns (lastname, firstname, email, matric)
    """
    name = folder_name.strip("/")

    # underscore format
    if "_" in name and " " not in name:
        parts = name.split("_")
        if len(parts) >= 3:
            email_idx = next((i for i, t in enumerate(parts) if "@" in t), None)
            if email_idx is not None:
                email = parts[email_idx]
                matric = parts[-1] if MATRIC_RE.match(parts[-1]) else None
                lastname = parts[0] if parts else None
                firstname_tokens = parts[1:email_idx] if email_idx > 1 else []
                firstname = " ".join(firstname_tokens) if firstname_tokens else None
                if EMAIL_RE.match(email):
                    return lastname, firstname, email, matric

    # space-delimited
    tokens = name.split()
    if len(tokens) >= 3:
        email_idx = next((i for i, t in enumerate(tokens) if EMAIL_RE.match(t)), None)
        if email_idx is not None:
            email = tokens[email_idx]
            matric = tokens[-1] if MATRIC_RE.match(tokens[-1]) else None
            name_tokens = tokens[:email_idx]
            lastname = name_tokens[0] if name_tokens else None
            firstname = " ".join(name_tokens[1:]) if len(name_tokens) > 1 else None
            return lastname, firstname, email, matric

    return None, None, None, None


def _iter_zip(z: zipfile.ZipFile) -> Iterable[zipfile.ZipInfo]:
    for info in z.infolist():
        info.filename = info.filename.replace("\\", "/")
        yield info


def _find_single_root(arcs: List[str]) -> str:
    """Return 'root/' if there is exactly one top-level dir, else ''."""
    roots = [a for a in arcs if a.endswith("/") and a.count("/") == 1]
    return roots[0] if len(roots) == 1 else ""


def _find_case_insensitive_submissions_root(arcs: List[str], root: str) -> Optional[str]:
    """
    Find the actual '<root>/<Submissions|submissions>/' directory name as it appears in the zip.
    Returns the exact arc prefix with trailing '/' (e.g., 'assignment-1/Submissions/').
    """
    # First, try to find explicit dir entries one level below root
    # e.g., 'assignment-1/Submissions/'
    for a in arcs:
        if not a.startswith(root):
            continue
        rel = a[len(root):]
        if not rel:
            continue
        # we only care about the first-level folder
        if rel.endswith("/") and rel.count("/") == 1:
            first = rel[:-1]  # strip trailing '/'
            if first.lower() == "submissions":
                return root + first + "/"

    # If no explicit dir entry, infer from files/folders that start with root + something
    for a in arcs:
        if not a.startswith(root):
            continue
        rel = a[len(root):]
        if not rel:
            continue
        # get first segment
        seg = rel.split("/", 1)[0]
        if seg and seg.lower() == "submissions":
            return root + seg + "/"

    return None


def _ensure_student(student_map: Dict[str, StudentFolder], sdir: str) -> StudentFolder:
    if sdir in student_map:
        return student_map[sdir]
    ln, fn, em, ma = parse_student_folder_name(sdir)
    sf = StudentFolder(
        raw_folder=sdir,
        lastname=ln,
        firstname=fn,
        email=em,
        matric=ma,
        files=[],
    )
    student_map[sdir] = sf
    return sf


def parse_ilias_zip(zip_path: str) -> IngestResult:
    """
    Flexible parser that supports:
      1) submissions/<student>/...
      2) <root>/submissions/<student>/...
      3) <root>/<student>/...   (multi_feedback style)
      4) <student>/...          (zip root)
    Case-insensitive detection of 'submissions' under <root>/.
    Also detects a top-level Excel (either at root or directly under <root>/).
    """
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(zip_path)
    if not zip_path.lower().endswith(".zip"):
        raise ValueError("Expected a .zip ILIAS export")

    assignment_name = os.path.splitext(os.path.basename(zip_path))[0]
    excel_candidate: Optional[str] = None
    student_map: Dict[str, StudentFolder] = {}

    with zipfile.ZipFile(zip_path, "r") as z:
        arcs = [i.filename.replace("\\", "/") for i in z.infolist()]
        root = _find_single_root(arcs)

        # Excel at root or under <root>/
        top_excels = [a for a in arcs if ("/" not in a.rstrip("/")) and a.lower().endswith((".xlsx", ".xls"))]
        if top_excels:
            excel_candidate = top_excels[0]
        elif root:
            excels_under_root = [a for a in arcs if a.startswith(root) and a.count("/") == 1 and a.lower().endswith((".xlsx", ".xls"))]
            if excels_under_root:
                excel_candidate = excels_under_root[0]

        # Case 1 & 2: submissions path (case-insensitive under root)
        handled_any = False
        prefixes: List[str] = []
        if root:
            subdir = _find_case_insensitive_submissions_root(arcs, root)
            if subdir:
                prefixes.append(subdir)
        # also consider bare 'submissions/' at zip root (rare)
        prefixes.append("submissions/")

        for pref in prefixes:
            if any(a.startswith(pref) for a in arcs):
                handled_any = True
                for info in _iter_zip(z):
                    arc = info.filename
                    if not arc.startswith(pref):
                        continue
                    rel = arc[len(pref):]
                    if not rel:
                        continue
                    if rel.endswith("/"):  # student dir entry
                        sdir = rel.rstrip("/")
                        if sdir:
                            _ensure_student(student_map, sdir)
                        continue
                    # file under submissions/<student>/...
                    parts = rel.split("/", 1)
                    if len(parts) < 2:
                        continue
                    sdir, file_rel = parts
                    st = _ensure_student(student_map, sdir)
                    fname = os.path.basename(file_rel)
                    st.files.append(StudentFile(
                        arcname=arc,
                        filename=fname,
                        size=info.file_size,
                        content_type=_guess_mime(fname),
                    ))

        # Case 3 & 4: first-level student dirs (multi_feedback or zip root students)
        if not handled_any:
            base = root
            for info in _iter_zip(z):
                arc = info.filename
                if base and not arc.startswith(base):
                    continue
                rel = arc[len(base):] if base else arc
                if not rel:
                    continue

                if rel.endswith("/"):
                    parts = rel.split("/")
                    if len(parts) == 2 and parts[0]:
                        _ensure_student(student_map, parts[0])
                    continue

                parts = rel.split("/", 1)
                if len(parts) < 2:
                    continue
                sdir, file_rel = parts
                if not sdir or "/" in sdir:
                    continue
                st = _ensure_student(student_map, sdir)
                fname = os.path.basename(file_rel)
                st.files.append(StudentFile(
                    arcname=arc,
                    filename=fname,
                    size=info.file_size,
                    content_type=_guess_mime(fname),
                ))

    return IngestResult(
        assignment_name=assignment_name,
        excel_path=excel_candidate,
        student_folders=list(student_map.values()),
    )


# --- STRICT parser for the canonical assignment layout (<root>/submissions/<student>/...) ---

def parse_ilias_assignment_zip_strict(zip_path: str) -> IngestResult:
    """
    Strictly parse zips like:
      <zip>.zip
      └── <root>/
          ├── <root>.xlsx (optional)
          └── submissions/ (any case)
              ├── <student_folder>/
              │   └── file.ext
              └── ...
    Accepts that the ZIP may not contain an explicit '<root>/submissions/' dir entry,
    as long as there are files under that prefix. Case-insensitive for 'submissions'.
    Raises ValueError on violations.
    """
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(zip_path)
    if not zip_path.lower().endswith(".zip"):
        raise ValueError("Expected a .zip ILIAS export")

    with zipfile.ZipFile(zip_path, "r") as z:
        arcs = [i.filename.replace("\\", "/") for i in z.infolist()]

        # 1) exactly one top-level root
        roots = [a for a in arcs if a.endswith("/") and a.count("/") == 1]
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one root folder, found: {roots}")
        root = roots[0]  # e.g., "assignment-1/"

        # 2) submissions dir under root (case-insensitive)
        subdir = _find_case_insensitive_submissions_root(arcs, root)
        if not subdir:
            raise ValueError(f"No 'submissions' folder found under root (case-insensitive): '{root}<Submissions>/'")

        # ensure there are files under subdir
        if not any(a.startswith(subdir) and not a.endswith("/") for a in arcs):
            raise ValueError(f"No files found under expected submissions dir: '{subdir}'")

        # 3) optional excel directly under root
        excels_under_root = [
            a for a in arcs
            if a.startswith(root) and a.count("/") == 1 and a.lower().endswith((".xlsx", ".xls"))
        ]
        excel_candidate = excels_under_root[0] if excels_under_root else None

        # 4) collect student folders (immediate children of <root>/<Submissions>/)
        student_map: Dict[str, StudentFolder] = {}

        def ensure_student(sdir: str) -> StudentFolder:
            if sdir in student_map:
                return student_map[sdir]
            ln, fn, em, ma = parse_student_folder_name(sdir)
            student_map[sdir] = StudentFolder(
                raw_folder=sdir, lastname=ln, firstname=fn, email=em, matric=ma, files=[]
            )
            return student_map[sdir]

        # Prefer explicit dir entries; fall back to inferring from files
        student_dirs = set()
        for a in arcs:
            if a.startswith(subdir) and a.endswith("/"):
                rel = a[len(subdir):].rstrip("/")
                if rel and "/" not in rel:
                    student_dirs.add(rel)

        if not student_dirs:
            for a in arcs:
                if a.startswith(subdir) and not a.endswith("/"):
                    rel = a[len(subdir):]
                    if "/" in rel:
                        student_dirs.add(rel.split("/", 1)[0])

        if not student_dirs:
            raise ValueError(f"No student folders found under '{subdir}'")

        # Attach files to each student
        for info in z.infolist():
            arc = info.filename.replace("\\", "/")
            if not arc.startswith(subdir) or arc.endswith("/"):
                continue
            rel = arc[len(subdir):]              # "<student_dir>/file..."
            if "/" not in rel:
                continue
            sdir, tail = rel.split("/", 1)
            if sdir not in student_dirs:
                continue
            st = ensure_student(sdir)
            fname = os.path.basename(tail)
            st.files.append(
                StudentFile(
                    arcname=arc,
                    filename=fname,
                    size=info.file_size,
                    content_type=_guess_mime(fname),
                )
            )

        assignment_name = os.path.splitext(os.path.basename(zip_path))[0]
        return IngestResult(
            assignment_name=assignment_name,
            excel_path=excel_candidate,
            student_folders=list(student_map.values()),
        )


# --- helpers to save/load and selective extraction ---

def save_manifest(result: IngestResult, out_json_path: str) -> None:
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)


def load_manifest(json_path: str) -> IngestResult:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return IngestResult.from_dict(data)


def extract_student_files(zip_path: str, dest_dir: str, only_students: Optional[List[str]] = None) -> int:
    """
    Extract 'submissions/<student>/**' OR '<root>/(Submissions|submissions)/<student>/**'
    OR '<root>/<student>/**' OR '<student>/**' to dest_dir.
    Case-insensitive for 'submissions' under <root>/.
    """
    os.makedirs(dest_dir, exist_ok=True)
    selected = set(only_students or [])
    count = 0
    with zipfile.ZipFile(zip_path, "r") as z:
        arcs = [i.filename.replace("\\", "/") for i in z.infolist()]
        root = _find_single_root(arcs)
        prefixes = []

        if root:
            subdir = _find_case_insensitive_submissions_root(arcs, root)
            if subdir:
                prefixes.append(subdir)
            prefixes.append(root)  # first-level under root (multi_feedback style)
        prefixes.append("submissions/")  # bare submissions at zip root (rare)

        for info in _iter_zip(z):
            arc = info.filename
            match_pref = next((p for p in prefixes if arc.startswith(p)), None)
            if not match_pref:
                continue
            rel = arc[len(match_pref):]
            if not rel:
                continue

            parts = rel.split("/", 1)
            if len(parts) < 2:
                continue
            sdir, tail = parts
            if selected and sdir not in selected:
                continue

            target = os.path.join(dest_dir, sdir, tail)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if not info.is_dir():
                with z.open(info, "r") as src, open(target, "wb") as dst:
                    dst.write(src.read())
                    count += 1
    return count
