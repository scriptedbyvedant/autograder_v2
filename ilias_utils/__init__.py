# ilias_utils/__init__.py
from .models import StudentFile, StudentFolder, IngestResult
from .zip_parser import (
    parse_ilias_zip,
    parse_ilias_assignment_zip_strict,
    save_manifest,
    load_manifest,
    extract_student_files,
)
from .manifest_adapter import build_items_from_ingest
from .backend_bridge import (
    build_items,
    grade_items,
    group_results_by_student,
    persist_results_to_db,
)
from .feedback_zip import build_feedback_zip

__all__ = [
    "StudentFile",
    "StudentFolder",
    "IngestResult",
    "parse_ilias_zip",
    "parse_ilias_assignment_zip_strict",
    "save_manifest",
    "load_manifest",
    "extract_student_files",
    "build_items_from_ingest",
    "build_items",
    "grade_items",
    "group_results_by_student",
    "persist_results_to_db",
    "build_feedback_zip",
]
