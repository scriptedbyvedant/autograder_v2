# ilias_utils/models.py
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class StudentFile:
    arcname: str                 # path inside zip: submissions/<folder>/<file>
    filename: str                # basename.ext
    size: int                    # bytes
    content_type: Optional[str] = None  # guessed content type


@dataclass
class StudentFolder:
    raw_folder: str              # "Lastname Firstname email@domain 123456" (or underscore variant)
    lastname: Optional[str]
    firstname: Optional[str]
    email: Optional[str]
    matric: Optional[str]
    files: List[StudentFile] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StudentFolder":
        files = [StudentFile(**f) for f in d.get("files", [])]
        return StudentFolder(
            raw_folder=d.get("raw_folder"),
            lastname=d.get("lastname"),
            firstname=d.get("firstname"),
            email=d.get("email"),
            matric=d.get("matric"),
            files=files,
        )


@dataclass
class IngestResult:
    assignment_name: str                 # "assignment-1"
    excel_path: Optional[str]            # top-level xlsx/xls under root (if any)
    student_folders: List[StudentFolder]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignment_name": self.assignment_name,
            "excel_path": self.excel_path,
            "student_folders": [sf.to_dict() for sf in self.student_folders],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "IngestResult":
        sfs = [StudentFolder.from_dict(x) for x in d.get("student_folders", [])]
        return IngestResult(
            assignment_name=d.get("assignment_name"),
            excel_path=d.get("excel_path"),
            student_folders=sfs,
        )
