"""File parser — extracts text from various file formats."""

import json
import logging
import os
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FileParser:
    """Parse diverse file formats into plain text with metadata."""

    SUPPORTED = {".txt", ".json", ".md", ".log", ".csv", ".yaml", ".yml", ".xml"}

    @classmethod
    def parse(cls, file_path: str | Path) -> dict[str, Any]:
        """Parse a file and return {content, metadata, doc_type}.

        Raises ValueError for unsupported formats.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        doc_type = cls._infer_doc_type(path)

        if suffix in (".txt", ".md", ".log"):
            return cls._parse_text(path, doc_type)
        elif suffix == ".json":
            return cls._parse_json(path, doc_type)
        elif suffix == ".csv":
            return cls._parse_csv(path, doc_type)
        elif suffix in (".yaml", ".yml"):
            return cls._parse_yaml(path, doc_type)
        elif suffix == ".xml":
            return cls._parse_text(path, doc_type)  # treat as text for now
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: {cls.SUPPORTED}")

    @classmethod
    def _infer_doc_type(cls, path: Path) -> str:
        name = path.name.lower()
        if "incident" in name or "pagerduty" in name or "alert" in name:
            return "incident"
        elif "runbook" in name or "sop" in name or "procedure" in name:
            return "runbook"
        elif "log" in name or "syslog" in name or "error" in name:
            return "log"
        elif "config" in name or "configuration" in name:
            return "config"
        elif "note" in name or "meeting" in name or "doc" in name:
            return "note"
        elif "report" in name or "postmortem" in name:
            return "report"
        elif "service" in name or "catalog" in name:
            return "service_catalog"
        return "document"

    @classmethod
    def _refine_type_from_content(cls, content: str) -> str | None:
        """Check content headers for doc type hints when filename is ambiguous."""
        first_lines = content.strip().split("\n")[:5]
        header = " ".join(first_lines).lower()
        if any(kw in header for kw in ("runbook", "sop", "playbook", "run book", "standard operating procedure")):
            return "runbook"
        return None

    @classmethod
    def _parse_text(cls, path: Path, doc_type: str) -> dict[str, Any]:
        content = path.read_text(encoding="utf-8", errors="replace")
        # Content-based type refinement if filename gave no hint
        if doc_type == "document":
            content_type = cls._refine_type_from_content(content)
            if content_type:
                doc_type = content_type
        lines = content.split("\n")
        metadata = {
            "filename": path.name,
            "doc_type": doc_type,
            "line_count": len(lines),
            "char_count": len(content),
            "file_size": path.stat().st_size,
        }
        # Try to extract title from first heading
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# "):
                metadata["title"] = stripped.lstrip("# ")
                break
        return {"content": content, "metadata": metadata, "doc_type": doc_type}

    @classmethod
    def _parse_json(cls, path: Path, doc_type: str) -> dict[str, Any]:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        content = json.dumps(raw, indent=2)
        metadata = {
            "filename": path.name,
            "doc_type": doc_type,
            "top_level_keys": list(raw.keys()) if isinstance(raw, dict) else f"array[{len(raw)}]",
            "file_size": path.stat().st_size,
        }
        if isinstance(raw, list):
            metadata["item_count"] = len(raw)
        return {"content": content, "metadata": metadata, "doc_type": doc_type}

    @classmethod
    def _parse_csv(cls, path: Path, doc_type: str) -> dict[str, Any]:
        import csv
        import io
        content = path.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        metadata = {
            "filename": path.name,
            "doc_type": doc_type,
            "row_count": len(rows),
            "columns": reader.fieldnames,
            "file_size": path.stat().st_size,
        }
        return {"content": content, "metadata": metadata, "doc_type": doc_type}

    @classmethod
    def _parse_yaml(cls, path: Path, doc_type: str) -> dict[str, Any]:
        try:
            import yaml
            raw = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
            content = yaml.dump(raw, default_flow_style=False) if raw else ""
        except ImportError:
            content = path.read_text(encoding="utf-8", errors="replace")
        metadata = {
            "filename": path.name,
            "doc_type": doc_type,
            "file_size": path.stat().st_size,
        }
        return {"content": content, "metadata": metadata, "doc_type": doc_type}
