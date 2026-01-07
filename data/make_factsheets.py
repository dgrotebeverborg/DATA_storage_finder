#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------- Config ----------
INPUT_JSON = "storage_data_2.json"
OUT_DIR = Path("factsheets")

# Same structure for every factsheet (always these sections, same order)
SECTIONS = [
    "What it is",
    "Main purpose",
    "Typical data types",
    "Position in the research data lifecycle",
    "Collaboration & access",
    "Storage capacity & file size",
    "Security & privacy",
    "Backup & versioning",
    "Costs & funding model",
    "Hosting & data location",
    "When to use",
    "When NOT to use",
    "Support & governance",
]

# Map JSON fields -> factsheet content slots
FIELD_MAP = {
    "What it is": ["storage_details"],
    "Main purpose": ["storage_details", "positioning_datalyfecycle", "file_type"],
    "Typical data types": ["file_type", "file_formats"],
    "Position in the research data lifecycle": ["positioning_datalyfecycle"],
    "Collaboration & access": [
        "collaboration",
        "collaboration_notes",
        "bound-to- personal-account",
        "ownershiptransfer",
        "data_accesscontrolfeatures",
        "Data transfer protocol",
        "Data transfer protocol notes",
        "sync_equipment",
        "sync_HPC",
        "sync_R/JupyterNsync",
        "connectivity,availability",
    ],
    "Storage capacity & file size": ["capacity", "max_file_size"],
    "Security & privacy": [
        "GDPR_compliant",
        "GDPR_compliant_note",
        "GDPR_Location_datasupplier",
        "GDPR_data-removal-period",
        "availability_classification",
        "sensitivity_classification",
        "Data CIA",
        "file_encryption (by default",
        "data_location",
    ],
    "Backup & versioning": ["back-ups (by default)", "data_versioning (by default)", "complete_data_deletion2"],
    "Costs & funding model": ["price", "Price per TB / month", "Price up front", "costmodel"],
    "Hosting & data location": ["Host", "data_location", "GDPR_Location_datasupplier"],
    "When to use": ["positioning_datalyfecycle", "collaboration", "availability_classification", "sensitivity_classification"],
    "When NOT to use": ["Recommendation notes", "positioning_datalyfecycle", "bound-to- personal-account"],
    "Support & governance": [
        "Supported by UU?",
        "Where to request",
        "Where to request notes",
        "Internal_contact_point_RDM",
        "Link_ITManuals",
        "Link_SLA",
        "End-of-Life",
        "sustainability",
        "Recommendation notes",
    ],
}


# ---------- Helpers ----------
def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s)
    return s.strip("-") or "storage-solution"


def is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        vv = v.strip().lower()
        return vv in {"", "nan", "none", "null", "?", "no"}  # treat these as missing/unknown in narrative
    if isinstance(v, list) and len(v) == 0:
        return True
    return False


def normalize_value(v: Any) -> Optional[str]:
    """Turn JSON values into readable text, or None if missing/unknown."""
    if is_missing(v):
        return None
    if isinstance(v, list):
        # list of lifecycle phases etc.
        items = [str(x).strip() for x in v if not is_missing(x)]
        if not items:
            return None
        return ", ".join(items)
    return str(v).strip()


def bullets_from_fields(item: Dict[str, Any], fields: List[str]) -> List[str]:
    """Create concise bullet lines from a list of fields in a stable order."""
    out: List[str] = []
    for f in fields:
        if f not in item:
            continue
        val = normalize_value(item.get(f))
        if not val:
            continue
        # make it human-readable, but grounded in the source
        label = f
        # small label cleanups
        label = label.replace(" (by default)", "").replace("connectivity,availability", "connectivity/availability")
        label = label.replace("bound-to- personal-account", "bound to personal account")
        out.append(f"- **{label}:** {val}")
    return out


def narrative_intro(name: str, item: Dict[str, Any]) -> str:
    """One short paragraph, using storage_details if available."""
    details = normalize_value(item.get("storage_details"))
    if details:
        return f"{name} — {details}"
    return f"{name} — storage solution described in the source data."


def build_section(name: str, item: Dict[str, Any], section: str) -> str:
    fields = FIELD_MAP.get(section, [])
    bullets = bullets_from_fields(item, fields)

    # Add small section-specific enrichment *without adding new facts*
    if section == "Position in the research data lifecycle":
        lifecycle = item.get("positioning_datalyfecycle")
        lifecycle_txt = normalize_value(lifecycle)
        if lifecycle_txt:
            bullets = [f"- **Supported phases:** {lifecycle_txt}"]
        else:
            bullets = ["- Not specified in the source data."]

    if section == "What it is":
        # Prefer a clean descriptive paragraph over raw bullet dump
        details = normalize_value(item.get("storage_details"))
        if details:
            return f"### {section}\n{details}\n"
        return f"### {section}\nNot specified in the source data.\n"

    if not bullets:
        bullets = ["- Not specified in the source data."]

    return f"### {section}\n" + "\n".join(bullets) + "\n"


def build_factsheet(name: str, item: Dict[str, Any]) -> str:
    header = f"# {name}\n\n"
    intro = narrative_intro(name, item) + "\n\n"

    body_parts = []
    for sec in SECTIONS:
        body_parts.append(build_section(name, item, sec))

    # Optional: add a compact “Raw links” list for usability
    links = []
    for key in ("Link_ITManuals", "Link_SLA"):
        v = normalize_value(item.get(key))
        if v:
            links.append(f"- **{key}:** {v}")
    if links:
        body_parts.append("### References (from source data)\n" + "\n".join(links) + "\n")

    return header + intro + "\n".join(body_parts)


def load_storage_items(path: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("storage_data_2", data)
    if not isinstance(items, list):
        raise ValueError("Expected 'storage_data_2' to be a list.")
    return items


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    items = load_storage_items(INPUT_JSON)

    all_md_parts: List[str] = []
    jsonl_lines: List[str] = []

    for item in items:
        name = item.get("name") or item.get("storage_name") or "Unnamed solution"
        name = str(name).strip()

        md = build_factsheet(name, item)

        slug = slugify(name)
        md_path = OUT_DIR / f"{slug}.md"
        md_path.write_text(md, encoding="utf-8")

        all_md_parts.append(md)

        # JSONL record: nice for embedding pipelines
        jsonl_obj = {
            "id": slug,
            "name": name,
            "text": md,
            "source": "storage_data_2.json",
        }
        jsonl_lines.append(json.dumps(jsonl_obj, ensure_ascii=False))

    (OUT_DIR / "all_factsheets.md").write_text("\n\n---\n\n".join(all_md_parts), encoding="utf-8")
    (OUT_DIR / "factsheets.jsonl").write_text("\n".join(jsonl_lines), encoding="utf-8")

    print(f"✅ Generated {len(items)} factsheets in: {OUT_DIR.resolve()}")
    print(f"   - {OUT_DIR / 'all_factsheets.md'}")
    print(f"   - {OUT_DIR / 'factsheets.jsonl'}")


if __name__ == "__main__":
    main()
