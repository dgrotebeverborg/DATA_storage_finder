#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple


BASE_DIR = Path(__file__).resolve().parent
FACTSHEETS_DIR = BASE_DIR / "factsheets"
DATA_2 = BASE_DIR / "storage_data_2.json"

URL_RE = re.compile(r"https?://[^\s\]\[\)\(\"'<>]+", re.I)


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s)
    return s.strip("-") or "storage-solution"


def is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        t = v.strip().lower()
        return t in {"", "nan", "none", "null", "n/a", "-", ".", "?"}
    if isinstance(v, list):
        return len(v) == 0
    return False


def as_text(v: Any) -> str:
    if is_missing(v):
        return ""
    if isinstance(v, list):
        return ", ".join(str(x).strip() for x in v if not is_missing(x))
    return str(v).strip()


def extract_urls(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [u.rstrip(".,;:") for u in URL_RE.findall(text)]


def collect_rows_storage_data_2() -> List[Dict[str, Any]]:
    data = json.loads(DATA_2.read_text(encoding="utf-8"))
    rows = []
    for row in data.get("storage_data_2", []):
        r = dict(row)
        r["_origin"] = "storage_data_2.json:storage_data_2"
        rows.append(r)
    return rows


def canonical_name(row: Dict[str, Any]) -> str:
    for key in ("name", "storage_name"):
        t = as_text(row.get(key))
        if t:
            return t
    return "Unnamed solution"


def merge_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    origins: Dict[str, List[str]] = defaultdict(list)
    urls_by_solution: Dict[str, Dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    # Prefer storage_data_2 values where both exist.
    rows_sorted = sorted(rows, key=lambda r: 0 if str(r.get("_origin", "")).startswith("storage_data_2") else 1)

    for row in rows_sorted:
        name = canonical_name(row)
        if name not in merged:
            merged[name] = {"name": name}
        target = merged[name]
        origins[name].append(str(row.get("_origin", "unknown")))

        for k, v in row.items():
            if k.startswith("_"):
                continue
            txt = as_text(v)
            if txt:
                target[k] = txt
                for url in extract_urls(txt):
                    urls_by_solution[name][url].add(k)

    for name, record in merged.items():
        record["_sources"] = sorted(set(origins[name]))
        record["_links"] = {
            url: sorted(fields)
            for url, fields in sorted(urls_by_solution[name].items(), key=lambda x: x[0].lower())
        }

    return merged


def pick(record: Dict[str, Any], *keys: str) -> str:
    for k in keys:
        v = as_text(record.get(k))
        if v:
            return v
    return ""


def bullet_lines(items: List[Tuple[str, str]]) -> List[str]:
    out = []
    for label, val in items:
        if val:
            out.append(f"- **{label}:** {val}")
    if not out:
        out = ["- Not specified in the source data."]
    return out


def build_factsheet(record: Dict[str, Any]) -> str:
    name = record["name"]
    today = date.today().isoformat()

    what_it_is = pick(record, "storage_details", "Purpose", "Storage type")
    phases = pick(record, "positioning_datalyfecycle", "Longterm storage")
    collaboration = pick(record, "collaboration", "Sharing and Collaboration")
    sensitivity = pick(record, "sensitivity_classification", "Data sensitivity classification", "availability_classification")
    capacity = pick(record, "capacity", "Storage capacity")
    file_size = pick(record, "max_file_size", "Maximum File size")

    best_fit = []
    if phases:
        best_fit.append(f"- Lifecycle support: {phases}")
    if collaboration:
        best_fit.append(f"- Collaboration model: {collaboration}")
    if sensitivity:
        best_fit.append(f"- Data sensitivity profile: {sensitivity}")
    if capacity:
        best_fit.append(f"- Capacity context: {capacity}")
    if not best_fit:
        best_fit = ["- Use this option when it matches your policy, collaboration, and lifecycle requirements."]

    less_fit = []
    rec_notes = pick(record, "Recommendation notes")
    if rec_notes:
        less_fit.append(f"- Known caveat: {rec_notes}")
    if not less_fit:
        less_fit = ["- Less suitable when critical requirements (security, collaboration, lifecycle) are not supported."]

    security = bullet_lines([
        ("GDPR compliant", pick(record, "GDPR_compliant")),
        ("Sensitivity classification", pick(record, "sensitivity_classification", "Data sensitivity classification")),
        ("Availability classification", pick(record, "availability_classification")),
        ("Data CIA classification", pick(record, "Data CIA")),
        ("Encryption", pick(record, "file_encryption (by default", "File encryption")),
        ("Data location", pick(record, "data_location", "Storage location")),
    ])

    collaboration_access = bullet_lines([
        ("Collaboration", collaboration),
        ("Collaboration notes", pick(record, "collaboration_notes")),
        ("Access control", pick(record, "data_accesscontrolfeatures")),
        ("How to access", pick(record, "How to access")),
        ("Transfer protocol", pick(record, "Data transfer protocol")),
        ("Transfer protocol notes", pick(record, "Data transfer protocol notes")),
    ])

    storage_perf = bullet_lines([
        ("Capacity", capacity),
        ("Max file size", file_size),
        ("File formats", pick(record, "file_formats", "File formats")),
        ("Sync support", pick(record, "sync_equipment", "Sync to Client")),
    ])

    backup_versioning = bullet_lines([
        ("Backups", pick(record, "back-ups (by default)", "Back-ups & Recovery")),
        ("Versioning", pick(record, "data_versioning (by default)", "Versioning")),
        ("Complete data deletion", pick(record, "complete_data_deletion2", "Complete data deletion")),
    ])

    costs = bullet_lines([
        ("Price per TB/month", pick(record, "Price per TB / month")),
        ("Price up front", pick(record, "Price up front")),
        ("Cost model", pick(record, "costmodel", "Costs and eligibility")),
    ])

    support = bullet_lines([
        ("Supported by UU", pick(record, "Supported by UU?")),
        ("Host", pick(record, "Host")),
        ("Where to request", pick(record, "Where to request")),
        ("Request notes", pick(record, "Where to request notes")),
        ("Internal contact point RDM", pick(record, "Internal_contact_point_RDM", "Internal contact point for RDM")),
        ("End-of-life", pick(record, "End-of-Life")),
    ])

    link_lines = []
    links = record.get("_links", {})
    if isinstance(links, dict) and links:
        for url, fields in links.items():
            src = ", ".join(fields)
            link_lines.append(f"- {url}\n  - Mentioned in fields: `{src}`")
    else:
        link_lines = ["- No explicit URLs found in current source rows."]

    source_rows = [f"- `{src}`" for src in record.get("_sources", [])] or ["- Not available"]

    sections = [
        f"# {name}",
        "",
        f"Last reviewed: {today}",
        "",
        "## At a glance",
        what_it_is if what_it_is else "Storage solution in the DISC storage catalog.",
        "",
        "## Best fit",
        *best_fit,
        "",
        "## Less suitable when",
        *less_fit,
        "",
        "## Research lifecycle coverage",
        *bullet_lines([("Lifecycle phases", phases)]),
        "",
        "## Collaboration and access",
        *collaboration_access,
        "",
        "## Storage profile",
        *storage_perf,
        "",
        "## Security and privacy",
        *security,
        "",
        "## Backup and versioning",
        *backup_versioning,
        "",
        "## Costs and ownership",
        *costs,
        "",
        "## Support and governance",
        *support,
        "",
        "## Canonical links",
        *link_lines,
        "",
        "## Internal source rows",
        *source_rows,
        "",
    ]
    return "\n".join(sections).strip() + "\n"


def main() -> None:
    FACTSHEETS_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old markdown factsheets first, then rebuild from scratch.
    for md in FACTSHEETS_DIR.glob("*.md"):
        md.unlink()

    rows = collect_rows_storage_data_2()
    merged = merge_rows(rows)

    generated = []
    jsonl_lines = []

    for name in sorted(merged.keys(), key=lambda x: x.lower()):
        record = merged[name]
        md = build_factsheet(record)
        slug = slugify(name)
        path = FACTSHEETS_DIR / f"{slug}.md"
        path.write_text(md, encoding="utf-8")
        generated.append(md)

        jsonl_lines.append(json.dumps({
            "id": slug,
            "name": name,
            "text": md,
            "source": "merged_storage_data",
        }, ensure_ascii=False))

    (FACTSHEETS_DIR / "all_factsheets.md").write_text("\n\n---\n\n".join(generated), encoding="utf-8")
    (FACTSHEETS_DIR / "factsheets.jsonl").write_text("\n".join(jsonl_lines), encoding="utf-8")

    print(f"Generated {len(generated)} factsheets in {FACTSHEETS_DIR}")
    print(f"- {FACTSHEETS_DIR / 'all_factsheets.md'}")
    print(f"- {FACTSHEETS_DIR / 'factsheets.jsonl'}")


if __name__ == "__main__":
    main()
