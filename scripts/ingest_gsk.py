"""Create a review scaffold for GSK solvent guide extraction.

The available GSK source is a PDF whose solvent tables are embedded as images.
This script therefore refuses to invent values. It writes a manually reviewable
CSV scaffold and asks the operator to use an official supplementary spreadsheet
when available.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import zipfile
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


INPUT_PATH = Path("data/raw/gsk/SSG-2011-published-paper.pdf")
OUTPUT_PATH = Path("data/knowledge/gsk_extraction_review.csv")

REVIEW_COLUMNS = [
    "solvent_name",
    "cas",
    "melting_point_c",
    "boiling_point_c",
    "waste_score",
    "environmental_impact_score",
    "health_score",
    "flammability_explosion_score",
    "reactivity_stability_score",
    "life_cycle_score",
    "legislation_flag",
    "ehs_red_flag",
    "extraction_confidence",
    "source_page",
    "source_figure",
    "review_status",
]

UNRELIABLE_PDF_MESSAGE = "Use the official GSK supplementary spreadsheet if available."
NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def ingest_gsk(
    raw_dir: str | Path = "data/raw/gsk",
    output_path: str | Path = OUTPUT_PATH,
) -> Path:
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    supplementary = _find_supplementary_source(raw_dir)
    if supplementary:
        rows = _read_supplementary_rows(supplementary)
        _write_review_rows(output_path, rows)
        return output_path
    create_gsk_review_scaffold(raw_dir / INPUT_PATH.name, output_path)
    raise RuntimeError(
        "GSK PDF tables are image-embedded and cannot be reliably extracted automatically. "
        f"{UNRELIABLE_PDF_MESSAGE}"
    )


def create_gsk_review_scaffold(
    input_path: str | Path = INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"GSK source PDF not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
    return output_path


def _find_supplementary_source(raw_dir: Path) -> Path | None:
    if not raw_dir.exists():
        return None
    for suffix in ("*.csv", "*.xlsx"):
        for path in sorted(raw_dir.glob(suffix)):
            if path.name != OUTPUT_PATH.name:
                return path
    return None


def _read_supplementary_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [_normalize_review_row(row) for row in csv.DictReader(handle)]
    if path.suffix.casefold() == ".xlsx":
        return [_normalize_review_row(row) for row in _read_first_xlsx_sheet(path)]
    raise ValueError(f"Unsupported GSK supplementary source: {path}")


def _write_review_rows(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_review_row(row: dict[str, str]) -> dict[str, str]:
    normalized = {column: "" for column in REVIEW_COLUMNS}
    lookup = {_normalize_header(key): value for key, value in row.items()}
    for column in REVIEW_COLUMNS:
        normalized[column] = str(lookup.get(_normalize_header(column), "")).strip()
    normalized["review_status"] = normalized["review_status"] or "pending"
    return normalized


def _normalize_header(value: str) -> str:
    return str(value or "").strip().casefold().replace(" ", "_").replace("-", "_")


def _read_first_xlsx_sheet(path: Path) -> list[dict[str, str]]:
    archive = zipfile.ZipFile(path)
    shared_strings = _read_shared_strings(archive)
    root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows: list[dict[str, str]] = []
    headers: dict[str, str] = {}
    for row in root.findall(".//a:row", NS):
        cells = {
            _column_name(cell.attrib["r"]): _cell_value(cell, shared_strings)
            for cell in row.findall("a:c", NS)
        }
        if not headers:
            headers = cells
            continue
        rows.append({headers[column]: value for column, value in cells.items() if column in headers})
    return rows


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [
        "".join(text.text or "" for text in item.findall(".//a:t", NS))
        for item in root.findall("a:si", NS)
    ]


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    value = cell.find("a:v", NS)
    if value is None or value.text is None:
        inline = cell.find(".//a:t", NS)
        return inline.text if inline is not None and inline.text is not None else ""
    text = value.text
    if cell.attrib.get("t") == "s":
        return shared_strings[int(text)]
    return text


def _column_name(cell_reference: str) -> str:
    return "".join(char for char in cell_reference if char.isalpha())


def main() -> None:
    try:
        output_path = ingest_gsk()
    except RuntimeError as error:
        print(f"[GSK] Wrote manual extraction review scaffold to {OUTPUT_PATH}", flush=True)
        raise SystemExit(str(error))
    print(f"[GSK] Wrote extraction review rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
