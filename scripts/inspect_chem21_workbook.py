"""Inspect CHEM21 workbook sheets and target solvent occurrences."""

from __future__ import annotations

from pathlib import Path
import sys
import zipfile
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest_chem21 import _XlsxReader


INPUT_PATH = Path("data/raw/chem21/c5gc01008j1.xlsx")
TARGET_TERMS = [
    "THF",
    "DMF",
    "NMP",
    "1,4-dioxane",
    "tetrahydrofuran",
    "dimethylformamide",
    "109-99-9",
    "68-12-2",
    "872-50-4",
    "123-91-1",
]


def main() -> None:
    workbook = _XlsxReader(INPUT_PATH)
    print(f"Workbook: {INPUT_PATH}", flush=True)
    print("Sheet names:", flush=True)
    for sheet_name, sheet_path in workbook.sheet_paths:
        rows = workbook.read_sheet(sheet_path)
        print(f"- {sheet_name} ({sheet_path})", flush=True)
        print(f"  dimensions: {_dimension(INPUT_PATH, sheet_path)}", flush=True)
        print(f"  row_count: {len(rows)}", flush=True)
        print("  header candidates:", flush=True)
        for row_number, cells in _header_candidates(rows):
            print(f"    row {row_number}: {cells}", flush=True)
        print("  target rows:", flush=True)
        hits = _target_rows(rows)
        if not hits:
            print("    none", flush=True)
        for row_number, cells in hits:
            print(f"    row {row_number}: {cells}", flush=True)


def _dimension(input_path: Path, sheet_path: str) -> str:
    with zipfile.ZipFile(input_path) as archive:
        root = ET.fromstring(archive.read(sheet_path))
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    dimension = root.find("a:dimension", ns)
    return dimension.attrib.get("ref", "unknown") if dimension is not None else "unknown"


def _header_candidates(rows: list[tuple[int, dict[str, str]]]) -> list[tuple[int, dict[str, str]]]:
    candidates = []
    for row_number, cells in rows:
        text = " ".join(cells.values()).casefold()
        if "solvent" in text or "cas" in text or row_number <= 3:
            candidates.append((row_number, {key: value for key, value in cells.items() if value}))
        if len(candidates) >= 8:
            break
    return candidates


def _target_rows(rows: list[tuple[int, dict[str, str]]]) -> list[tuple[int, dict[str, str]]]:
    hits = []
    for row_number, cells in rows:
        text = " | ".join(str(value) for value in cells.values()).casefold()
        if any(term.casefold() in text for term in TARGET_TERMS):
            hits.append((row_number, {key: value for key, value in cells.items() if value}))
    return hits


if __name__ == "__main__":
    main()
