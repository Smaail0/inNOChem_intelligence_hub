"""Ingest the CHEM21 solvent guide workbook into normalized CSV.

The project intentionally avoids hardcoding solvent guide decisions in Python.
This script reads the official workbook from data/raw and writes a normalized
machine-readable table with source provenance for each row.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.providers.base import normalize_identity


INPUT_PATH = Path("data/raw/chem21/c5gc01008j1.xlsx")
OUTPUT_PATH = Path("data/knowledge/chem21_solvents.csv")
SOURCE = "CHEM21"
SOURCE_VERSION = "2015 supplementary c5gc01008j1"
SOURCE_REFERENCE = "CHEM21 solvent selection guide supplementary information, c5gc01008j1"

OUTPUT_COLUMNS = [
    "source",
    "source_version",
    "source_reference",
    "solvent_name",
    "normalized_name",
    "aliases",
    "cas",
    "family",
    "boiling_point_c",
    "flash_point_c",
    "worst_h3xx",
    "h4xx",
    "safety_score",
    "health_score",
    "environment_score",
    "ranking_by_default",
    "final_ranking",
    "notes",
    "provenance",
]

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def ingest_chem21(
    input_path: str | Path = INPUT_PATH,
    output_path: str | Path = OUTPUT_PATH,
) -> list[dict[str, str]]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"CHEM21 source workbook not found: {input_path}")

    workbook = _XlsxReader(input_path)
    normalized_rows = _collect_workbook_rows(workbook, input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(normalized_rows)
    return normalized_rows


class _XlsxReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.archive = zipfile.ZipFile(path)
        self.shared_strings = self._read_shared_strings()
        self.sheet_paths = self._read_sheet_paths()

    def _read_shared_strings(self) -> list[str]:
        root = ET.fromstring(self.archive.read("xl/sharedStrings.xml"))
        return [
            "".join(text.text or "" for text in item.findall(".//a:t", NS))
            for item in root.findall("a:si", NS)
        ]

    def read_sheet(self, sheet_path: str) -> list[tuple[int, dict[str, str]]]:
        root = ET.fromstring(self.archive.read(sheet_path))
        rows: list[tuple[int, dict[str, str]]] = []
        for row in root.findall(".//a:row", NS):
            row_number = int(row.attrib["r"])
            cells = {
                _column_name(cell.attrib["r"]): self._cell_value(cell)
                for cell in row.findall("a:c", NS)
            }
            rows.append((row_number, cells))
        return rows

    def _cell_value(self, cell: ET.Element) -> str:
        value = cell.find("a:v", NS)
        if value is None or value.text is None:
            inline = cell.find(".//a:t", NS)
            return inline.text if inline is not None and inline.text is not None else ""
        text = value.text
        if cell.attrib.get("t") == "s":
            return self.shared_strings[int(text)]
        return text

    def _read_sheet_paths(self) -> list[tuple[str, str]]:
        workbook = ET.fromstring(self.archive.read("xl/workbook.xml"))
        rels = ET.fromstring(self.archive.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        relationships_namespace = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        sheets = []
        for sheet in workbook.findall(".//a:sheet", NS):
            target = relmap[sheet.attrib[relationships_namespace]]
            sheets.append((sheet.attrib["name"], f"xl/{target.lstrip('/')}"))
        return sheets


def _collect_workbook_rows(workbook: _XlsxReader, input_path: Path) -> list[dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for sheet_name, sheet_path in workbook.sheet_paths:
        rows = workbook.read_sheet(sheet_path)
        source_table = _detect_source_table(rows)
        if source_table == "no_solvent_table":
            continue
        headers = _header_candidates(rows)
        for row_number, row in _solvent_rows(rows):
            normalized_row = _normalize_sheet_row(
                input_path=input_path,
                sheet_name=sheet_name,
                source_table=source_table,
                row_number=row_number,
                row=row,
                headers=headers,
            )
            if not normalized_row:
                continue
            key = _record_key(normalized_row)
            if key not in records:
                records[key] = normalized_row
            else:
                records[key] = _merge_records(records[key], normalized_row)
    return sorted(records.values(), key=lambda item: (item["family"], item["solvent_name"]))


def _normalize_sheet_row(
    input_path: Path,
    sheet_name: str,
    source_table: str,
    row_number: int,
    row: dict[str, str],
    headers: dict[str, str],
) -> dict[str, str] | None:
    solvent_name = row.get("B", "").strip()
    cas = row.get("C", "").strip()
    if not _is_valid_solvent_name(solvent_name):
        return None

    if source_table == "results summary":
        safety_score = row.get("D", "").strip()
        health_score = row.get("E", "").strip()
        environment_score = row.get("F", "").strip()
        ranking = row.get("J", "").strip()
        boiling_point = ""
        flash_point = ""
        family = row.get("A", "").strip()
        worst_h3xx = ""
        h4xx = ""
        notes = ""
    else:
        safety_score = row.get("CZ", "").strip()
        health_score = row.get("DA", "").strip()
        environment_score = row.get("DB", "").strip()
        ranking = row.get("DQ", "").strip()
        boiling_point = row.get("D", "").strip()
        flash_point = row.get("J", "").strip()
        family = row.get("A", "").strip()
        worst_h3xx = _active_hazard_codes(row, headers, _is_h3xx_code)
        h4xx = _active_hazard_codes(row, headers, _is_h4xx_code)
        notes = _notes_for_row(row)

    return {
        "source": SOURCE,
        "source_version": SOURCE_VERSION,
        "source_reference": SOURCE_REFERENCE,
        "solvent_name": solvent_name,
        "normalized_name": normalize_identity(solvent_name),
        "aliases": _aliases_for_name(solvent_name),
        "cas": cas,
        "family": family,
        "boiling_point_c": boiling_point,
        "flash_point_c": flash_point,
        "worst_h3xx": worst_h3xx,
        "h4xx": h4xx,
        "safety_score": safety_score,
        "health_score": health_score,
        "environment_score": environment_score,
        "ranking_by_default": ranking,
        "final_ranking": ranking,
        "notes": notes,
        "provenance": (
            f"{input_path}::{sheet_name} row {row_number}; "
            f"source_table={source_table}"
        ),
    }


def _solvent_rows(rows: list[tuple[int, dict[str, str]]]) -> list[tuple[int, dict[str, str]]]:
    solvent_rows: list[tuple[int, dict[str, str]]] = []
    current_family = ""
    for row_number, row in rows:
        solvent_name = row.get("B", "").strip()
        if not _is_valid_solvent_name(solvent_name):
            continue
        family = row.get("A", "").strip()
        if family:
            current_family = family
        row["A"] = current_family
        solvent_rows.append((row_number, row))
    return solvent_rows


def _detect_source_table(rows: list[tuple[int, dict[str, str]]]) -> str:
    sheet_values = [
        value.strip().casefold()
        for _, row in rows
        for value in row.values()
        if value
    ]
    has_solvent_header = any("solvent" in value for value in sheet_values)
    has_cas_header = any("cas" in value for value in sheet_values)
    if has_solvent_header and has_cas_header:
        if any("ranking by default" in value for value in sheet_values):
            return "results summary"
        return "detailed solvent data"
    return "no_solvent_table"


def _header_candidates(rows: list[tuple[int, dict[str, str]]]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for _, row in rows:
        values = {value.strip().casefold() for value in row.values() if value}
        if any("solvent" in value for value in values) or any("cas" in value for value in values):
            headers.update({column: value for column, value in row.items() if value})
    return headers


def _merge_records(existing: dict[str, str], incoming: dict[str, str]) -> dict[str, str]:
    merged = dict(existing)
    for column in OUTPUT_COLUMNS:
        if column == "provenance":
            merged[column] = " | ".join(
                dict.fromkeys(
                    value
                    for value in [existing.get(column, ""), incoming.get(column, "")]
                    if value
                )
            )
        elif column == "aliases":
            aliases = []
            for value in [existing.get(column, ""), incoming.get(column, "")]:
                aliases.extend(alias for alias in value.replace(";", "|").split("|") if alias)
            merged[column] = "|".join(dict.fromkeys(aliases))
        elif not merged.get(column) and incoming.get(column):
            merged[column] = incoming[column]
    return merged


def _record_key(row: dict[str, str]) -> str:
    cas = row.get("cas", "").strip()
    if cas:
        return f"cas:{cas}"
    return f"name:{row.get('normalized_name', '')}"


def _is_valid_solvent_name(solvent_name: str) -> bool:
    if not solvent_name:
        return False
    normalized = solvent_name.strip().casefold()
    return normalized not in {"solvents", "solvents\xa0", "solvent name"}


def _active_hazard_codes(
    row: dict[str, str],
    headers: dict[str, str],
    predicate,
) -> str:
    active_codes: list[str] = []
    for column, header in headers.items():
        if not predicate(header):
            continue
        if row.get(column, "").strip() in {"1", "2", "3", "4", "5", "6", "7", "9"}:
            code = _hazard_code(header)
            if code:
                active_codes.append(code)
    return "; ".join(dict.fromkeys(active_codes))


def _hazard_code(header: str) -> str:
    match = re.search(r"\b(?:EUH|H)\d{3}\b", header)
    return match.group(0) if match else ""


def _is_h3xx_code(header: str) -> bool:
    code = _hazard_code(header)
    return code.startswith("H3") or code.startswith("EUH")


def _is_h4xx_code(header: str) -> bool:
    return _hazard_code(header).startswith("H4")


def _aliases_for_name(solvent_name: str) -> str:
    aliases: list[str] = []
    for value in re.findall(r"\(([^)]+)\)", solvent_name):
        aliases.append(value.strip())
    aliases.extend(_IDENTITY_ALIAS_EXPANSIONS.get(normalize_identity(solvent_name), []))
    if "γ" in solvent_name:
        aliases.append(solvent_name.replace("γ", "gamma"))
    normalized = normalize_identity(solvent_name)
    if normalized != solvent_name.casefold():
        aliases.append(normalized)
    return "|".join(dict.fromkeys(alias for alias in aliases if alias))


_IDENTITY_ALIAS_EXPANSIONS = {
    "2-methyl-thf": [
        "2-MeTHF",
        "2-methyltetrahydrofuran",
        "2-methyl tetrahydrofuran",
    ],
    "tbme": [
        "MTBE",
        "tert-butyl methyl ether",
        "methyl tert-butyl ether",
    ],
    "sulpholane": [
        "sulfolane",
    ],
    "gamma-valerolactone": [
        "GVL",
        "γ-valerolactone",
    ],
    "fames methyl oleate": [
        "methyl oleate",
    ],
}


def _notes_for_row(row: dict[str, str]) -> str:
    notes = []
    reach = row.get("CQ", "").strip()
    if reach:
        notes.append(f"REACH registration flag: {reach}")
    significant = row.get("DP", "").strip()
    if significant:
        notes.append(f"Most significant total flag: {significant}")
    return "; ".join(notes)


def _row_key(solvent_name: str, cas: str) -> str:
    return f"{normalize_identity(solvent_name)}|{cas.strip()}"


def _column_name(cell_reference: str) -> str:
    return "".join(char for char in cell_reference if char.isalpha())


def main() -> None:
    rows = ingest_chem21()
    print(f"[CHEM21] Wrote {len(rows)} rows to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
