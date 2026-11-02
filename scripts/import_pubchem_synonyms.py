from __future__ import annotations

import argparse
import gzip
import re
import sqlite3
import time
from pathlib import Path


DEFAULT_INPUT = Path("data/raw/pubchem/CID-Synonym-filtered.gz")
DEFAULT_OUTPUT = Path("data/knowledge/pubchem_synonyms.sqlite")


def normalize_synonym(value: str) -> str:
    """Normalize a synonym for exact lookup."""
    return " ".join(value.strip().lower().split())


def should_keep_synonym(value: str) -> bool:
    """
    Keep useful chemistry aliases while skipping obvious noisy identifiers.

    This is intentionally conservative:
    - keeps normal chemical names
    - keeps CAS-like identifiers
    - skips very long names and common catalog/vendor identifiers
    """
    if not value:
        return False

    if len(value) < 2:
        return False

    if len(value) > 200:
        return False

    if value.isdigit():
        return False

    # CAS number pattern, e.g. 67-64-1
    if re.fullmatch(r"\d{2,7}-\d{2}-\d", value):
        return True

    blocked_fragments = [
        "chembl",
        "molport",
        "akos",
        "bindingdb",
        "maybridge",
        "selleck",
    ]

    if any(fragment in value for fragment in blocked_fragments):
        return False

    return True


def create_schema(conn: sqlite3.Connection) -> None:
    """
    Create only the table before import.

    Indexes are created after all inserts, which is much faster for large files.
    """
    conn.executescript(
        """
        DROP TABLE IF EXISTS pubchem_synonyms;

        CREATE TABLE pubchem_synonyms (
            synonym_norm TEXT NOT NULL,
            synonym TEXT NOT NULL,
            cid INTEGER NOT NULL
        );
        """
    )
    conn.commit()


def create_indexes(conn: sqlite3.Connection) -> None:
    print("Creating indexes...")

    conn.executescript(
        """
        CREATE INDEX idx_pubchem_synonyms_norm
        ON pubchem_synonyms (synonym_norm);

        CREATE INDEX idx_pubchem_synonyms_cid
        ON pubchem_synonyms (cid);
        """
    )

    conn.commit()


def import_synonyms(input_path: Path, output_path: Path, batch_size: int = 500_000) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    conn = sqlite3.connect(output_path)

    # Import performance settings.
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA cache_size = -200000;")

    create_schema(conn)

    inserted = 0
    skipped = 0
    batch: list[tuple[str, str, int]] = []
    start = time.time()

    with gzip.open(input_path, "rt", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\n")

            if not line:
                skipped += 1
                continue

            # Expected PubChem format:
            # CID<TAB>Synonym
            if "\t" in line:
                cid_text, synonym = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    skipped += 1
                    continue
                cid_text, synonym = parts

            try:
                cid = int(cid_text)
            except ValueError:
                skipped += 1
                continue

            synonym = synonym.strip()
            synonym_norm = normalize_synonym(synonym)

            if not synonym_norm:
                skipped += 1
                continue

            if not should_keep_synonym(synonym_norm):
                skipped += 1
                continue

            batch.append((synonym_norm, synonym, cid))

            if len(batch) >= batch_size:
                conn.executemany(
                    """
                    INSERT INTO pubchem_synonyms (synonym_norm, synonym, cid)
                    VALUES (?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
                inserted += len(batch)
                batch.clear()

                elapsed = time.time() - start
                print(
                    f"Inserted {inserted:,} rows | "
                    f"skipped {skipped:,} | "
                    f"read lines {line_number:,} | "
                    f"elapsed {elapsed:.1f}s"
                )

    if batch:
        conn.executemany(
            """
            INSERT INTO pubchem_synonyms (synonym_norm, synonym, cid)
            VALUES (?, ?, ?)
            """,
            batch,
        )
        conn.commit()
        inserted += len(batch)

    create_indexes(conn)

    print("Analyzing database...")
    conn.execute("ANALYZE;")
    conn.commit()

    conn.close()

    elapsed = time.time() - start
    print("Done.")
    print(f"Inserted rows: {inserted:,}")
    print(f"Skipped rows: {skipped:,}")
    print(f"Output DB: {output_path}")
    print(f"Elapsed: {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import PubChem CID-Synonym mappings into a local SQLite alias database."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=500_000)
    args = parser.parse_args()

    import_synonyms(args.input, args.output, args.batch_size)


if __name__ == "__main__":
    main()
