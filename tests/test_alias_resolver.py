import sqlite3
import tempfile
import unittest
from pathlib import Path

from services.alias_resolver import AliasResolver, resolve_alias


class AliasResolverTests(unittest.TestCase):
    def test_resolves_synonym_to_pubchem_cid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "synonyms.sqlite"
            _write_alias_db(db_path)

            resolver = AliasResolver(db_path)
            try:
                result = resolver.resolve("dimethyl ketone")
            finally:
                resolver.close()

        self.assertIsNotNone(result)
        self.assertEqual(result.cid, 180)

    def test_missing_database_is_non_fatal_for_convenience_lookup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = resolve_alias("acetone", db_path=str(Path(temp_dir) / "missing.sqlite"))

        self.assertIsNone(result)


def _write_alias_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE pubchem_synonyms (synonym_norm TEXT, synonym TEXT, cid INTEGER)"
        )
        conn.execute(
            "INSERT INTO pubchem_synonyms VALUES (?, ?, ?)",
            ("dimethyl ketone", "dimethyl ketone", 180),
        )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    unittest.main()
