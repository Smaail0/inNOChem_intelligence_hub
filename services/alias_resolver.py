from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import threading
from typing import Optional


DEFAULT_ALIAS_DB = Path("data/knowledge/pubchem_synonyms.sqlite")


def normalize_alias(value: str) -> str:
    """Normalize user/NER chemical names for exact alias lookup."""
    return " ".join(value.strip().lower().split())


@dataclass(frozen=True)
class AliasResolution:
    query: str
    query_norm: str
    cid: int
    matched_synonym: str
    matched_synonym_norm: str
    confidence: float
    source: str = "pubchem_synonyms"


class AliasResolver:
    """
    Fast local resolver for PubChem synonym -> CID lookup.

    This service should run before PubChem validation:
        raw NER term -> alias resolver -> PubChem CID -> provider validation

    It is intentionally exact-match first. Fuzzy matching can be added later,
    but should not be mixed into this foundational resolver until evaluated.
    """

    def __init__(self, db_path: Path | str = DEFAULT_ALIAS_DB) -> None:
        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"PubChem synonym database not found: {self.db_path}. "
                "Run scripts/import_pubchem_synonyms.py first."
            )

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()

    def close(self) -> None:
        self._conn.close()

    def resolve(self, query: str) -> Optional[AliasResolution]:
        """
        Resolve one chemical alias to a PubChem CID using exact normalized lookup.

        If multiple CIDs share the same synonym, the first CID is returned.
        Use resolve_all() when ambiguity matters.
        """
        query_norm = normalize_alias(query)

        if not query_norm:
            return None

        with self._lock:
            row = self._conn.execute(
                """
                SELECT synonym_norm, synonym, cid
                FROM pubchem_synonyms
                WHERE synonym_norm = ?
                LIMIT 1
                """,
                (query_norm,),
            ).fetchone()

        if row is None:
            return None

        return AliasResolution(
            query=query,
            query_norm=query_norm,
            cid=int(row["cid"]),
            matched_synonym=str(row["synonym"]),
            matched_synonym_norm=str(row["synonym_norm"]),
            confidence=1.0,
        )

    def resolve_all(self, query: str, limit: int = 10) -> list[AliasResolution]:
        """
        Return all candidate CIDs for an alias.

        This is useful for ambiguous names such as:
        - xylene
        - vitamin b12
        - salts / mixtures / parent compounds
        """
        query_norm = normalize_alias(query)

        if not query_norm:
            return []

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT synonym_norm, synonym, cid
                FROM pubchem_synonyms
                WHERE synonym_norm = ?
                LIMIT ?
                """,
                (query_norm, limit),
            ).fetchall()

        return [
            AliasResolution(
                query=query,
                query_norm=query_norm,
                cid=int(row["cid"]),
                matched_synonym=str(row["synonym"]),
                matched_synonym_norm=str(row["synonym_norm"]),
                confidence=1.0,
            )
            for row in rows
        ]

    def has_alias(self, query: str) -> bool:
        return self.resolve(query) is not None


@lru_cache(maxsize=1)
def get_alias_resolver(db_path: str = str(DEFAULT_ALIAS_DB)) -> AliasResolver:
    """Cached resolver for Streamlit/app usage."""
    return AliasResolver(Path(db_path))


def resolve_alias(query: str, db_path: str = str(DEFAULT_ALIAS_DB)) -> Optional[AliasResolution]:
    """Convenience wrapper for one-off alias resolution."""
    try:
        resolver = get_alias_resolver(db_path)
    except FileNotFoundError:
        return None
    return resolver.resolve(query)
