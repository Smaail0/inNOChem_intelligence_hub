"""Verify whether the HSDB provider returns compound-level annotations.

This diagnostic script does not modify application state. It uses the HSDB topic
catalog as an allowlist, queries by PubChem CID through the provider, and prints
fact provenance/evidence snippets so catalog-only placeholders are easy to spot.
"""

from __future__ import annotations

from pathlib import Path
import sys
import textwrap
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.providers.hsdb_provider import HsdbProvider, TOPIC_GROUPS  # noqa: E402


COMPOUNDS = [
    {"canonical_name": "acetone", "pubchem_cid": 180, "cas": "67-64-1"},
    {"canonical_name": "ethanol", "pubchem_cid": 702, "cas": "64-17-5"},
    {"canonical_name": "DMF", "pubchem_cid": 6228, "cas": "68-12-2"},
    {"canonical_name": "NMP", "pubchem_cid": 13387, "cas": "872-50-4"},
    {"canonical_name": "THF", "pubchem_cid": 8028, "cas": "109-99-9"},
]


def main() -> None:
    provider = HsdbProvider()
    catalog_topics = provider.topic_catalog.allowed_topics()
    loaded_groups = _group_topics(catalog_topics)

    print("HSDB Provider Verification")
    print("=" * 80)
    print(f"provider loaded: {provider.name}")
    print(f"topic catalog: {provider.topic_catalog.catalog_path}")
    print(f"HSDB topic groups loaded: {len(loaded_groups)}")
    for group, topics in loaded_groups.items():
        print(f"- {group}: {len(topics)} topics")
        for topic in topics:
            print(f"  - {topic.heading} (priority={topic.priority}, catalog_count={topic.catalog_count})")
    print()

    total_facts = 0
    for compound in COMPOUNDS:
        facts = provider.get_facts(compound)
        total_facts += len(facts)
        print("-" * 80)
        print(
            "compound: "
            f"{compound['canonical_name']} | CID {compound['pubchem_cid']} | CAS {compound['cas']}"
        )
        print("compound-level facts returned:", "yes" if facts else "no")
        print(f"fact count: {len(facts)}")

        if not facts:
            print("interpretation: no HSDB compound-level annotations were returned for this CID.")
            print("catalog-only placeholder facts: no")
            continue

        catalog_only_count = sum(_looks_catalog_only(fact) for fact in facts)
        print("catalog-only placeholder facts:", "yes" if catalog_only_count else "no")
        print("real PubChem/PUG-View provenance:", "yes" if any(_looks_pug_view(f) for f in facts) else "no")

        for index, fact in enumerate(facts, start=1):
            print(f"\n  fact {index}")
            print(f"  attribute: {fact.attribute}")
            print(f"  value: {_snippet(fact.value)}")
            print(f"  provenance: {_snippet(fact.provenance, width=220)}")
            print(f"  evidence snippet: {_snippet(fact.evidence, width=220)}")
        print()

    print("=" * 80)
    print(f"total compound-level facts returned: {total_facts}")
    if total_facts:
        print("overall interpretation: HSDB is returning compound-level annotations.")
    else:
        print(
            "overall interpretation: HSDB returned no compound-level annotations in this run. "
            "If network access was available, this suggests the current extraction path is not "
            "finding HSDB topic sections for these CIDs."
        )


def _group_topics(topics) -> dict[str, list[Any]]:
    grouped = {group: [] for group in TOPIC_GROUPS}
    for topic in topics:
        grouped.setdefault(topic.group, []).append(topic)
    return grouped


def _snippet(value: Any, width: int = 180) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    return textwrap.shorten(text, width=width, placeholder="...")


def _looks_catalog_only(fact) -> bool:
    text = f"{fact.evidence} {fact.provenance}".casefold()
    return "catalog_count" in text and "pubchem pug-view hsdb annotation" not in text


def _looks_pug_view(fact) -> bool:
    text = f"{fact.evidence} {fact.provenance}".casefold()
    return "pubchem pug-view" in text


if __name__ == "__main__":
    main()
