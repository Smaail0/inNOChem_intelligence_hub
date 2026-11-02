from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from core.config import PROVIDER_REGISTRY_PATH


ALLOWED_PROVIDER_CATEGORIES = {
    "identity",
    "safety",
    "sustainability",
    "regulatory",
    "evidence",
}


DEFAULT_PROVIDER_REGISTRY: dict[str, Any] = {
    "providers": {
        "pubchem": {
            "label": "PubChem / RDKit",
            "category": "identity",
            "aliases": ["pubchem", "rdkit", "pubchem_rdkit"],
        },
        "hsdb": {
            "label": "HSDB",
            "category": "safety",
            "aliases": ["hsdb", "pubchem_hsdb", "pubchem_pug_view_hsdb"],
        },
        "chem21": {
            "label": "CHEM21",
            "category": "sustainability",
            "aliases": ["chem21", "chem_21"],
        },
        "gsk": {
            "label": "GSK",
            "category": "sustainability",
            "aliases": ["gsk", "gsk_solvent_sustainability_guide"],
        },
        "reach": {
            "label": "REACH",
            "category": "regulatory",
            "aliases": ["reach", "echa_reach"],
        },
        "svhc": {
            "label": "SVHC",
            "category": "regulatory",
            "aliases": ["svhc", "candidate_list"],
        },
        "clp": {
            "label": "CLP",
            "category": "regulatory",
            "aliases": ["clp"],
        },
        "dot": {
            "label": "DOT",
            "category": "regulatory",
            "aliases": ["dot", "us_dot"],
        },
        "rag_evidence": {
            "label": "RAG Evidence",
            "category": "evidence",
            "aliases": ["rag", "rag_evidence"],
        },
    }
}


@dataclass(frozen=True)
class ProviderDefinition:
    name: str
    label: str
    category: str
    aliases: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "category": self.category,
            "aliases": list(self.aliases),
        }


class ProviderRegistry:
    def __init__(self, providers: Mapping[str, ProviderDefinition], warnings_: list[str] | None = None):
        self.providers = dict(providers)
        self.warnings = list(warnings_ or [])
        self.alias_to_provider = self._build_alias_map()

    @classmethod
    def from_yaml(cls, path: str | Path = PROVIDER_REGISTRY_PATH) -> "ProviderRegistry":
        path = Path(path)
        warnings_: list[str] = []

        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        except FileNotFoundError:
            payload = DEFAULT_PROVIDER_REGISTRY
            warnings_.append(
                f"Provider registry not found at {path}; using built-in defaults."
            )
        except Exception as exc:
            payload = DEFAULT_PROVIDER_REGISTRY
            warnings_.append(
                f"Provider registry at {path} could not be loaded ({type(exc).__name__}: {exc}); using built-in defaults."
            )

        providers = cls._parse_providers(payload, warnings_)
        registry = cls(providers, warnings_)
        registry._emit_warnings()
        return registry

    @staticmethod
    def _parse_providers(payload: Mapping[str, Any], warnings_: list[str]) -> dict[str, ProviderDefinition]:
        provider_payload = payload.get("providers") if isinstance(payload, Mapping) else None
        if not isinstance(provider_payload, Mapping):
            warnings_.append("Provider registry is missing a 'providers' mapping.")
            provider_payload = DEFAULT_PROVIDER_REGISTRY["providers"]

        providers: dict[str, ProviderDefinition] = {}

        for raw_name, raw_definition in provider_payload.items():
            name = normalize_provider_key(str(raw_name))

            if not isinstance(raw_definition, Mapping):
                warnings_.append(f"Provider '{raw_name}' is malformed and was skipped.")
                continue

            label = raw_definition.get("label")
            category = normalize_provider_key(str(raw_definition.get("category") or ""))
            aliases = raw_definition.get("aliases", [])

            if not label:
                warnings_.append(f"Provider '{raw_name}' is missing label and was skipped.")
                continue

            if not category:
                warnings_.append(f"Provider '{raw_name}' is missing category and was skipped.")
                continue

            if category not in ALLOWED_PROVIDER_CATEGORIES:
                warnings_.append(
                    f"Provider '{raw_name}' uses unsupported category '{category}' and was skipped."
                )
                continue

            if not isinstance(aliases, list):
                warnings_.append(f"Provider '{raw_name}' aliases must be a list and were ignored.")
                aliases = []

            alias_keys = [normalize_provider_key(str(alias)) for alias in aliases if str(alias).strip()]
            alias_keys = _unique([name, *alias_keys])

            providers[name] = ProviderDefinition(
                name=name,
                label=str(label),
                category=category,
                aliases=tuple(alias_keys),
            )

        return providers

    def _build_alias_map(self) -> dict[str, str]:
        alias_to_provider: dict[str, str] = {}

        for provider_name, definition in self.providers.items():
            aliases = _unique([provider_name, *definition.aliases])

            for alias in aliases:
                normalized_alias = normalize_provider_key(alias)
                if not normalized_alias:
                    continue

                existing = alias_to_provider.get(normalized_alias)
                if existing and existing != provider_name:
                    self.warnings.append(
                        f"Duplicate provider alias '{normalized_alias}' maps to both '{existing}' and '{provider_name}'. Keeping '{existing}'."
                    )
                    continue

                alias_to_provider[normalized_alias] = provider_name

        return alias_to_provider

    def _emit_warnings(self) -> None:
        for message in self.warnings:
            warnings.warn(message, RuntimeWarning, stacklevel=2)

    def get_provider(self, provider_name: str | None) -> ProviderDefinition | None:
        canonical = self.normalize_provider_name(provider_name)
        return self.providers.get(canonical)

    def normalize_provider_name(self, provider_name: str | None) -> str:
        key = normalize_provider_key(provider_name or "")
        return self.alias_to_provider.get(key, key)

    def get_provider_category(self, provider_name: str | None) -> str | None:
        provider = self.get_provider(provider_name)
        return provider.category if provider else None

    def get_providers_by_category(self, category: str) -> list[ProviderDefinition]:
        normalized_category = normalize_provider_key(category)
        return [
            provider
            for provider in self.providers.values()
            if provider.category == normalized_category
        ]


@lru_cache(maxsize=8)
def get_provider_registry(path: str = PROVIDER_REGISTRY_PATH) -> ProviderRegistry:
    return ProviderRegistry.from_yaml(path)


def get_provider(provider_name: str | None) -> ProviderDefinition | None:
    return get_provider_registry().get_provider(provider_name)


def normalize_provider_name(provider_name: str | None) -> str:
    return get_provider_registry().normalize_provider_name(provider_name)


def get_provider_category(provider_name: str | None) -> str | None:
    return get_provider_registry().get_provider_category(provider_name)


def get_providers_by_category(category: str) -> list[ProviderDefinition]:
    return get_provider_registry().get_providers_by_category(category)


def normalize_provider_key(value: str) -> str:
    normalized = " ".join(str(value or "").strip().lower().split())
    normalized = normalized.replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)

    return result
