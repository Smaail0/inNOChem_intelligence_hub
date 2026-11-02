"""Cache and verify the local Hugging Face chemical NER model.

This script downloads model files only. It does not use the Hugging Face
Inference API and does not send document text to any external inference service.
After caching, it verifies local-only tokenizer/model/pipeline loading.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import snapshot_download
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from core.config import CHEM_NER_CACHE_DIR, CHEM_NER_MODEL_NAME


SMOKE_TEXT = "DMF, NMP, THF, 2-MeTHF, Cyrene and \u03b3-valerolactone were evaluated."


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Cache and verify the chemical NER model locally.")
    parser.add_argument("--model-name", default=CHEM_NER_MODEL_NAME)
    parser.add_argument("--cache-dir", default=CHEM_NER_CACHE_DIR)
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Verify using only previously cached model files; do not download.",
    )
    args = parser.parse_args()

    print(f"Checking model files for: {args.model_name}")
    print(f"Cache directory: {args.cache_dir}")
    snapshot_path = snapshot_download(
        repo_id=args.model_name,
        cache_dir=args.cache_dir,
        local_files_only=args.local_only,
    )
    print(f"Snapshot cached at: {snapshot_path}")

    print("Verifying tokenizer from local cache...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        local_files_only=True,
    )

    print("Verifying AutoModelForTokenClassification from local cache...")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        local_files_only=True,
    )

    print("Verifying local token-classification pipeline...")
    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )
    outputs = ner_pipeline(SMOKE_TEXT)

    print("Smoke text:")
    print(SMOKE_TEXT)
    print("Pipeline outputs:")
    print(outputs)
    print("Local chemical NER model cache verification complete.")


if __name__ == "__main__":
    main()
