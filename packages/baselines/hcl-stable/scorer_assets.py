"""Pinned scoring data; paths are explicit outside the monorepo."""
from hashlib import sha256
import os
from pathlib import Path


def resolve_asset(relative_path: str, environment: str, expected_hash: str) -> Path:
    configured = os.getenv(environment)
    if configured:
        path = Path(configured).expanduser().resolve()
    else:
        path = next((parent / "experiments/data" / relative_path
                     for parent in Path(__file__).resolve().parents
                     if (parent / "experiments/data" / relative_path).is_file()), None)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Missing scoring asset {relative_path}; set {environment} to its file path.")
    if sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"Scoring asset hash mismatch: {path}")
    return path


def gpt2_tokenizer_path() -> Path:
    return resolve_asset("tokenizers/gpt2/tokenizer.json", "TRACE_GPT2_TOKENIZER",
                         "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6")


def punkt_tokenizer_path() -> Path:
    return resolve_asset("nltk/punkt/english.pickle", "TRACE_PUNKT_TOKENIZER",
                         "dda37972ae88998a6fd3e3ec002697a6bd362b32d050fda7d7ca5276873092aa")
