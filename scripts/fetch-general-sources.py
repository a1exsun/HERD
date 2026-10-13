"""Fetch the pinned G-test sources into experiments/data/raw/general-v2.

The sources, their Hugging Face dataset revisions, files, sizes and SHA-256 hashes are read from
the tracked build report datasets/general-test-v1.json (`rawSources`). Each file is
downloaded at its pinned revision and kept only if its size and hash match; a file already in place
with the right hash is not downloaded again. Standard library only; set HF_TOKEN if the Hub asks
for authentication.

    python3 scripts/fetch-general-sources.py

Then build the panels with scripts/prepare-general-test.py.
"""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "datasets/general-test-v1.json"
RAW = ROOT / "experiments/data/raw/general-v2"


def fetch(repository: str, revision: str, file: str) -> bytes:
    url = f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{file}"
    request = urllib.request.Request(url)
    if os.environ.get("HF_TOKEN"):
        request.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main() -> None:
    sources = json.loads(REPORT.read_text())["rawSources"]
    failed = []
    for repository, source in sources.items():
        for entry in source["files"]:
            target = RAW / repository.replace("/", "__") / entry["file"].replace("/", "__")
            if target.is_file() and sha256(target.read_bytes()).hexdigest() == entry["sha256"]:
                print(f"  present  {target.relative_to(ROOT)}")
                continue
            data = fetch(repository, source["revision"], entry["file"])
            if len(data) != entry["bytes"] or sha256(data).hexdigest() != entry["sha256"]:
                failed.append(f"{repository} {entry['file']} at {source['revision']}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            print(f"  fetched  {target.relative_to(ROOT)}  {len(data):,} B")
    if failed:
        sys.exit("Size or hash differs from the pinned source:\n  " + "\n  ".join(failed))
    (RAW / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
