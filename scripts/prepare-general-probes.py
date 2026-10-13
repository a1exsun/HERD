"""Reserve a fixed, bounded non-test general set and a disjoint held-out set."""
from hashlib import sha1, sha256
from pathlib import Path
import json
import os
from urllib.request import urlopen

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
TASKS = [
    ("037", "task037_qasc_generate_related_fact.json", "factual", "rouge_l", "default"),
    ("061", "task061_ropes_answer_generation.json", "reading", "exact_match", "default"),
    ("116", "task116_com2sense_commonsense_reasoning.json", "commonsense", "exact_match", "default"),
    ("118", "task118_semeval_2019_task10_open_vocabulary_mathematical_answer_generation.json", "mathematics", "exact_match", "default"),
    ("117", "task117_spl_translation_en_de.json", "multilingual", "rouge_l", "xlingual"),
]
SEED = 42
BUDGET = 8192


def normalize(value):
    import unicodedata
    return " ".join(unicodedata.normalize("NFKC", value).split())


def main():
    target = EXP / "data/general-probes/v1"
    if target.exists():
        raise ValueError(f"Immutable probe version already exists: {target}")
    tokenizer = AutoTokenizer.from_pretrained(EXP / "checkpoints/qwen3.5-4b/base", local_files_only=True)
    revision = json.loads((ROOT / 'config/datasets/trace-plus-17.json').read_text())['sourceRevisions']['superNiRepository']
    with urlopen(f'https://api.github.com/repos/allenai/natural-instructions/git/trees/{revision}?recursive=1', timeout=60) as response:
        tree = json.load(response)
    if tree['truncated']:
        raise ValueError('SuperNI source tree is incomplete')
    hashes = {item["path"]: item["sha"] for item in tree["tree"]}
    stream = json.loads((EXP / "data/trace-plus-17/manifest.json").read_text())
    prompts = set()
    inputs = set()
    for task in stream["tasks"]:
        for key in ["train", "validationGold", "testGold"]:
            for line in (EXP / "data/trace-plus-17" / task["files"][key]["path"]).read_text().splitlines():
                row = json.loads(line)
                prompts.add(normalize(row["prompt"]))
                if task["source"] == "superni":
                    inputs.add(normalize(row["prompt"].split("\n\nInput:\n", 1)[1].rsplit("\n\nOutput:", 1)[0]))
    general = []
    held_out = []
    sources = []
    for task_id, filename, category, metric, track in TASKS:
        source = ROOT / os.environ["HERO_TRACE_SOURCE_ROOT"] / "superni/natural-instructions-v2.8/tasks" / filename
        data = source.read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == hashes[f"tasks/{filename}"]
        raw = json.loads(data)
        unique = {}
        excluded = 0
        for index, item in enumerate(raw["Instances"]):
            references = list(dict.fromkeys(r for r in item["output"] if isinstance(r, str) and r.strip()))
            prompt = raw["Definition"][0].strip() + "\n\nInput:\n" + item["input"] + "\n\nOutput:"
            key = normalize(prompt)
            if not references or not normalize(item["input"]) or key in prompts or normalize(item["input"]) in inputs:
                excluded += 1
                continue
            if key in unique:
                unique[key]["references"] = list(dict.fromkeys(unique[key]["references"] + references))
                continue
            unique[key] = {
                "id": f"general:{task_id}:{item.get('id', index)}", "taskId": f"general:{task_id}",
                "prompt": prompt, "answer": references[0], "references": references,
                "metric": metric, "category": category,
                "metadata": {"source": "superni", "sourceTaskId": task_id, "metricTrack": track, "sourceIndex": index},
            }
        ordered = sorted(unique.values(), key=lambda r: sha256(f"{SEED}:{r['id']}".encode()).hexdigest())
        # Independent per-category budgets avoid allowing one long source to
        # consume all of the fixed general-state budget.
        selected = []
        used = 0
        for row in ordered:
            row["serializedTokens"] = 0
            for _ in range(3):
                row["serializedTokens"] = len(tokenizer.encode(json.dumps(row, ensure_ascii=False, separators=(",", ":")), add_special_tokens=False))
            if used + row["serializedTokens"] <= BUDGET // len(TASKS):
                selected.append(row)
                used += row["serializedTokens"]
            if len(selected) == 8:
                break
        assert selected, f"No general probes fit for {task_id}"
        selected_ids = {row["id"] for row in selected}
        tests = [row for row in ordered if row["id"] not in selected_ids][:40]
        assert len(tests) == 40
        # Freeze cross-category isolation as well as stream isolation.
        for row in selected + tests:
            key = normalize(row["prompt"])
            assert key not in prompts
            prompts.add(key)
        general.extend(selected)
        held_out.extend(tests)
        sources.append({"taskId": task_id, "category": category, "file": filename,
                        "sha256": sha256(data).hexdigest(), "sourceInstances": len(raw["Instances"]),
                        "excluded": excluded, "generalCount": len(selected), "testCount": len(tests), "generalTokens": used})
    assert sum(row["serializedTokens"] for row in general) <= BUDGET
    (target / "non-test").mkdir(parents=True)
    (target / "hidden-scorer").mkdir()
    files = []
    for path, rows in [(target / "non-test/general.jsonl", general), (target / "hidden-scorer/general-test.jsonl", held_out)]:
        text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
        path.write_text(text)
        files.append({"path": str(path.relative_to(target)), "records": len(rows), "sha256": sha256(text.encode()).hexdigest()})
    manifest = {"version": "hero-general-probes-v1", "seed": SEED, "tokenBudget": BUDGET,
                "realizedTokens": sum(r["serializedTokens"] for r in general), "sources": sources, "files": files,
                "selection": "SHA256(seed:id) order; up to 8 complete non-test records per category under equal token budgets; first 40 remaining records per category held out",
                "crossSplitPromptOverlap": 0, "streamPromptOrSuperNiInputOverlap": 0}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = ROOT / "datasets/general-probes-v1.json"
    report.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"general": len(general), "test": len(held_out), "tokens": manifest["realizedTokens"], "manifest": str(target / "manifest.json")}))


if __name__ == "__main__":
    main()
