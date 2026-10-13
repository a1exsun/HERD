"""Build the five-category general-capability panels (the G-test and its development and probe panels).

Sources are pinned by HuggingFace commit revision and archived under
experiments/data/raw/general-v2 by scripts/fetch-general-sources.py. Selection is
deterministic under SHA256(seed:id); every record is checked for prompt overlap
against the whole TRACE+17 stream and against SuperNI inputs before it is used.
"""
from hashlib import sha256
from pathlib import Path
import json
import unicodedata

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
RAW = EXP / "data/raw/general-v2"
TARGET = EXP / "data/general-test/v1"
SEED = 42
LETTERS = "ABCDEFGH"

# Five equal-weight categories. 64 non-test records stratified
# 13/13/13/13/12 - the short category is the last in canonical order.
CATEGORIES = [
    ("general-knowledge", "mmlu", 13),
    ("mathematical-reasoning", "gsm8k", 13),
    ("commonsense-reasoning", "hellaswag", 13),
    ("instruction-following", "ifeval", 13),
    ("multilingual-understanding", "xcopa", 12),
]
TEST_PER_CATEGORY = 200
PROBE_PER_CATEGORY = 40
DEV_PER_CATEGORY = 40


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", value).split())


def order(rows):
    """Hash order, but honour an explicit stratification rank when a builder sets one."""
    return sorted(rows, key=lambda r: (r.get("_rank", 0), sha256(f"{SEED}:{r['id']}".encode()).hexdigest()))


def choice_prompt(instruction, question, options):
    body = "\n".join(f"{LETTERS[i]}. {text}" for i, text in enumerate(options))
    return f"{instruction}\n\nQuestion:\n{question}\n\nChoices:\n{body}\n\nAnswer:"


def table(relative):
    return pq.read_table(RAW / relative).to_pydict()


def build_mmlu():
    data = table("cais__mmlu/all__test-00000-of-00001.parquet")
    rows = []
    for i, (question, subject, choices, answer) in enumerate(
            zip(data["question"], data["subject"], data["choices"], data["answer"])):
        if not (0 <= answer < len(choices)) or len(choices) != 4:
            continue
        rows.append({
            "id": f"general:mmlu:{i}",
            "prompt": choice_prompt("Choose an answer for the following question.", question, choices),
            "answer": LETTERS[answer],
            "metric": "leading_choice_accuracy",
            "metadata": {"source": "mmlu", "subject": subject, "sourceIndex": i},
        })
    return rows


def build_gsm8k():
    data = table("openai__gsm8k/main__test-00000-of-00001.parquet")
    rows = []
    for i, (question, answer) in enumerate(zip(data["question"], data["answer"])):
        if "####" not in answer:
            continue
        final = answer.rsplit("####", 1)[1].strip()
        rows.append({
            "id": f"general:gsm8k:{i}",
            "prompt": f"Solve the following problem.\n\nQuestion:\n{question}\n\nAnswer:",
            "answer": final,
            "metric": "final_number_match",
            "metadata": {"source": "gsm8k", "sourceIndex": i},
        })
    return rows


def build_hellaswag():
    data = table("Rowan__hellaswag/data__validation-00000-of-00001.parquet")
    rows = []
    for i, (context, endings, label, activity) in enumerate(
            zip(data["ctx"], data["endings"], data["label"], data["activity_label"])):
        if label is None or str(label) == "" or len(endings) != 4:
            continue
        rows.append({
            "id": f"general:hellaswag:{i}",
            "prompt": choice_prompt("Choose the most plausible continuation of the passage.", context, endings),
            "answer": LETTERS[int(label)],
            "metric": "leading_choice_accuracy",
            "metadata": {"source": "hellaswag", "activityLabel": activity, "sourceIndex": i},
        })
    return rows


def build_ifeval():
    path = RAW / "google__IFEval/ifeval_input_data.jsonl"
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        identifiers = record["instruction_id_list"]
        arguments = [{k: v for k, v in kw.items() if v is not None} for kw in record["kwargs"]]
        if len(identifiers) != len(arguments):
            continue
        rows.append({
            "id": f"general:ifeval:{record['key']}",
            "prompt": record["prompt"],
            # Verifier-scored: the gold is the instruction set, not a reference string.
            "metric": "ifeval_strict",
            "metadata": {"source": "ifeval", "sourceKey": record["key"],
                         "instructionIdList": identifiers, "instructionKwargs": arguments},
        })
    return rows


def build_xcopa():
    """Round-robin across the eleven native languages so the panel stays balanced."""
    question_text = {"cause": "What was the cause?", "effect": "What happened as a result?"}
    per_language = {}
    for path in sorted((RAW / "cambridgeltl__xcopa").glob("*.parquet")):
        language = path.name.split("__")[0]
        if language.startswith("translation-"):
            continue  # Machine-translated variants are not native multilingual data.
        data = pq.read_table(path).to_pydict()
        rows = []
        for premise, first, second, kind, label, index in zip(
                data["premise"], data["choice1"], data["choice2"],
                data["question"], data["label"], data["idx"]):
            if label not in (0, 1) or kind not in question_text:
                continue
            rows.append({
                "id": f"general:xcopa:{language}:{index}",
                "prompt": choice_prompt(
                    "Choose the more plausible alternative for the following premise.",
                    f"{premise}\n{question_text[kind]}", [first, second]),
                "answer": LETTERS[label],
                "metric": "leading_choice_accuracy",
                "metadata": {"source": "xcopa", "language": language, "sourceIndex": index},
            })
        # Rank within the language so the global order interleaves all eleven
        # evenly; without this the outer hash sort would unbalance the panel.
        for rank, row in enumerate(order(rows)):
            row["_rank"] = rank
        per_language[language] = rows
    return [row for rows in per_language.values() for row in rows]


BUILDERS = {"mmlu": build_mmlu, "gsm8k": build_gsm8k, "hellaswag": build_hellaswag,
            "ifeval": build_ifeval, "xcopa": build_xcopa}


def stream_exclusions():
    """Every TRACE+17 prompt, plus the bare input of each SuperNI record."""
    manifest = json.loads((EXP / "data/trace-plus-17/manifest.json").read_text())
    prompts, inputs = set(), set()
    for task in manifest["tasks"]:
        for key in ("train", "validationGold", "testGold"):
            path = EXP / "data/trace-plus-17" / task["files"][key]["path"]
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                prompts.add(normalize(row["prompt"]))
                if task["source"] == "superni" and "\n\nInput:\n" in row["prompt"]:
                    body = row["prompt"].split("\n\nInput:\n", 1)[1].rsplit("\n\nOutput:", 1)[0]
                    inputs.add(normalize(body))
    return prompts, inputs


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)
    path.write_text(text)
    return {"path": str(path.relative_to(TARGET)), "records": len(rows),
            "sha256": sha256(text.encode()).hexdigest()}


def main():
    if TARGET.exists():
        raise ValueError(f"Immutable panel version already exists: {TARGET}")
    sources = json.loads((RAW / "sources.json").read_text())
    stream_prompts, superni_inputs = stream_exclusions()
    seen = set(stream_prompts)

    non_test, held_out, development, probe_ids, report = [], [], [], [], []
    for category, key, non_test_count in CATEGORIES:
        candidates, collided, duplicated = [], 0, 0
        for row in order(BUILDERS[key]()):
            fingerprint = normalize(row["prompt"])
            if not fingerprint:
                duplicated += 1
                continue
            if fingerprint in stream_prompts or fingerprint in superni_inputs:
                collided += 1  # Genuine contamination against the evaluated stream.
                continue
            if fingerprint in seen:
                duplicated += 1  # Repeat within this source, or across the new categories.
                continue
            seen.add(fingerprint)  # Also enforces isolation across the new categories.
            row.pop("_rank", None)
            row["taskId"] = f"general:{key}"
            row["category"] = category
            if "answer" in row:
                row["references"] = [row["answer"]]
            candidates.append(row)
        need = non_test_count + TEST_PER_CATEGORY + DEV_PER_CATEGORY
        if len(candidates) < need:
            raise ValueError(f"{category}: {len(candidates)} usable records, need {need}")
        # Disjoint slices in one deterministic order: non-test, then G-test, then dev.
        reserved = candidates[:non_test_count]
        test = candidates[non_test_count:non_test_count + TEST_PER_CATEGORY]
        dev = candidates[non_test_count + TEST_PER_CATEGORY:need]
        non_test.extend(reserved)
        held_out.extend(test)
        development.extend(dev)
        # The probe is a strict stratified subset of the G-test, never a new sample.
        probe_ids.extend(r["id"] for r in test[:PROBE_PER_CATEGORY])
        entry = {"category": category, "source": key, "metric": test[0]["metric"],
                 "usableRecords": len(candidates),
                 "excludedByStreamOverlap": collided, "excludedAsDuplicate": duplicated,
                 "nonTestCount": len(reserved), "testCount": len(test), "developmentCount": len(dev),
                 "probeCount": min(PROBE_PER_CATEGORY, len(test))}
        if key == "xcopa":
            languages = {}
            for row in test:
                languages[row["metadata"]["language"]] = languages.get(row["metadata"]["language"], 0) + 1
            entry["testLanguageCounts"] = dict(sorted(languages.items()))
        report.append(entry)

    everything = non_test + held_out + development
    identifiers = [r["id"] for r in everything]
    assert len(set(identifiers)) == len(identifiers), "duplicate record id"
    assert set(probe_ids) <= {r["id"] for r in held_out}, "probe must be a G-test subset"

    files = [
        write_jsonl(TARGET / "non-test/general.jsonl", non_test),
        write_jsonl(TARGET / "hidden-scorer/general-test.jsonl", held_out),
        write_jsonl(TARGET / "development/general-dev.jsonl", development),
    ]
    probe_path = TARGET / "hidden-scorer/probe-ids.json"
    probe_text = json.dumps(probe_ids, indent=2) + "\n"
    probe_path.write_text(probe_text)
    files.append({"path": str(probe_path.relative_to(TARGET)), "records": len(probe_ids),
                  "sha256": sha256(probe_text.encode()).hexdigest()})

    manifest = {
        "version": "hero-general-test-v1",
        "seed": SEED,
        "plan": "five equal-weight categories",
        "categories": report,
        "files": files,
        "rawSources": sources,
        "selection": ("SHA256(seed:id) order; disjoint slices per category in the order "
                      "non-test, G-test, development; probe is the first 40 G-test records per category"),
        "nonTestStratification": "13/13/13/13/12 (multilingual is the short category)",
        "crossSplitPromptOverlap": 0,
        "streamPromptOrSuperNiInputOverlap": 0,
        "totals": {"nonTest": len(non_test), "test": len(held_out),
                   "probe": len(probe_ids), "development": len(development)},
        "boundary": ("No G-test or probe record may enter proposals, residual ranking, KL "
                     "constraints, recovery or checkpoint selection."),
    }
    (TARGET / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (ROOT / "datasets/general-test-v1.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest["totals"]))
    for entry in report:
        print(f"  {entry['category']:28s} {entry['source']:10s} {entry['metric']:22s} "
              f"usable={entry['usableRecords']:6d} "
              f"streamOverlap={entry['excludedByStreamOverlap']:3d} duplicate={entry['excludedAsDuplicate']:4d}")


if __name__ == "__main__":
    main()
