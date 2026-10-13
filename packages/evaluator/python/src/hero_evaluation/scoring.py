"""Pinned TRACE and SuperNI scorers; all returned task scores are in [0, 1]."""

from collections import Counter
from functools import lru_cache
from pathlib import Path
import json
import re
import random
import threading
import string
import sys

from fuzzywuzzy import fuzz
from rouge import Rouge
from rouge_score import rouge_scorer
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

VERSION = "hero-task-scores-v3"
METRICS = {"exact_match", "strict_exact_match", "leading_choice_accuracy", "rouge_l", "edit_similarity", "sari",
           "final_number_match", "ifeval_strict"}
_rouge_trace = Rouge(metrics=["rouge-l"])
_rouge_ni = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_tokenize_13a = Tokenizer13a()


def normalize_answer(value):
    return " ".join("".join(c for c in value.lower() if c not in string.punctuation).split())


def postprocess_code(value):
    value = value.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    return re.sub(r"<(STR|NUM|CHAR)_LIT:(.*?)>", lambda m: m.group(2), value, flags=re.S)


@lru_cache(maxsize=1)
def xlingual_scorer():
    from tokenizers import Tokenizer

    path = Path(__file__).resolve().parents[5] / "experiments/data/tokenizers/gpt2/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(path))

    class GPTTokenizer:
        def tokenize(self, value):
            return [token.lstrip("Ġ") for token in tokenizer.encode(value, add_special_tokens=False).tokens]

    return rouge_scorer.RougeScorer(["rougeL"], tokenizer=GPTTokenizer())


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_number(value):
    """GSM8K convention: the answer is the last number emitted, commas stripped."""
    matches = _NUMBER.findall(value)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


_IFEVAL_RANDOM_LOCK = threading.Lock()


def ifeval_strict(prompt, prediction, metadata):
    # Some upstream metadata triggers random defaults (e.g. letter="#"). Keep
    # that verifier behavior reproducible without consuming the caller's RNG.
    with _IFEVAL_RANDOM_LOCK:
        state = random.getstate()
        random.seed(0)
        try:
            return _ifeval_strict(prompt, prediction, metadata)
        finally:
            random.setstate(state)


def _ifeval_strict(prompt, prediction, metadata):
    """Upstream prompt-level strict accuracy: every declared instruction must hold."""
    from .ifeval import INSTRUCTION_DICT

    identifiers = metadata.get("instructionIdList")
    arguments = metadata.get("instructionKwargs")
    if not identifiers or arguments is None or len(identifiers) != len(arguments):
        raise ValueError("ifeval_strict requires matching instructionIdList and instructionKwargs.")
    if not prediction.strip():
        return 0.0
    for identifier, kwargs in zip(identifiers, arguments):
        instruction = INSTRUCTION_DICT[identifier](identifier)
        instruction.build_description(**{k: v for k, v in kwargs.items() if v is not None})
        declared = instruction.get_instruction_args()
        if declared and "prompt" in declared:
            instruction.build_description(prompt=prompt)
        if not instruction.check_following(prediction):
            return 0.0
    return 1.0


# G-test labels are selected from the response, never from the reference answer.
GENERAL_CHOICE_SOURCES = {"mmlu": "ABCD", "hellaswag": "ABCD", "xcopa": "AB"}
_CHOICE_CUE = re.compile(
    r"\b(?:(?:final|correct|best)\s+)?answer\s*(?::|is\b\s*:?)\s*"
    r"|\b(?:correct|best|most\s+(?:plausible|logical))\s+"
    r"(?:choice|option|alternative|continuation)\s*(?::|is\b\s*:?)\s*", re.I)
_CHOICE_LABEL = re.compile(r"(?:option\s+|choice\s+)?[([]?\s*([A-Z])[)\]]?(?=$|[.:,。：、/–—-]|\s*\n|\s+(?:because|since)\b)", re.I)


def extract_general_choice(prediction, source):
    """Last explicit answer declaration; otherwise a single opening or closing label.

    Markdown wrappers and letter case do not matter. Invalid or ambiguous labels
    score zero; a list of options alone is not a selected answer.
    """
    if not prediction.strip():
        return None
    choices = GENERAL_CHOICE_SOURCES[source]
    text = re.sub(r"[*_`#>]", "", prediction).strip()
    cues = list(_CHOICE_CUE.finditer(text))
    if cues:
        candidate = text[cues[-1].end():].lstrip()
    else:
        labelled = re.findall(r"(?m)^\s*[([]?[A-Z][)\]]?(?:[.:。：]|\s*$)", text, re.I)
        if len(labelled) != 1:
            return None
        first, last = text.splitlines()[0].strip(), text.splitlines()[-1].strip()
        candidate = first if re.match(r"^[([]?[A-Z][)\]]?(?:[.:。：]|$)", first, re.I) else last
    match = _CHOICE_LABEL.match(candidate)
    if match is None or match[1].upper() not in choices:
        return None
    if re.match(r"\s*(?:or|and|/|,|&)\s*[([]?[A-Z]\b", candidate[match.end():], re.I):
        return None
    return match[1].upper()


def score_one(metric, prediction, reference, metadata):
    if not prediction.strip():
        return 0.0
    if metric == "strict_exact_match":
        return float(prediction == reference)
    if metric == "exact_match":
        return float(normalize_answer(prediction) == normalize_answer(reference))
    if metric == "leading_choice_accuracy":
        if metadata.get("source") in GENERAL_CHOICE_SOURCES:
            return float(extract_general_choice(prediction, metadata["source"]) == reference)
        # TRACE's official first-character rule remains unchanged.
        return float(prediction[0] == reference[0])
    if metric == "final_number_match":
        predicted, target = final_number(prediction), final_number(reference)
        return float(predicted is not None and target is not None and predicted == target)
    if metric == "edit_similarity":
        return fuzz.ratio(postprocess_code(prediction), postprocess_code(reference)) / 100
    if metric == "rouge_l":
        if metadata.get("source") == "trace":
            # TRACE calls its scorer with gold as hyp and prediction as ref.
            if not reference.replace(".", "").strip() or not prediction.replace(".", "").strip():
                return 0.0
            return _rouge_trace.get_scores(reference, prediction, avg=True)["rouge-l"]["f"]
        scorer = xlingual_scorer() if metadata.get("sourceTaskId") == "1219" or metadata.get("metricTrack") == "xlingual" else _rouge_ni
        return scorer.score(target=reference, prediction=prediction)["rougeL"].fmeasure
    raise ValueError(f"Unsupported pairwise metric {metric}")


def sari(source, prediction, references):
    """HF SARI semantics: lowercased 13a, four orders, 0/0=1, corrected keep recall."""
    if not prediction.strip():
        return 0.0
    source_tokens = _tokenize_13a(source.lower()).split(" ")
    candidate_tokens = _tokenize_13a(prediction.lower()).split(" ")
    reference_tokens = [_tokenize_13a(r.lower()).split(" ") for r in references]
    count = len(references)
    scores = []
    for n in range(1, 5):
        def grams(tokens):
            return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
        s = Counter({key: value * count for key, value in grams(source_tokens).items()})
        c = Counter({key: value * count for key, value in grams(candidate_tokens).items()})
        r = sum((grams(tokens) for tokens in reference_tokens), Counter())
        kept = s & c
        good = kept & r
        possible = s & r
        precision = sum(good[k] / kept[k] for k in good) / len(kept) if kept else 1
        recall = sum(good.values()) / sum(possible.values()) if possible else 1
        keep = 2 * precision * recall / (precision + recall) if precision + recall else 0
        deleted = s - c
        good_deleted = deleted - r
        deletion = sum(good_deleted[k] / deleted[k] for k in good_deleted) / len(deleted) if deleted else 1
        added = set(c) - set(s)
        good_added = added & set(r)
        possible_added = set(r) - set(s)
        precision = len(good_added) / len(added) if added else 1
        recall = len(good_added) / len(possible_added) if possible_added else 1
        addition = 2 * precision * recall / (precision + recall) if precision + recall else 0
        scores.append((keep + deletion + addition) / 3)
    return sum(scores) / 4


def score_predictions(metric, predictions, gold_items):
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric: {metric}")
    if not gold_items:
        raise ValueError("Gold items must be non-empty.")
    gold_ids = [item["id"] for item in gold_items]
    prediction_ids = [item["id"] for item in predictions]
    if len(set(gold_ids)) != len(gold_ids) or len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("Duplicate prediction or gold IDs.")
    if set(gold_ids) != set(prediction_ids):
        raise ValueError("Prediction coverage mismatch.")
    by_id = {p["id"]: p["prediction"] for p in predictions}
    scores = []
    for item in gold_items:
        prediction = by_id[item["id"]]
        if not isinstance(prediction, str):
            raise ValueError("Predictions must be strings.")
        if metric == "ifeval_strict":
            # Verifier-scored: the gold is the instruction set, not a reference string.
            scores.append({"id": item["id"], "score": ifeval_strict(item["prompt"], prediction, item.get("metadata", {}))})
            continue
        references = list(dict.fromkeys(item.get("references", []) + ([item["answer"]] if "answer" in item else [])))
        references = [r for r in references if isinstance(r, str) and r.strip()]
        if not references:
            raise ValueError(f"Gold item {item['id']} has no valid references.")
        if metric == "sari":
            match = re.search(r"Paragraph:\n(.*?)\n\nSimplification:", item["prompt"], re.S)
            if match is None:
                raise ValueError("SARI requires the original TRACE paragraph.")
            value = sari(match.group(1), prediction, references)
        else:
            value = max(score_one(metric, prediction, r, item.get("metadata", {})) for r in references)
        scores.append({"id": item["id"], "score": value})
    total = sum(item["score"] for item in scores)
    return {"metric": metric, "score": total / len(scores), "scoreSum": total, "sampleSize": len(scores), "itemScores": scores, "scorerVersion": VERSION}


def main():
    request = json.load(sys.stdin)
    result = score_predictions(request["metric"], request["predictions"], request["goldItems"])
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
