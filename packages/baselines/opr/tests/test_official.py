"""The OPR method code: the reference copy, the marked edits of the running copy, and its TRACE+17 wiring."""
import difflib
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

import run

OPR = Path(__file__).resolve().parents[1]
REPOSITORY = OPR.parents[2]
DATASET = REPOSITORY / "experiments/data/trace-plus-17"
# Files that run unchanged, and the ones carrying [HERO] edits.
UNCHANGED = ("tools/eval.py",)
EDITED = ("core/OPRTrainer.py", "scripts/train.sh", "tools/generate_opr_ru.py")


def test_upstream_copy_is_verbatim():
    record = json.loads((OPR / "upstream/UPSTREAM.json").read_text())
    assert record["commit"] == run.UPSTREAM_COMMIT
    present = {p.relative_to(OPR / "upstream").as_posix() for p in (OPR / "upstream").rglob("*")
               if p.is_file() and "__pycache__" not in p.parts}
    assert present - {"UPSTREAM.json"} == set(record["files"])
    for relative, digest in record["files"].items():
        assert hashlib.sha256((OPR / "upstream" / relative).read_bytes()).hexdigest() == digest, relative


def test_only_the_listed_files_are_edited_and_every_edit_is_marked():
    assert set(run.OFFICIAL_FILES) == set(UNCHANGED) | set(EDITED)
    for relative in UNCHANGED:
        assert (OPR / "official" / relative).read_bytes() == (OPR / "upstream" / relative).read_bytes()
    for relative in EDITED:
        old = (OPR / "upstream" / relative).read_text().splitlines()
        new = (OPR / "official" / relative).read_text().splitlines()
        matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
        # A changed hunk with up to 24 lines above it: a marker comment may head a multi-line edit (train.sh lists its
        # flag edits above the one backslash-continued command, where no comment can go).
        hunks = [(new[j1:j2], new[max(0, j1 - 24):j2]) for tag, _, _, j1, j2 in matcher.get_opcodes() if tag != "equal"]
        assert hunks, relative
        for hunk, context in hunks:
            assert any("[HERO]" in line for line in context), (relative, hunk)


def test_official_dataset_lists_are_the_trace_plus_17_stream():
    manifest = json.loads((DATASET / "manifest.json").read_text())
    names = [run.official_name(task["id"]) for task in manifest["tasks"]]
    assert names[:8] == ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
    assert all(":" not in name for name in names)
    lists = run.official_dataset_lists(OPR / "official")
    assert lists == [names, names]


def trainer(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(OPR / "official/core"))
    from OPRTrainer import OnPolicyReplayTrainer

    return OnPolicyReplayTrainer("/base", str(tmp_path / "output"), "ru", run.RHO)


def test_trainer_budget_and_lora_continuation(tmp_path, monkeypatch):
    opr = trainer(tmp_path, monkeypatch)
    # The method budget: int(dataset_size * rho) records, int(b / (task_id + 1)) per historical task.
    assert opr.buffer_size == 5 and opr.task_num == 25 and opr.epoch_list == ["32"] * 25
    quotas = [int(opr.buffer_size / historical_tasks) for historical_tasks in range(1, opr.task_num)]
    assert all(quota >= 1 for quota in quotas[:5])
    assert quotas[5:] == [0] * 19  # integer division gives no replay with six or more past tasks
    assert [quotas[i - 1] for i in (1, 2, 6, 12, 24)] == [5, 2, 0, 0, 0]
    assert all(tasks * quota <= opr.buffer_size for tasks, quota in enumerate(quotas, start=1))
    assert 24 * quotas[-1] == 0  # do not pad or redistribute the method quotas
    commands = []
    monkeypatch.setattr("subprocess.run", lambda command, check: commands.append(command))
    monkeypatch.chdir(tmp_path)
    for name in ("FOMC", "MeetingBank"):
        (tmp_path / "data" / name).mkdir(parents=True)
    (tmp_path / "data/FOMC/buffer.jsonl").write_text('{"prompt": "p", "answer": "a", "score": 100}\n')
    (tmp_path / "data/MeetingBank/buffer.jsonl").write_text("")
    checkpoint = tmp_path / "output/v0-20261009-000000/checkpoint-32"
    checkpoint.mkdir(parents=True)
    opr.task_id = 0
    opr.train()
    assert commands[-1] == ["bash", "scripts/train.sh", "/base", "data/C-STANCE/train.jsonl", "32",
                            str(tmp_path / "output"), ""]
    assert opr.adapter_path == str(checkpoint) and opr.model_path == "/base"
    opr.task_id = 1
    opr.train()
    assert commands[-1] == ["bash", "scripts/train.sh", "/base", "data/FOMC/train.jsonl data/FOMC/buffer.jsonl", "32",
                            str(tmp_path / "output"), str(checkpoint)]
    # An empty buffer (quota 0 for every task) is left out of --dataset.
    opr.task_id = 2
    opr.train()
    assert commands[-1][3] == "data/MeetingBank/train.jsonl"


def test_superni_rewards_use_the_shared_evaluator(tmp_path, monkeypatch):
    # The rollout script's GPU-side imports are not installed here.
    for name in ("vllm", "evaluate"):
        module = types.ModuleType(name)
        module.LLM = module.SamplingParams = module.load = object()
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.syspath_prepend(str(OPR / "official/tools"))
    monkeypatch.delitem(sys.modules, "generate_opr_ru", raising=False)
    monkeypatch.delitem(sys.modules, "eval", raising=False)
    import generate_opr_ru

    gold = [{"id": "x0", "prompt": "p0", "answer": "yes", "references": ["yes"], "metadata": {}, "metric": "exact_match"},
            {"id": "x1", "prompt": "p1", "answer": "no", "references": ["no", "nope"], "metadata": {}, "metric": "exact_match"},
            {"id": "x2", "prompt": "p2", "answer": "maybe", "references": ["maybe"], "metadata": {}, "metric": "exact_match"}]
    (tmp_path / "data/superni-195").mkdir(parents=True)
    (tmp_path / "data/superni-195/gold.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gold))
    monkeypatch.chdir(tmp_path)
    # The second prompt is over-length: filtered out of the rollouts, so responses align with x0 and x2.
    assert generate_opr_ru.score_shared("superni-195", [True, False, True], ["Yes", "perhaps"]) == [1.0, 0.0]


def test_plan_writes_the_method_data_layout(tmp_path):
    plan, train_files, gold_files, eval_files = run.build_plan(
        DATASET, REPOSITORY / "experiments/data/general-test/v1", OPR / "official", profile="smoke", stages=2,
        train_limit=3, eval_limit=2, general_limit=1)
    assert list(train_files)[:2] == ["C-STANCE", "FOMC"] and len(train_files) == 25
    assert all(set(row) == {"prompt", "answer"} for rows in train_files.values() for row in rows)
    assert set(gold_files) == {name for name in train_files if name.startswith("superni-")}
    first = gold_files["superni-024"][0]
    assert first["metric"] == "rouge_l" and first["prompt"] == train_files["superni-024"][0]["prompt"]
    assert [s["officialName"] for s in plan["stages"]] == ["C-STANCE", "FOMC"]
    assert plan["opr"]["officialSha256"]["tools/eval.py"] == run.file_hash(OPR / "official/tools/eval.py")
    assert plan["opr"]["rho"] == run.RHO == 0.01


@pytest.mark.parametrize("position,phases", [(1, ["train", "evaluate", "rollout"]), (25, ["train", "evaluate", "evaluate"])])
def test_stage_units(position, phases):
    plan = {"stages": [{"stage": s, "rollout": s < 25, "evaluations": [{"group": "g", "ids": ["a"] * 200}]
                        if s != 25 else [{"group": "g", "ids": ["a"] * 200}, {"group": "h", "ids": ["a"] * 200}]}
                       for s in range(1, 26)]}
    assert [u["phase"] for u in run.stage_units(plan, position)] == phases
