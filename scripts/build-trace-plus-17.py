"""Build TRACE+17 from its selection stage.

Stage one (`pnpm dataset:select`) selects and deduplicates LLM-CL-Benchmark_500 and the
17 SuperNI tasks into experiments/data/trace-plus-17-500. This stage completes the
TRACE splits to 500/100/100: test prompts above the 16,384-token model input ceiling
are excluded, and shortfalls are filled from LLM-CL-Benchmark_5000 train.json. A
candidate must be a non-empty, answer-consistent prompt whose NFKC-whitespace form
appears in no split of any task. Candidates are ranked by
SHA256("42:<task>:<normalized prompt>") and assigned test first, then validation,
then train; test and validation additions must fit the input ceiling. Selected
records are kept byte-for-byte. NumGLUE-cm shortfalls are filled by seeded
sampling with replacement within each split, with distinct occurrence IDs.

Requires the pinned Qwen tokenizer (base environment):
  uv run --frozen --project packages/baselines/base python scripts/build-trace-plus-17.py
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import shutil
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / 'experiments/data/trace-plus-17-500'
OUTPUT = ROOT / 'experiments/data/trace-plus-17'
TRACKED = ROOT / 'datasets/trace-plus-17/manifest.json'
TRACKED_SELECTION = ROOT / 'datasets/trace-plus-17/selection-500/manifest.json'
# The completion pool sits beside LLM-CL-Benchmark_500 in the source root of the selection stage.
DEFAULT_POOL = (ROOT / os.environ['HERO_TRACE_SOURCE_ROOT'] / 'trace/LLM-CL-Benchmark_5000'
                if os.environ.get('HERO_TRACE_SOURCE_ROOT') else None)
SEED = 42
INPUT_LIMIT = 16384
SPLITS = (('test', 'testGold'), ('validation', 'validationGold'), ('train', 'train'))


def norm(value):
    return ' '.join(unicodedata.normalize('NFKC', value).split())


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [line for line in path.read_text(encoding='utf-8').split('\n') if line]


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(row + '\n' for row in rows), encoding='utf-8')


def dump(record):
    return json.dumps(record, ensure_ascii=False, separators=(',', ':'))


def resample_split(rows, count, task_id, split):
    """Keep existing rows, then sample only this split's records with replacement."""
    if len(rows) >= count:
        return list(rows)
    if not rows:
        raise ValueError(f'Cannot resample empty {task_id}/{split}')
    originals = [json.loads(row) for row in rows]
    rng = random.Random(f'{SEED}:{task_id}:{split}')
    result = list(rows)
    for occurrence in range(count - len(rows)):
        source = rng.choice(originals)
        result.append(dump({**source, 'id': f'{source["id"]}:repeat:{occurrence + 1}',
                            'metadata': {**source['metadata'], 'resampledFrom': source['id']}}))
    return result


def build(pool_root):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'experiments/checkpoints/qwen3.5-4b/base', local_files_only=True)

    def input_tokens(prompt):
        return len(tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True, return_dict=True,
                                                 add_generation_prompt=True, enable_thinking=False)['input_ids'])

    if OUTPUT.exists():
        raise SystemExit(f'{OUTPUT} already exists')
    assert input_tokens('hello') < 32, 'token counter must count tokens, not dictionary keys'
    manifest = json.loads((SELECTION / 'manifest.json').read_text())
    requested = manifest['requestedSplits']
    used = set()
    loaded = {}
    for task in manifest['tasks']:
        loaded[task['id']] = {name: lines(SELECTION / task['files'][name]['path']) for name in task['files']}
        for name in ('train', 'validationGold', 'testGold'):
            used |= {norm(json.loads(row)['prompt']) for row in loaded[task['id']][name]}

    staging = OUTPUT.with_name('trace-plus-17.building')
    shutil.rmtree(staging, ignore_errors=True)
    pool_files, totals, deviation_count = {}, {'train': 0, 'validation': 0, 'test': 0}, 0
    for task in manifest['tasks']:
        rows = {name: list(values) for name, values in loaded[task['id']].items()}
        supplement = {'added': {'train': 0, 'validation': 0, 'test': 0}, 'inputLimitExclusions': []}
        if task['source'] == 'trace':
            sid = task['sourceTaskId']
            if sid == 'MeetingBank':
                kept = []
                for row in rows['testGold']:
                    record = json.loads(row)
                    if input_tokens(record['prompt']) > INPUT_LIMIT:
                        supplement['inputLimitExclusions'].append(record['id'])
                    else:
                        kept.append(row)
                rows['testGold'] = kept
            source = pool_root / sid / 'train.json'
            pool_files[sid] = file_sha(source)
            answers, first = {}, {}
            for index, record in enumerate(json.loads(source.read_text())):
                prompt = norm(record['prompt'])
                if not prompt or not record['answer'].strip():
                    continue
                answers.setdefault(prompt, set()).add(record['answer'])
                first.setdefault(prompt, (index, record))
            candidates = sorted((p for p in first if p not in used and len(answers[p]) == 1),
                                key=lambda p: sha256(f'{SEED}:{task["id"]}:{p}'.encode()).hexdigest())
            cursor = 0
            for split, name in SPLITS:
                while len(rows[name]) < requested[split] and cursor < len(candidates):
                    prompt = candidates[cursor]
                    cursor += 1
                    index, record = first[prompt]
                    if split != 'train' and input_tokens(record['prompt']) > INPUT_LIMIT:
                        continue
                    item = {'id': f'trace:{sid}:supp5000:{index}', 'prompt': record['prompt'], 'answer': record['answer'],
                            'references': [record['answer']],
                            'metadata': {'source': 'trace', 'sourceTaskId': sid, 'sourceSplit': 'train', 'sourceIndex': index,
                                         'sourceBenchmark': 'LLM-CL-Benchmark_5000'}}
                    rows[name].append(dump(item))
                    if split == 'validation':
                        rows['validationInputs'].append(dump({k: item[k] for k in ('id', 'prompt', 'metadata')}))
                    used.add(prompt)
                    supplement['added'][split] += 1
            supplement['candidatePool'] = len(candidates)
            supplement['candidatesConsumed'] = cursor
        if task['id'] == 'trace:NumGLUE-cm':
            task['resampling'] = {'method': 'within-split sampling with replacement', 'seed': SEED, 'splits': {}}
            for split, name in SPLITS:
                original_count = len(rows[name])
                rows[name] = resample_split(rows[name], requested[split], task['id'], split)
                task['resampling']['splits'][split] = {
                    'uniquePrompts': len({norm(json.loads(row)['prompt']) for row in rows[name]}),
                    'addedRecords': len(rows[name]) - original_count}
            rows['validationInputs'] = [dump({k: record[k] for k in ('id', 'prompt', 'metadata')})
                                        for record in map(json.loads, rows['validationGold'])]
        for name, descriptor in task['files'].items():
            target = staging / descriptor['path']
            if rows[name] == loaded[task['id']][name]:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(SELECTION / descriptor['path'], target)
            else:
                write_lines(target, rows[name])
            descriptor['records'] = len(rows[name])
            descriptor['sha256'] = file_sha(target)
        realized = {split: len(rows[name]) for split, name in SPLITS}
        task['realized'] = realized
        task['deviations'] = [{'split': split, 'requested': requested[split], 'realized': realized[split],
                               'reason': 'source-exhausted'}
                              for split, _ in SPLITS if realized[split] < requested[split]]
        if task['source'] == 'trace':
            task['supplement'] = supplement
        deviation_count += len(task['deviations'])
        for split in totals:
            totals[split] += realized[split]
        assert len(rows['validationInputs']) == len(rows['validationGold'])

    # Unique occurrence IDs; repeated prompts may occur only within one task/split.
    seen_ids, prompt_owners = set(), {}
    for task in manifest['tasks']:
        for name in ('train', 'validationGold', 'testGold'):
            for row in lines(staging / task['files'][name]['path']):
                record = json.loads(row)
                prompt = norm(record['prompt'])
                owner = (task['id'], name)
                assert record['id'] not in seen_ids, record['id']
                assert prompt_owners.get(prompt, owner) == owner, record['id']
                seen_ids.add(record['id'])
                prompt_owners[prompt] = owner

    manifest.update(datasetVersion='1.1', totals={'requested': manifest['totals']['requested'], 'realized': totals,
                                                   'deviationCount': deviation_count})
    manifest['construction'] = {
        'selection': {'root': str(SELECTION.relative_to(ROOT)), 'manifestSha256': file_sha(SELECTION / 'manifest.json'),
                      'traceBenchmark': manifest['sourceLayout']['traceBenchmark']},
        'completion': {'benchmark': 'LLM-CL-Benchmark_5000', 'split': 'train', 'trainJsonSha256': pool_files,
                       'selection': f'SHA256("{SEED}:<task>:<normalized prompt>") order; test, then validation, then train; '
                                    'excluding prompts present in any split and prompts with conflicting answers'},
        'inputTokenLimit': {'tokens': INPUT_LIMIT, 'appliesTo': ['test', 'validation'],
                            'tokenizer': 'experiments/checkpoints/qwen3.5-4b/base chat template, thinking disabled'}}
    manifest['construction']['resampling'] = {
        'task': 'trace:NumGLUE-cm', 'seed': SEED,
        'rule': 'Keep selected records; sample remaining occurrences with replacement within each split; '
                'PRNG seeded by "42:<task>:<split>"; new IDs and metadata.resampledFrom identify repeats'}
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    staging.rename(OUTPUT)
    TRACKED_SELECTION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUTPUT / 'manifest.json', TRACKED)
    shutil.copyfile(SELECTION / 'manifest.json', TRACKED_SELECTION)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, default=DEFAULT_POOL, required=DEFAULT_POOL is None,
                        help='Directory with <task>/train.json from LLM-CL-Benchmark_5000 '
                             '(default: $HERO_TRACE_SOURCE_ROOT/trace/LLM-CL-Benchmark_5000)')
    args = parser.parse_args()
    manifest = build(args.pool.resolve())
    for task in manifest['tasks'][:8]:
        print(task['id'], task['realized'], task.get('supplement', {}).get('added'), task.get('supplement', {}).get('inputLimitExclusions'))
    print('totals', manifest['totals'])


if __name__ == '__main__':
    main()
