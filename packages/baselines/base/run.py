"""Bare, immutable Qwen3.5-4B inference. No harness, adapters or training."""
from __future__ import annotations

import argparse
import ast
import hashlib
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pnpm-workspace.yaml').is_file()), Path.cwd())
MODEL_ID = 'Qwen/Qwen3.5-4B'
TASK_DATASET = ROOT / 'experiments/data/trace-plus-17'
CHECKPOINT = ROOT / 'experiments/checkpoints/qwen3.5-4b/base'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(path):
    h = hashlib.sha256()
    for child in sorted(p for p in path.rglob('*') if p.is_file()):
        name = child.relative_to(path).as_posix().encode()
        h.update(len(name).to_bytes(8, 'big'))
        h.update(name)
        h.update(bytes.fromhex(file_hash(child)))
    return h.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def smoke_panel(dataset):
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if manifest['seed'] != 42 or [t['id'] for t in manifest['tasks'][:2]] != ['trace:C-STANCE', 'trace:FOMC']:
        raise ValueError('Expected canonical seed-42 C-STANCE → FOMC stream')
    panel, selected_train, sources = [], {}, {}
    for task in manifest['tasks'][:2]:
        for kind, count in [('train', 32), ('validationGold', 16), ('testGold', 16)]:
            spec = task['files'][kind]
            path = dataset / spec['path']
            actual = file_hash(path)
            if actual != spec['sha256']:
                raise ValueError(f'Dataset hash mismatch: {path}')
            sources[spec['path']] = actual
            rows = sorted(read_jsonl(path), key=lambda r: hashlib.sha256(('42:' + r['id']).encode()).hexdigest())[:count]
            if len(rows) != count:
                raise ValueError(f'Insufficient records: {task["id"]}/{kind}')
            if kind == 'train':
                selected_train[task['id']] = [r['id'] for r in rows]
                continue  # Base has no update and never receives training targets.
            panel.extend(dict(r, group=f'{task["id"]}/{kind}', metric=task['metric']) for r in rows)
    # Explicitly synthetic development fixtures, never a replacement for G-test.
    fixtures = {
        'general-knowledge': [('What is the capital of France? Answer with only the city.', 'Paris'), ('What planet is called the Red Planet? Answer with one word.', 'Mars'), ('What is H2O commonly called? Answer with one word.', 'Water'), ('How many days are in a week? Answer with a digit.', '7')],
        'mathematical-reasoning': [('Compute 12 + 7. Answer with only the number.', '19'), ('Compute 9 * 6. Answer with only the number.', '54'), ('Compute 81 / 9. Answer with only the number.', '9'), ('Compute 15 - 8. Answer with only the number.', '7')],
        'commonsense': [('Which keeps rain off your head: umbrella or fork? Answer with one word.', 'umbrella'), ('Which is used to cut paper: scissors or pillow? Answer with one word.', 'scissors'), ('Which melts in sunshine: ice or stone? Answer with one word.', 'ice'), ('Which can fly: sparrow or cow? Answer with one word.', 'sparrow')],
        'instruction-following': [('Output exactly: BLUE', 'BLUE'), ('Output exactly: alpha beta', 'alpha beta'), ('Output exactly: 123', '123'), ('Output exactly: ready', 'ready')],
        'multilingual': [('Translate cat into French. Output only the word.', 'chat'), ('Translate dog into Spanish. Output only the word.', 'perro'), ('将 hello 翻译成中文，只输出译文。', '你好'), ('Translate water into German. Output only the word.', 'Wasser')],
    }
    for category, pairs in fixtures.items():
        for i, (prompt, answer) in enumerate(pairs):
            panel.append(dict(id=f'synthetic:{category}:{i}', prompt=prompt, answer=answer, metric='strict_exact_match', group=f'synthetic-G/{category}'))
    panel.append(dict(id='synthetic:meetingbank:long', prompt='Summarize the following meeting transcript in one sentence.\n' + '\n'.join(f'Speaker {i % 3}: The council discussed the park renovation, public access, budget review, and agreed to reopen the park in June.' for i in range(400)) + '\nSummary:', answer='The council discussed park renovations and agreed to reopen the park in June.', metric='rouge_l', group='synthetic-long-context', metadata={'source': 'superni'}))
    ids = [r['id'] for r in panel]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate panel IDs')
    return panel, {'selectedTrainIds': selected_train, 'sourceFiles': sources, 'datasetManifestSha256': file_hash(dataset / 'manifest.json')}


GENERAL_CATEGORIES = ('general-knowledge', 'mathematical-reasoning', 'commonsense-reasoning',
                      'instruction-following', 'multilingual-understanding')


def general_panel(dataset):
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if manifest['version'] != 'hero-general-test-v1' or manifest['seed'] != 42:
        raise ValueError('Expected pinned seed-42 general-test/v1 panel')
    if [c['category'] for c in manifest['categories']] != list(GENERAL_CATEGORIES):
        raise ValueError('General category contract mismatch')
    specs = {f['path']: f for f in manifest['files']}
    names = ('hidden-scorer/general-test.jsonl', 'hidden-scorer/probe-ids.json')
    for name in names:
        if file_hash(dataset / name) != specs[name]['sha256']:
            raise ValueError(f'General panel hash mismatch: {name}')
    rows = read_jsonl(dataset / names[0])
    probes = json.loads((dataset / names[1]).read_text())
    if len(rows) != 1000 or len({r['id'] for r in rows}) != 1000 or len(probes) != 200 or len(set(probes)) != 200:
        raise ValueError('General test/probe coverage mismatch')
    if not set(probes) <= {r['id'] for r in rows}:
        raise ValueError('General probe is not a subset of G-test')
    for category in manifest['categories']:
        selected = [r for r in rows if r['category'] == category['category']]
        if len(selected) != 200 or sum(r['id'] in probes for r in selected) != 40 or any(r['metric'] != category['metric'] for r in selected):
            raise ValueError(f'Invalid category counts or metric: {category["category"]}')
    panel = [dict(r, group='general:' + r['category']) for r in rows]
    return panel, {'generalManifestSha256': file_hash(dataset / 'manifest.json'),
                   'generalSourceFiles': {name: specs[name]['sha256'] for name in names},
                   'generalCategories': list(GENERAL_CATEGORIES), 'generalProbeIds': probes,
                   'generalDataset': str(dataset.resolve()), 'generalTestStatus': 'included: 1000 test / 200 nested probe'}


def general_metrics(scores, provenance):
    probes = set(provenance['generalProbeIds'])
    categories, probe_categories = {}, {}
    for category in provenance['generalCategories']:
        result = scores['general:' + category]
        if result['sampleSize'] != 200 or len(result['itemScores']) != 200:
            raise ValueError('General metrics require complete category results')
        subset = [r['score'] for r in result['itemScores'] if r['id'] in probes]
        if len(subset) != 40:
            raise ValueError('Incomplete general probe results')
        categories[category] = result['score']
        probe_categories[category] = sum(subset) / len(subset)
    macro = sum(categories.values()) / 5
    probe_macro = sum(probe_categories.values()) / 5
    return {'generalChange': 0.0, 'generalMacro': macro, 'generalCategoryScores': categories,
            'generalProbeMacro': probe_macro, 'generalProbeCategoryScores': probe_categories,
            'generalCategoryChange': {c: 0.0 for c in categories}, 'generalWorstCategoryChange': 0.0,
            'generalByStage': [{'stage': stage, 'panel': 'full' if stage in (0, 4, 8, 16, 25) else 'probe',
                'sampleSize': 1000 if stage in (0, 4, 8, 16, 25) else 200,
                'macro': macro if stage in (0, 4, 8, 16, 25) else probe_macro,
                'changeFromSamePanelBase': 0.0, 'reusedConstantPrediction': True} for stage in range(26)],
            'generalNote': 'Unchanged immutable base; zero change follows reuse of the same measured predictions, not an extra evaluation.'}


def full_panel(dataset, general_dataset=None):
    manifest = json.loads((dataset / 'manifest.json').read_text())
    tasks = manifest['tasks']
    if manifest['seed'] != 42 or len(tasks) != 25 or [t['position'] for t in tasks] != list(range(1, 26)):
        raise ValueError('Expected pinned 25-task seed-42 stream')
    panel, sources, counts = [], {}, {}
    for task in tasks:
        spec = task['files']['testGold']
        path = dataset / spec['path']
        actual = file_hash(path)
        if actual != spec['sha256']:
            raise ValueError(f'Dataset hash mismatch: {path}')
        rows = read_jsonl(path)
        if not rows or len(rows) != spec['records']:
            raise ValueError(f'Dataset count mismatch: {path}')
        sources[spec['path']] = actual
        counts[task['id']] = len(rows)
        panel.extend(dict(r, group=task['id'], metric=task['metric']) for r in rows)
    if len({r['id'] for r in panel}) != len(panel):
        raise ValueError('Duplicate test IDs')
    provenance = {'sourceFiles': sources, 'datasetManifestSha256': file_hash(dataset / 'manifest.json'),
                   'taskOrder': [t['id'] for t in tasks], 'testCounts': counts,
                   'generalTestStatus': 'not included in task-only panel; available via --profile general or --general-dataset',
                   'countDeviations': {k: v for k, v in counts.items() if v != 100},
                   'taskDataset': str(dataset.resolve())}
    if general_dataset is not None:
        general, general_provenance = general_panel(general_dataset)
        panel.extend(general)
        provenance.update(general_provenance)
    return panel, provenance


def constant_metrics(scores, task_order):
    values = [scores[task]['score'] for task in task_order]
    rows = []
    for stage in range(1, len(task_order) + 1):
        seen = range(stage) if stage in (4, 8, 16, 25) else range(8) if stage in (12, 20) else [stage - 1]
        indices = sorted(set(seen) | {stage - 1})
        rows.extend({'stage': stage, 'task': task_order[j], 'score': values[j], 'reusedConstantPrediction': True} for j in indices)
    return {'AA': sum(values) / len(values), 'forgetting': 0.0, 'signedBWT': 0.0,
            'acquisitionMeanDiagonal': sum(values) / len(values), 'incrementalGain': 0.0,
            'sparseMatrix': rows, 'generalChange': None,
            'note': 'Constant predictor, identical cached item results at all stages; no independent stage replications.'}


class InputLimitExceeded(ValueError):
    def __init__(self, count):
        super().__init__(f'Input exceeds 16384 tokens: {count}; truncation prohibited')
        self.count = count


class Predictor:
    def __init__(self, checkpoint, device, max_new_tokens):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        self.torch, self.device, self.max_new_tokens = torch, device, max_new_tokens
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; no implicit device fallback')
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.bfloat16, local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)
        eos = self.model.generation_config.eos_token_id
        self.eos = sorted(set(([eos] if isinstance(eos, int) else eos or []) + [self.tokenizer.eos_token_id]))

    def predict(self, prompt):
        # This API accepts only an ordinary prompt: no labels, memory, tools or context.
        torch = self.torch
        started = time.perf_counter()
        inputs = self.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True, return_tensors='pt').to(self.device)
        count = inputs['input_ids'].shape[-1]
        if count > 16384:
            raise InputLimitExceeded(count)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, eos_token_id=self.eos, pad_token_id=self.tokenizer.pad_token_id)
        tokens = output[0, count:].tolist()
        seconds = time.perf_counter() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {'prediction': self.tokenizer.decode(tokens, skip_special_tokens=True), 'outputTokenIds': tokens, 'promptTokens': count, 'outputTokens': len(tokens), 'seconds': seconds, 'tokensPerSecond': len(tokens) / seconds, 'stopReason': 'eos' if tokens and tokens[-1] in self.eos else 'length', 'processPeakRssBytes': rss if sys.platform == 'darwin' else rss * 1024, 'deviceAllocatedBytes': torch.mps.current_allocated_memory() if self.device == 'mps' else torch.cuda.max_memory_allocated() if self.device == 'cuda' else None}


class VllmPredictor:
    """Predictor's inputs, stops and limits, decoded in batches by vLLM (vllm_decode.GreedyEngine)."""
    def __init__(self, checkpoint, max_new_tokens):
        from transformers import AutoTokenizer
        from vllm_decode import GreedyEngine
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        # The model EOS of Predictor (generation_config, derived from config.json) and the tokenizer EOS.
        eos = json.loads((Path(checkpoint) / 'config.json').read_text())['text_config']['eos_token_id']
        self.eos = sorted(set(([eos] if isinstance(eos, int) else eos or []) + [self.tokenizer.eos_token_id]))
        self.engine = GreedyEngine(Path(checkpoint), max_input_tokens=16384, max_new_tokens=max_new_tokens)

    def predict_many(self, prompts):
        """One result per prompt; an over-limit input is an InputLimitExceeded instance."""
        from vllm_decode import prompt_ids
        results = self.engine.decode(self.tokenizer, [prompt_ids(self.tokenizer, prompt) for prompt in prompts], self.eos)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rows = []
        for result in results:
            if result['stopReason'] == 'input_limit':
                rows.append(InputLimitExceeded(result['promptTokens']))
                continue
            seconds = result['seconds']
            rows.append({key: result[key] for key in ('prediction', 'outputTokenIds', 'promptTokens', 'outputTokens', 'seconds', 'stopReason')}
                        | {'tokensPerSecond': result['outputTokens'] / seconds if seconds else 0.0, 'processPeakRssBytes': rss if sys.platform == 'darwin' else rss * 1024,
                           'deviceAllocatedBytes': None, 'engine': result['engine'], 'batchSize': result['batchSize']})
        return rows

    def predict(self, prompt):
        (result,) = self.predict_many([prompt])
        if isinstance(result, InputLimitExceeded):
            raise result
        return result


def load_completed(output, panel, contract_hash):
    completed = []
    for index, item in enumerate(panel):
        path = output / 'items' / f'{index:04d}.json'
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        checksum = record.pop('sha256')
        if digest(record) != checksum or record['contractSha256'] != contract_hash or record['id'] != item['id'] or record['promptSha256'] != digest(item['prompt']):
            raise ValueError(f'Invalid cached prediction: {path}')
        record['sha256'] = checksum
        completed.append(record)
    return completed


class ResumeMismatch(ValueError):
    pass


def inference_code_hash(source, engine='transformers'):
    """Fingerprint the direct model boundary of the run's engine; wrapper/monitor/scorer edits are separate.
    The vLLM engine's own module is pinned separately by the contract's decoderSha256."""
    names = {'Predictor', 'InputLimitExceeded'} if engine == 'transformers' else {'VllmPredictor', 'InputLimitExceeded'}
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ResumeMismatch('Cannot verify inference implementation in source snapshot')
    return digest([ast.dump(node, include_attributes=False) for node in nodes])


def compatible_resume(stored, current, old_source=None, new_source=None):
    # Metadata changes are recorded, not used to invalidate expensive predictions.
    metadata = {'runnerSha256', 'scorerSha256', 'scorerVersion', 'host', 'limitations', 'provenance', 'scorerAssets', 'scorerLibraries'}
    changed = sorted(key for key in stored.keys() | current.keys()
                     if key not in stored or key not in current or stored[key] != current[key])
    incompatible = [key for key in changed if key not in metadata]
    if incompatible:
        raise ResumeMismatch('Prediction contract changed: ' + ', '.join(incompatible))
    if 'runnerSha256' in changed:
        for source, contract in ((old_source, stored), (new_source, current)):
            if source is None or hashlib.sha256(source.encode()).hexdigest() != contract['runnerSha256']:
                raise ResumeMismatch('Runner changed: hash-verified source snapshot required for compatibility audit')
        engine = 'transformers' if current.get('engine', 'transformers') == 'transformers' else 'vllm'
        if inference_code_hash(old_source, engine) != inference_code_hash(new_source, engine):
            raise ResumeMismatch('Inference implementation changed; audit affected predictions before reuse')
    return changed


def prepare_contract(output, current, source):
    path = output / 'manifest.json'
    snapshot = output / 'runner-source.py'
    if path.exists():
        stored = json.loads(path.read_text())
        changed = compatible_resume(stored, current, snapshot.read_text() if snapshot.exists() else None, source)
    else:
        stored, changed = current, []
        atomic_json(path, stored)
        snapshot.write_text(source)
    # Keep the original manifest and item contract hashes immutable. Each attempt
    # separately identifies its actual host, wrapper and scorer implementation.
    execution = output / 'executions' / str(time.time_ns())
    execution.mkdir(parents=True)
    atomic_json(execution / 'contract.json', current)
    (execution / 'runner-source.py').write_text(source)
    atomic_json(execution / 'resume.json', {'originalContractSha256': digest(stored),
                'executionSha256': digest(current), 'acceptedMetadataChanges': changed})
    if any(key in changed for key in ('scorerSha256', 'scorerVersion', 'scorerAssets', 'scorerLibraries')):
        for name in ('scores.json', 'metrics.json', 'cost-ledger.json', 'status.json'):
            previous = output / name
            if previous.exists():
                (execution / ('previous-' + name)).write_bytes(previous.read_bytes())
    return digest(stored)


def build_contract(args):
    """Validate weights and data, and derive the immutable prediction contract."""
    from scoring import VERSION
    checkpoint = args.checkpoint.resolve()
    registry = json.loads((args.registry or checkpoint.parent / 'registry.json').read_text())
    if registry['model'] != MODEL_ID or registry['modelRevision'] != REVISION:
        raise ValueError('Unexpected base model/revision')
    print('Verifying immutable base weights and dataset...', flush=True)
    weights_hash = tree_hash(checkpoint)
    if weights_hash != registry['base']['sha256']:
        raise ValueError('Base checkpoint differs from pinned registry')
    if args.profile == 'general':
        panel, provenance = general_panel(args.general_dataset or ROOT / 'experiments/data/general-test/v1')
    elif args.profile == 'full':
        panel, provenance = full_panel(args.task_dataset, args.general_dataset)
    else:
        panel, provenance = smoke_panel(args.task_dataset)
    provenance['checkpointPath'] = str(checkpoint)
    has_general = 'generalCategories' in provenance
    limitations = [] if args.profile != 'smoke' else ['Engineering smoke only; synthetic G fixtures; not reportable scientific scores']
    if args.profile == 'full':
        if provenance['countDeviations']:
            limitations += ['Available task counts differ from Protocol M']
        if not has_general:
            limitations += ['G-test results must be joined from a separate general run']

    engine = getattr(args, 'engine', 'transformers')
    contract = dict(profile=args.profile, method='base', model=MODEL_ID, revision=REVISION, checkpointSha256=weights_hash, runnerSha256=file_hash(Path(__file__)), scorerSha256=file_hash(Path(__file__).with_name('scoring.py')), scorerVersion=VERSION, eosTokenIds='union(model.generation_config.eos_token_id, tokenizer.eos_token_id)', chatTemplateSha256=file_hash(checkpoint / 'tokenizer_config.json'), device=args.device, precision='bfloat16', reasoning=False, doSample=False, maxNewTokens=args.max_new_tokens, maxInputTokens=16384, seed=42, harness=None, adapter=None, training=False, panelSha256=digest(panel), recordIds=[r['id'] for r in panel], provenance=provenance, libraries={p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'tokenizers', *(('vllm',) if engine == 'vllm' else ()))}, host=platform.platform(), limitations=limitations)
    if engine == 'vllm':
        from vllm_decode import ENGINE
        contract.update(engine=ENGINE, decoderSha256=file_hash(Path(__file__).with_name('vllm_decode.py')))
    from scorer_assets import gpt2_tokenizer_path, punkt_tokenizer_path
    gpt2 = gpt2_tokenizer_path()
    contract['scorerAssets'] = {str(gpt2): file_hash(gpt2)}
    if has_general:
        evaluator = Path(__file__).resolve().parent
        contract['scorerAssets'].update({str(p): file_hash(p) for p in sorted((evaluator / 'ifeval').glob('*.py'))})
        punkt = punkt_tokenizer_path()
        contract['scorerAssets'][str(punkt)] = file_hash(punkt)
        contract['scorerLibraries'] = {name: importlib.metadata.version(name) for name in ('langdetect', 'immutabledict', 'nltk')}
    return contract, panel, provenance, checkpoint, weights_hash


def run(args):
    from scoring import score_predictions
    contract, panel, provenance, checkpoint, weights_hash = build_contract(args)
    has_general = 'generalCategories' in provenance
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'items').mkdir(exist_ok=True)
    contract_hash = prepare_contract(output, contract, Path(__file__).read_text())
    completed = load_completed(output, panel, contract_hash)
    done = {r['id'] for r in completed}
    atomic_json(output / 'status.json', {'status': 'running', 'updatedAt': time.time(), 'pid': os.getpid(), 'completed': len(done), 'total': len(panel), 'contractSha256': contract_hash})
    vllm = contract.get('engine', 'transformers') != 'transformers'
    print(f'{len(done)}/{len(panel)} cached; loading bare BF16 model on {args.device}' + (' with ' + contract['engine'] if vllm else ''), flush=True)
    predictor = None
    if len(done) < len(panel) or args.verify_resume:
        started = time.perf_counter()
        predictor = VllmPredictor(checkpoint, args.max_new_tokens) if vllm else Predictor(checkpoint, args.device, args.max_new_tokens)
        atomic_json(output / f'load-{time.time_ns()}.json', {'seconds': time.perf_counter() - started, 'pid': os.getpid(), 'device': args.device})

    def seal(index, item, result):
        if isinstance(result, InputLimitExceeded):
            if args.profile == 'smoke':
                raise result
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result = dict(prediction='', outputTokenIds=[], promptTokens=result.count, outputTokens=0,
                          seconds=0.0 if vllm else time.perf_counter() - started, tokensPerSecond=0.0,
                          stopReason='input_limit', itemStatus='failed', error=str(result),
                          processPeakRssBytes=rss if sys.platform == 'darwin' else rss * 1024,
                          deviceAllocatedBytes=None)
        else:
            result = dict(result, itemStatus='complete')
        record = dict(result, id=item['id'], group=item['group'], promptSha256=digest(item['prompt']), contractSha256=contract_hash, executionSha256=digest(contract))
        record['sha256'] = digest(record)
        atomic_json(output / 'items' / f'{index:04d}.json', record)
        completed.append(record)
        print(f'{len(completed)}/{len(panel)} {item["id"]}: {result["outputTokens"]} tokens, {result["seconds"]:.2f}s, {result["stopReason"]}', flush=True)

    pending = [(index, item) for index, item in enumerate(panel) if item['id'] not in done]
    if args.stop_after:
        pending = pending[:args.stop_after]
    if vllm:
        from vllm_decode import CALL_ITEMS
        for start in range(0, len(pending), CALL_ITEMS):
            part = pending[start:start + CALL_ITEMS]
            atomic_json(output / 'current-item.json', {'id': part[0][1]['id'], 'index': part[0][0], 'group': part[0][1]['group'], 'startedAt': time.time(), 'batchItems': len(part)})
            for (index, item), result in zip(part, predictor.predict_many([item['prompt'] for _, item in part])):
                seal(index, item, result)
            atomic_json(output / 'status.json', {'status': 'running', 'updatedAt': time.time(), 'pid': os.getpid(), 'completed': len(completed), 'total': len(panel), 'contractSha256': contract_hash})
    else:
        for index, item in pending:
            atomic_json(output / 'current-item.json', {'id': item['id'], 'index': index, 'group': item['group'], 'startedAt': time.time()})
            started = time.perf_counter()
            try:
                result = predictor.predict(item['prompt'])
            except InputLimitExceeded as exc:
                result = exc
            seal(index, item, result)
            atomic_json(output / 'status.json', {'status': 'running', 'updatedAt': time.time(), 'pid': os.getpid(), 'completed': len(completed), 'total': len(panel), 'contractSha256': contract_hash})
    if args.stop_after and len(completed) < len(panel):
        atomic_json(output / 'status.json', {'status': 'paused', 'completed': len(completed), 'total': len(panel), 'contractSha256': contract_hash})
        return
    if args.verify_resume:
        # Fresh model process replays one boundary item; exact token equality, no tolerance.
        boundary = completed[0]
        item = next(r for r in panel if r['id'] == boundary['id'])
        replay = predictor.predict(item['prompt'])
        matched = replay['outputTokenIds'] == boundary['outputTokenIds']
        atomic_json(output / 'resume-audit.json', {'id': item['id'], 'exactTokensMatch': matched, 'replayCost': replay, 'scope': 'One fresh-process boundary prediction; all cached item checksums validated'})
        if not matched:
            raise ValueError('Resume replay token mismatch')
    scores = {}
    for group in dict.fromkeys(r['group'] for r in panel):
        gold = [r for r in panel if r['group'] == group]
        predictions = [r for r in completed if r['group'] == group]
        scores[group] = score_predictions(gold[0]['metric'], predictions, gold)
    atomic_json(output / 'scores.json', scores)
    metrics = constant_metrics(scores, provenance['taskOrder']) if args.profile == 'full' else {}
    if has_general:
        metrics.update(general_metrics(scores, provenance))
    if metrics:
        atomic_json(output / 'metrics.json', metrics)
    atomic_json(output / 'cost-ledger.json', {'operations': completed, 'inferenceSeconds': sum(r['seconds'] for r in completed), 'outputTokens': sum(r['outputTokens'] for r in completed), 'processPeakRssBytes': max(r['processPeakRssBytes'] for r in completed), 'memoryNote': 'RSS is process lifetime high-water; MPS allocated bytes are post-operation samples, not device peaks', 'trainingSteps': 0, 'harnessCalls': 0})
    if tree_hash(checkpoint) != weights_hash:
        raise ValueError('Base weights changed during inference')
    atomic_json(output / 'status.json', {'status': 'complete_with_item_failures' if any(r.get('itemStatus') == 'failed' for r in completed) else 'complete', 'itemFailures': sum(r.get('itemStatus') == 'failed' for r in completed), 'completed': len(completed), 'total': len(panel), 'contractSha256': contract_hash, 'outputHashes': {name: file_hash(output / name) for name in (('scores.json', 'cost-ledger.json', 'metrics.json') if metrics else ('scores.json', 'cost-ledger.json'))}})
    print(f'COMPLETE: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['smoke', 'full', 'general'], default='smoke')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT, help='Pinned local Qwen3.5-4B weights')
    parser.add_argument('--registry', type=Path, help='Checkpoint registry; defaults to the parent of --checkpoint')
    parser.add_argument('--general-dataset', type=Path, help='Add the pinned G-test panel to full, or select it for general')
    parser.add_argument('--engine', choices=['vllm', 'transformers'], default='vllm',
                        help='vllm: batched greedy decoding (formal protocol, CUDA); transformers: unbatched generate')
    parser.add_argument('--device', choices=['mps', 'cuda', 'cpu'], default='cuda')
    parser.add_argument('--stop-after', type=int, help='Pause after N new predictions for restart testing')
    parser.add_argument('--verify-resume', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=5120)
    parser.add_argument('--task-dataset', type=Path, default=TASK_DATASET, help='TRACE+17 materialization')
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        parser.error('--stop-after must be positive')
    if args.max_new_tokens < 1:
        parser.error('--max-new-tokens must be positive')
    if args.engine == 'vllm' and args.device != 'cuda':
        parser.error('--engine vllm decodes on CUDA')
    if args.profile == 'smoke' and args.general_dataset:
        parser.error('--general-dataset requires full or general')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args)
        except ResumeMismatch:
            raise  # A rejected invocation must not overwrite the existing run status.
        except Exception as exc:
            previous = json.loads((args.output / 'status.json').read_text()) if (args.output / 'status.json').exists() else {}
            failure = {**previous, 'status': 'failed', 'updatedAt': time.time(), 'error': str(exc), 'type': type(exc).__name__}
            atomic_json(args.output / 'status.json', failure)
            atomic_json(args.output / 'failure.json', failure)
            print(json.dumps(failure), file=sys.stderr, flush=True)
            raise


if __name__ == '__main__':
    main()
