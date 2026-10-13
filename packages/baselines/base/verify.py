"""Verify sealed base results independently of the runner's completion flag."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import run as base


def verify_items(output, panel, contract_hash, require_complete, max_new_tokens):
    rows = base.load_completed(output, panel, contract_hash)
    expected_files = {f'{i:04d}.json' for i in range(len(panel))}
    actual_files = {p.name for p in (output / 'items').glob('*.json')}
    if actual_files - expected_files:
        raise ValueError('Unexpected prediction files outside the pinned panel')
    if require_complete and len(rows) != len(panel):
        raise ValueError(f'Incomplete prediction coverage: {len(rows)}/{len(panel)}')
    by_id = {r['id']: r for r in panel}
    for row in rows:
        if row['group'] != by_id[row['id']]['group']:
            raise ValueError(f'Prediction group mismatch: {row["id"]}')
        if row['outputTokens'] != len(row['outputTokenIds']) or not 0 <= row['outputTokens'] <= max_new_tokens:
            raise ValueError(f'Invalid output token count: {row["id"]}')
        if not math.isfinite(row['seconds']) or row['seconds'] < 0:
            raise ValueError(f'Invalid operation cost: {row["id"]}')
        stop = row['stopReason']
        if stop == 'input_limit':
            if row['promptTokens'] <= 16384 or row['prediction'] or row['outputTokens'] or row.get('itemStatus') != 'failed':
                raise ValueError(f'Invalid input-limit outcome: {row["id"]}')
        elif stop not in ('eos', 'length') or row['promptTokens'] > 16384 or row.get('itemStatus', 'complete') != 'complete':
            raise ValueError(f'Invalid successful outcome: {row["id"]}')
        if stop == 'length' and row['outputTokens'] != max_new_tokens:
            raise ValueError(f'Premature length stop: {row["id"]}')
    return rows


def verify(output, partial=False):
    from scoring import score_predictions
    from transformers import AutoTokenizer

    manifest = json.loads((output / 'manifest.json').read_text())
    checkpoint = Path(manifest['provenance'].get('checkpointPath', base.CHECKPOINT))
    general_dataset = manifest['provenance'].get('generalDataset')
    if manifest['profile'] == 'general':
        panel, provenance = base.general_panel(Path(general_dataset))
    elif manifest['profile'] == 'full':
        panel, provenance = base.full_panel(Path(manifest['provenance']['taskDataset']), Path(general_dataset) if general_dataset else None)
    else:
        raise ValueError('This verifier expects full or general profile')
    if base.digest(panel) != manifest['panelSha256'] or [r['id'] for r in panel] != manifest['recordIds']:
        raise ValueError('Pinned panel differs from source data')
    rows = verify_items(output, panel, base.digest(manifest), not partial, manifest['maxNewTokens'])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model_config = json.loads((checkpoint / 'config.json').read_text())
    model_eos = model_config['text_config']['eos_token_id']
    eos = set(([model_eos] if isinstance(model_eos, int) else model_eos) + [tokenizer.eos_token_id])
    by_id = {r['id']: r for r in panel}
    for row in rows:
        if row['stopReason'] == 'eos' and (not row['outputTokenIds'] or row['outputTokenIds'][-1] not in eos):
            raise ValueError(f'Invalid EOS token: {row["id"]}')
        if tokenizer.decode(row['outputTokenIds'], skip_special_tokens=True) != row['prediction']:
            raise ValueError(f'Token decode mismatch: {row["id"]}')
        tokens = tokenizer.apply_chat_template([{'role': 'user', 'content': by_id[row['id']]['prompt']}], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True)['input_ids']
        if len(tokens) != row['promptTokens']:
            raise ValueError(f'Input token count mismatch: {row["id"]}')
    report = {'verifiedAt': time.time(), 'scope': 'partial integrity audit' if partial else 'full sealed panel audit',
              'contractSha256': base.digest(manifest), 'verifiedItems': len(rows), 'expectedItems': len(panel),
              'itemFailures': sum(r.get('itemStatus') == 'failed' for r in rows),
              'lengthLimitedOutputs': sum(r['stopReason'] == 'length' for r in rows),
              'predictionChecks': 'source hashes, coverage, IDs, groups, prompt hashes, item checksums, tokenization, decoding, counts, stops, costs',
              'limitations': manifest['limitations'], 'maxNewTokens': manifest['maxNewTokens']}
    if partial:
        return report
    status = json.loads((output / 'status.json').read_text())
    expected_status = 'complete_with_item_failures' if report['itemFailures'] else 'complete'
    if status['status'] != expected_status or status['completed'] != len(panel) or status['total'] != len(panel) or status['itemFailures'] != report['itemFailures'] or status['contractSha256'] != base.digest(manifest):
        raise ValueError('Completion status disagrees with sealed results')
    for name in ('scores.json', 'metrics.json', 'cost-ledger.json'):
        if base.file_hash(output / name) != status['outputHashes'][name]:
            raise ValueError(f'Output hash mismatch: {name}')
    scores = json.loads((output / 'scores.json').read_text())
    if set(scores) != {r['group'] for r in panel}:
        raise ValueError('Task score coverage mismatch')
    for task in dict.fromkeys(r['group'] for r in panel):
        gold = [r for r in panel if r['group'] == task]
        predictions = [r for r in rows if r['group'] == task]
        recomputed = score_predictions(gold[0]['metric'], predictions, gold)
        if {k: v for k, v in scores[task].items() if k != 'scorerVersion'} != {k: v for k, v in recomputed.items() if k != 'scorerVersion'} or not math.isfinite(recomputed['score']) or not 0 <= recomputed['score'] <= 1:
            raise ValueError(f'Score recomputation mismatch: {task}')
    metrics = json.loads((output / 'metrics.json').read_text())
    expected_metrics = base.constant_metrics(scores, provenance['taskOrder']) if manifest['profile'] == 'full' else {}
    if 'generalCategories' in provenance:
        expected_metrics.update(base.general_metrics(scores, provenance))
    if metrics != expected_metrics:
        raise ValueError('Stage matrix/aggregate mismatch')
    ledger = json.loads((output / 'cost-ledger.json').read_text())
    if ledger['trainingSteps'] != 0 or ledger['harnessCalls'] != 0:
        raise ValueError('Base ledger reports training or harness activity')
    if sorted(ledger['operations'], key=lambda r: r['id']) != sorted(rows, key=lambda r: r['id']):
        raise ValueError('Cost ledger operation mismatch')
    if not math.isclose(ledger['inferenceSeconds'], sum(r['seconds'] for r in rows), rel_tol=1e-12) or ledger['outputTokens'] != sum(r['outputTokens'] for r in rows):
        raise ValueError('Cost totals mismatch')
    if base.tree_hash(checkpoint) != manifest['checkpointSha256']:
        raise ValueError('Checkpoint immutability check failed')
    for name, checksum in manifest.get('scorerAssets', {}).items():
        if base.file_hash(base.ROOT / name) != checksum:
            raise ValueError(f'Scorer asset changed: {name}')
    report.update(recordedScorerVersions=sorted({v['scorerVersion'] for v in scores.values()}), scoresVerified=True, outputHashesVerified=True, checkpointHashVerified=True,
                  metrics=metrics, scores=scores, inferenceSeconds=ledger['inferenceSeconds'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error('Audit output already exists; choose a new filename')
    report = verify(args.run, args.partial)
    if args.output:
        base.atomic_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
