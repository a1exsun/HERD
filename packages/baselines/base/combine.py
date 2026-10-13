"""Seal the formal base result from an independently verified task run and G run."""
import argparse
import json
from pathlib import Path

import run as base
from verify import verify


PREDICTOR_FIELDS = ('model', 'revision', 'checkpointSha256', 'chatTemplateSha256', 'device',
                    'precision', 'reasoning', 'doSample', 'maxNewTokens', 'maxInputTokens',
                    'seed', 'harness', 'adapter', 'training', 'eosTokenIds', 'libraries', 'engine', 'decoderSha256')


def repetition_period(tokens, longest=2000):
    """Period of an exact repetition closing the output (three copies), or 0."""
    for period in range(1, min(longest, len(tokens) // 3) + 1):
        if tokens[-period:] == tokens[-2 * period:-period] == tokens[-3 * period:-2 * period]:
            return period
    return 0


def length_stops(run):
    stops = []
    for path in sorted((run / 'items').glob('*.json')):
        row = json.loads(path.read_text())
        if row['stopReason'] == 'length':
            stops.append({'id': row['id'], 'exactRepetitionPeriod': repetition_period(row['outputTokenIds'])})
    return stops


def combine(task_run, general_run):
    tasks = json.loads((task_run / 'manifest.json').read_text())
    general = json.loads((general_run / 'manifest.json').read_text())
    if tasks['profile'] != 'full' or general['profile'] != 'general':
        raise ValueError('Expected task full run and separate general run')
    # Runs from before the vLLM engine carry no engine fields: they are transformers runs.
    changed = [key for key in PREDICTOR_FIELDS if tasks.get(key) != general.get(key)]
    if changed:
        raise ValueError('Cannot combine different predictors: ' + ', '.join(changed))
    sources = []
    for directory, manifest in ((task_run, tasks), (general_run, general)):
        source = directory / 'runner-source.py'
        if base.file_hash(source) != manifest['runnerSha256']:
            raise ValueError('Original inference source snapshot failed its hash check')
        sources.append(base.inference_code_hash(source.read_text()))
    if sources[0] != sources[1]:
        raise ValueError('Cannot combine different inference implementations')
    task_audit, general_audit = verify(task_run), verify(general_run)
    return {'status': 'verified', 'scope': '25-task TRACE+17 base plus five-category G-test',
            'taskRun': str(task_run.resolve()), 'generalRun': str(general_run.resolve()),
            'taskContractSha256': base.digest(tasks), 'generalContractSha256': base.digest(general),
            'decoding': {key: tasks[key] for key in ('model', 'revision', 'precision', 'reasoning', 'doSample', 'maxNewTokens', 'maxInputTokens', 'device')},
            'predictions': task_audit['verifiedItems'] + general_audit['verifiedItems'],
            'itemFailures': task_audit['itemFailures'] + general_audit['itemFailures'],
            'metrics': {**task_audit['metrics'], **general_audit['metrics']},
            'taskScores': task_audit['scores'], 'generalScores': general_audit['scores'],
            'lengthStops': {'tasks': length_stops(task_run), 'general': length_stops(general_run)},
            'inferenceSeconds': {'tasks': task_audit['inferenceSeconds'], 'general': general_audit['inferenceSeconds']},
            'sourceStatusHashes': {'tasks': base.file_hash(task_run / 'status.json'), 'general': base.file_hash(general_run / 'status.json')},
            'limitations': sorted(set(tasks.get('limitations', []) + general.get('limitations', []))
                                  - {'G-test results must be joined from a separate general run'})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--general', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; use a new artifact path')
    report = combine(args.tasks, args.general)
    base.atomic_json(args.output, report)
    print(json.dumps({'output': str(args.output), 'AA': report['metrics']['AA'], 'generalMacro': report['metrics']['generalMacro']}))


if __name__ == '__main__':
    main()
