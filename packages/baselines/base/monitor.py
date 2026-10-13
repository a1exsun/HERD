"""Read-only run validation and task-aware ETA; never changes inference artifacts."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
from zoneinfo import ZoneInfo


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def estimate(tasks, lengths, records, now, started, max_new_tokens):
    done = {r['id'] for r in records}
    observed = defaultdict(list)
    for record in records:
        if record.get('itemStatus') != 'failed' and record['outputTokens']:
            observed[record['group']].append(record)
    long_outputs = [r for rs in observed.values() for r in rs if r['outputTokens'] >= 16]
    all_outputs = [r for rs in observed.values() for r in rs]
    # Hold output length fixed to estimate input-related cost without multiplying
    # the entire decode duration by the input-length ratio.
    capped = [r for r in all_outputs if r['outputTokens'] == max_new_tokens]
    input_rate = 0.0
    if len(capped) >= 5 and len({r['promptTokens'] for r in capped}) > 1:
        fit = statistics.linear_regression([r['promptTokens'] for r in capped], [r['seconds'] for r in capped])
        input_rate = max(0.0, fit.slope)
    rate = statistics.median(max(0.0, r['seconds'] - input_rate * r['promptTokens']) / r['outputTokens'] for r in (long_outputs or all_outputs)) if all_outputs else None
    estimates = []
    for task in tasks:
        remaining = [r for r in lengths if r['group'] == task['id'] and r['id'] not in done]
        valid = [r for r in remaining if r['tokens'] <= 16384]
        samples = observed[task['id']]
        if len(samples) >= 5:
            average = statistics.mean(r['seconds'] for r in samples)
            input_delta = statistics.mean(r['tokens'] for r in valid) - statistics.mean(r['promptTokens'] for r in samples) if valid else 0.0
            central = max(0.0, average + input_rate * input_delta) * len(valid)
            low, high = central * .5, central * 2
            basis = 'same-task mean plus additive input-cost correction'
        elif rate is not None:
            # Declared output scenarios; not observations of unseen tasks.
            token_low, token_mid, token_high = (2, 8, 32) if task['outputType'] == 'closed' else (32, 128, 512)
            low, central, high = [sum(input_rate * r['tokens'] + rate * tokens for r in valid) for tokens in (token_low, token_mid, token_high)]
            basis = 'measured additive input/output costs with assumed output-length scenarios'
        else:
            low = central = high = None
            basis = 'no measurements yet'
        estimates.append({'task': task['id'], 'remaining': len(remaining), 'successfulSamples': len(samples), 'remainingSecondsLow': low, 'remainingSecondsCentral': central, 'remainingSecondsHigh': high, 'basis': basis})
    result = {'tasks': estimates, 'measuredDecodeSecondsPerToken': rate, 'inputSecondsPerToken': input_rate, 'inputCalibrationSamples': len(capped), 'estimatorVersion': 'additive', 'elapsedSeconds': now - started,
              'assumptions': 'Scenario range, not statistical confidence bounds. Unseen closed outputs assume 2/8/32 tokens; open outputs 32/128/512. Input cost is calibrated at fixed 512-token outputs (at least 5 samples with varying input lengths); zero correction until available. Decode duration is never multiplied by an input-length ratio. Same-task means plus additive input-cost corrections replace assumptions after 5 successful records. Current in-flight item is counted in full. Estimates cover only this run directory; separately launched G/task runs are excluded.'}
    if rate is not None:
        for label in ('Low', 'Central', 'High'):
            remaining = sum(r['remainingSeconds' + label] for r in estimates)
            result['remainingSeconds' + label] = remaining
            result['totalSeconds' + label] = now - started + remaining
            result['finish' + label] = datetime.fromtimestamp(now + remaining, ZoneInfo('Australia/Melbourne')).isoformat(timespec='seconds')
    return result


def inspect(run, root):
    now = time.time()
    launch = json.loads((run / 'launch.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    preflight = json.loads((run / 'preflight.json').read_text())
    expected = {r['id']: r for r in preflight['inputLengths']}
    contract = digest(manifest)
    records = []
    for path in sorted((run / 'items').glob('*.json')):
        record = json.loads(path.read_text())
        checksum = record.pop('sha256')
        if digest(record) != checksum or record['contractSha256'] != contract:
            raise ValueError(f'Invalid sealed item: {path}')
        if record['id'] not in expected or record['group'] != expected[record['id']]['group']:
            raise ValueError(f'Unexpected item identity: {path}')
        records.append(record)
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate cached IDs')
    process = subprocess.run(['ps', '-p', str(launch['pid']), '-o', 'pid=,lstart=,etime=,pcpu=,rss=,command='], capture_output=True, text=True)
    process_line = process.stdout.strip()
    live = process.returncode == 0 and str(run.resolve()) in process_line and 'baselines/base/run.py' in process_line
    tasks = [] if manifest['profile'] == 'general' else json.loads((Path(manifest['provenance']['taskDataset']) / 'manifest.json').read_text())['tasks']
    tasks += [{'id': 'general:' + c, 'outputType': 'open' if c in ('mathematical-reasoning', 'instruction-following') else 'closed'} for c in manifest['provenance'].get('generalCategories', [])]
    finished_at = None
    if status['status'] in ('complete', 'complete_with_item_failures') and len(records) == len(expected):
        finished_at = status.get('completedAt', (run / 'status.json').stat().st_mtime)
    result = {'observedAt': datetime.fromtimestamp(now, ZoneInfo('Australia/Melbourne')).isoformat(timespec='seconds'),
              'processVerifiedLive': live, 'process': process_line, 'status': status, 'sealedItemsVerified': len(records),
              'itemFailures': sum(r.get('itemStatus') == 'failed' for r in records),
              'lastProgressAgeSeconds': now - status.get('updatedAt', finished_at or launch['startedAt']),
              'lengthLimitedOutputs': sum(r['stopReason'] == 'length' for r in records),
              'eta': estimate(tasks, preflight['inputLengths'], records, finished_at if finished_at is not None else now, launch['startedAt'], manifest['maxNewTokens'])}
    if finished_at is not None:
        result['eta']['actualCompletedAt'] = datetime.fromtimestamp(finished_at, ZoneInfo('Australia/Melbourne')).isoformat(timespec='seconds')
        result['eta']['completionTimeSource'] = 'status.completedAt or completion-seal file mtime'
        result['eta']['actualWallSeconds'] = finished_at - launch['startedAt']
    if (run / 'current-item.json').exists():
        result['currentItem'] = json.loads((run / 'current-item.json').read_text())
    if (run / 'failure.json').exists():
        result['failure'] = json.loads((run / 'failure.json').read_text())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    result = inspect(args.run, root)
    temporary = args.run / 'monitor-latest.tmp'
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(args.run / 'monitor-latest.json')
    with (args.run / 'monitor-history.jsonl').open('a') as handle:
        handle.write(json.dumps(result) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'eta'}, indent=2))
    print(json.dumps({k: v for k, v in result['eta'].items() if k != 'tasks'}, indent=2))


if __name__ == '__main__':
    main()
