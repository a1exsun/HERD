"""Stream metrics and paired item-bootstrap intervals from scored test panels.

Input: {"taskOrder": [...], "runs": [{"seed": 42, "learned": ..., "final": ...,
"generalBase": ..., "generalFinal": ...}, ...]}. Each panel maps task/category
names to {item_id: score}. Learned means the diagonal R[j,j]; final means R[T,j].
Use the evaluator's itemScores, including zero scores for failed predictions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from statistics import mean, stdev

PANELS = ('learned', 'final', 'generalBase', 'generalFinal')


def validate(data):
    tasks, runs = data['taskOrder'], data['runs']
    if not tasks or len(set(tasks)) != len(tasks) or not runs:
        raise ValueError('Distinct ordered tasks and at least one seed are required')
    if len({r['seed'] for r in runs}) != len(runs):
        raise ValueError('Duplicate seeds')
    first = runs[0]
    if not first['generalBase']:
        raise ValueError('General-capability panels are required')
    signature = {}
    for panel in PANELS:
        expected = set(tasks) if panel in ('learned', 'final') else set(first['generalBase'])
        for run in runs:
            if set(run[panel]) != expected:
                raise ValueError(f'Incomplete {panel} groups')
            for group, scores in run[panel].items():
                if not scores or any(isinstance(s, bool) or not isinstance(s, (float, int))
                                     or not math.isfinite(s) or not 0 <= s <= 1 for s in scores.values()):
                    raise ValueError(f'Invalid item scores: {panel}/{group}')
                key = ('task' if panel in ('learned', 'final') else 'general', group)
                ids = sorted(scores)
                if signature.setdefault(key, ids) != ids:
                    raise ValueError(f'Item identities differ: {panel}/{group}')
    return signature


def metrics(run, tasks, sample):
    def panel_means(panel, kind):
        return {g: mean(scores[i] for i in sample[(kind, g)]) for g, scores in run[panel].items()}
    learned, final = panel_means('learned', 'task'), panel_means('final', 'task')
    base, general = panel_means('generalBase', 'general'), panel_means('generalFinal', 'general')
    differences = [final[t] - learned[t] for t in tasks[:-1]]
    changes = [general[g] - base[g] for g in base]
    return {'acquisition': mean(learned.values()), 'AA': mean(final.values()),
            'forgetting': mean(max(0, -v) for v in differences) if differences else 0.0,
            'BWT': mean(differences) if differences else 0.0,
            'generalMean': mean(general.values()), 'deltaC': mean(changes), 'deltaCMin': min(changes)}


def average_metrics(data, sample):
    points = [metrics(run, data['taskOrder'], sample) for run in data['runs']]
    return {key: mean(p[key] for p in points) for key in points[0]}


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(data, reference=None, *, draws=4000, seed=42):
    """Resample items within each task/category, paired across stages, seeds and methods.

    Seeds remain fixed. Point estimates average per-seed metrics, and sample SD
    describes seed variation; intervals describe item sampling uncertainty.
    """
    if draws < 2:
        raise ValueError('At least two bootstrap draws are required')
    signature = validate(data)
    if reference is not None:
        if validate(reference) != signature or reference['taskOrder'] != data['taskOrder']:
            raise ValueError('Paired methods must use the same tasks and item identities')
    points = [metrics(run, data['taskOrder'], signature) for run in data['runs']]
    center = average_metrics(data, signature)
    samples = {key: [] for key in center}
    paired = {key: [] for key in center}
    rng = random.Random(seed)
    for _ in range(draws):
        sample = {key: rng.choices(ids, k=len(ids)) for key, ids in signature.items()}
        ours = average_metrics(data, sample)
        theirs = average_metrics(reference, sample) if reference is not None else None
        for key in center:
            samples[key].append(ours[key])
            if theirs is not None:
                paired[key].append(ours[key] - theirs[key])
    interval = lambda values: [quantile(values, 0.025), quantile(values, 0.975)]
    result = {'seeds': [r['seed'] for r in data['runs']], 'draws': draws, 'bootstrapSeed': seed,
              'mean': center, 'sampleSd': {k: stdev(p[k] for p in points) if len(points) > 1 else None for k in center},
              'interval95': {k: interval(v) for k, v in samples.items()}}
    if reference is not None:
        other = average_metrics(reference, signature)
        result['pairedDifference'] = {k: {'mean': center[k] - other[k], 'interval95': interval(paired[k])} for k in center}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.scores.read_text()),
                       json.loads(args.reference.read_text()) if args.reference else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
