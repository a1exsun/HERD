import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('base_monitor', Path(__file__).parents[1] / 'monitor.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_eta_uses_task_specific_rates_and_keeps_failed_items_done():
    tasks = [{'id': 'a', 'outputType': 'closed'}, {'id': 'b', 'outputType': 'open'}]
    records = [dict(id=f'a{i}', group='a', outputTokens=2, promptTokens=100, seconds=1) for i in range(5)]
    records.append(dict(id='failed', group='b', outputTokens=0, promptTokens=20000, seconds=.1, itemStatus='failed'))
    lengths = [dict(id='new-a', group='a', tokens=100), dict(id='new-b', group='b', tokens=100), dict(id='failed', group='b', tokens=20000)]
    result = monitor.estimate(tasks, lengths, records, 100, 0, 5120)
    assert result['tasks'][0]['remainingSecondsCentral'] == 1
    assert result['tasks'][1]['remaining'] == 1
    assert result['tasks'][1]['remainingSecondsCentral'] > 1
    assert result['totalSecondsCentral'] == 100 + result['remainingSecondsCentral']
    assert result['remainingSecondsLow'] < result['remainingSecondsCentral'] < result['remainingSecondsHigh']


def test_longer_inputs_add_cost_without_multiplying_decode_time():
    records = [dict(id=f'a{i}', group='a', outputTokens=512, promptTokens=n, seconds=1 + .02*n + .1*512) for i, n in enumerate([100, 200, 300, 400, 500])]
    result = monitor.estimate([{'id': 'a', 'outputType': 'open'}], [dict(id='next', group='a', tokens=600)], records, 100, 0, 512)
    import pytest
    assert result['inputSecondsPerToken'] == pytest.approx(.02)
    assert result['remainingSecondsCentral'] == pytest.approx(1 + .02*600 + .1*512)
    assert result['inputCalibrationSamples'] == 5


def test_finished_duration_does_not_grow_with_monitor_time(tmp_path, monkeypatch):
    import json
    import os
    from types import SimpleNamespace
    run = tmp_path / 'run'; run.mkdir(); (run / 'items').mkdir()
    dataset = tmp_path / 'experiments/data/trace-plus-17'; dataset.mkdir(parents=True)
    manifest = {'profile': 'full', 'maxNewTokens': 5120, 'provenance': {'taskDataset': str(dataset)}}
    (run / 'manifest.json').write_text(json.dumps(manifest))
    (run / 'launch.json').write_text(json.dumps({'pid': 999, 'startedAt': 0}))
    (run / 'status.json').write_text(json.dumps({'status': 'complete', 'completed': 1}))
    os.utime(run / 'status.json', (100, 100))
    (run / 'preflight.json').write_text(json.dumps({'inputLengths': [{'id': 'a', 'group': 't', 'tokens': 10}]}))
    (dataset / 'manifest.json').write_text(json.dumps({'tasks': [{'id': 't', 'outputType': 'closed'}]}))
    row = {'id': 'a', 'group': 't', 'contractSha256': monitor.digest(manifest), 'outputTokens': 2, 'promptTokens': 10, 'seconds': 1, 'stopReason': 'eos'}
    row['sha256'] = monitor.digest(row)
    (run / 'items/0000.json').write_text(json.dumps(row))
    monkeypatch.setattr(monitor.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1, stdout=''))
    monkeypatch.setattr(monitor.time, 'time', lambda: 200)
    first = monitor.inspect(run, tmp_path)
    monkeypatch.setattr(monitor.time, 'time', lambda: 300)
    second = monitor.inspect(run, tmp_path)
    assert first['eta']['actualWallSeconds'] == second['eta']['actualWallSeconds'] == 100
    assert first['eta']['finishCentral'] == second['eta']['finishCentral']
    assert second['eta']['remainingSecondsCentral'] == 0
