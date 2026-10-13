import copy
import json
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import run


def reply(content, calls=(), reason='stop'):
    return NS(choices=[NS(finish_reason=reason, message=NS(content=content, tool_calls=calls))])


def predictor(responses):
    requests = []
    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return responses.pop(0)
    client = object.__new__(run.StaticPredictor)
    client.max_new_tokens = 128
    client.client = NS(chat=NS(completions=NS(create=create)))
    return client, requests


def test_tool_loop_uses_frozen_prompt_and_only_sample_workspace(tmp_path, monkeypatch):
    call = NS(id='c1', function=NS(name='python', arguments=json.dumps({'code': 'print(3*3)'})))
    client, requests = predictor([reply('', [call], 'tool_calls'), reply('9')])
    seen = []
    def tool(workspace, code):
        seen.append((workspace, code))
        return {'stdout': '9\n', 'stderr': '', 'exitCode': 0, 'timedOut': False}
    monkeypatch.setattr(run, 'run_sandbox_python', tool)
    assert client.generate('Compute 3*3.', workspace=tmp_path) == '9'
    assert seen == [(tmp_path, 'print(3*3)')]
    assert requests[0]['messages'] == [
        {'role': 'system', 'content': run.SYSTEM_PROMPT}, {'role': 'user', 'content': 'Compute 3*3.'}]
    assert requests[0]['tools'] == [run.PYTHON_TOOL]
    result = requests[1]['messages'][-1]
    assert result['role'] == 'tool' and result['tool_call_id'] == 'c1'
    assert json.loads(result['content'])['stdout'] == '9\n'
    assert requests[0]['extra_body']['chat_template_kwargs']['enable_thinking'] is False


@pytest.mark.parametrize('response', [reply('partial', reason='length'), reply('')])
def test_incomplete_answers_are_not_silently_scored(response, tmp_path):
    client, _ = predictor([response])
    with pytest.raises((RuntimeError, ValueError)):
        client.generate('question', workspace=tmp_path)


def child(code):
    return subprocess.Popen([sys.executable, '-I', '-c', code], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, start_new_session=True)


def test_output_capture_stops_at_limit():
    result = run.capture_process(child('import os\nwhile True: os.write(1, b"x" * 8192)'), output_limit=1024)
    assert len(result['stdout']) == 1024
    assert result['exitCode'] != 0 and not result['timedOut']


def test_timeout_preserves_output_once():
    result = run.capture_process(child('import time\nprint("one", flush=True)\ntime.sleep(5)'), timeout=0.5)
    assert result['stdout'] == 'one\n'
    assert result['timedOut'] and result['exitCode'] != 0


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS sandbox boundary')
def test_real_sandbox_allows_computation_and_blocks_external_read(tmp_path):
    assert run.run_sandbox_python(tmp_path / 'sample', 'print(2+2)')['stdout'] == '4\n'
    private = tmp_path / 'outside-sample.txt'
    private.write_text('must not be readable')
    result = run.run_sandbox_python(tmp_path / 'sample', f'print(open({str(private)!r}).read())')
    assert result['exitCode'] != 0 and not result['stdout']


class StubPredictor:
    def __init__(self, outcome):
        self.outcome = outcome
        self.last_trace = {}

    def generate(self, prompt, *, workspace):
        self.last_trace = {'turns': 1, 'toolCalls': 0, 'promptTokens': 5, 'completionTokens': 2, 'finishReason': 'stop'}
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def status_error(kind, code):
    import httpx
    import openai
    response = httpx.Response(code, request=httpx.Request('POST', 'http://127.0.0.1/v1/chat/completions'))
    return getattr(openai, kind)('error', response=response, body=None)


def test_item_failures_score_zero_but_server_faults_abort(tmp_path):
    item = {'id': 'a', 'group': 'trace:X', 'prompt': 'p'}
    done = run.predict_item(StubPredictor('B'), item, tmp_path)
    assert done['itemStatus'] == 'complete' and done['prediction'] == 'B' and done['turns'] == 1
    for failure in (RuntimeError('Incomplete static-harness response: length'), ValueError('no answer'),
                    status_error('UnprocessableEntityError', 422)):
        failed = run.predict_item(StubPredictor(failure), item, tmp_path)
        assert failed['itemStatus'] == 'failed' and failed['prediction'] == '' and failed['error']
    with pytest.raises(Exception, match='error'):
        run.predict_item(StubPredictor(status_error('InternalServerError', 500)), item, tmp_path)


def test_score_run_reports_constant_predictor_metrics(tmp_path):
    panel, gold = [], {}
    groups = ['trace:X'] + ['general:' + c for c in run.GENERAL_CATEGORIES]
    for group in groups:
        rows = [{'id': f'{group}:{i}', 'prompt': 'p', 'answer': 'A', 'references': ['A']} for i in range(2)]
        gold[group] = {'metric': 'exact_match', 'rows': rows}
        panel += [{'group': group, 'id': row['id'], 'prompt': 'p'} for row in rows]
    for index, item in enumerate(panel):
        prediction = 'A' if index % 2 == 0 else 'B'
        run.atomic_json(run.item_path(tmp_path, item['group'], index),
                        {'id': item['id'], 'group': item['group'], 'prediction': prediction,
                         'itemStatus': 'complete', 'seconds': 1.0, 'turns': 1})
    probes = [panel[2]['id']]  # first item of the first G category
    result = run.score_run(tmp_path, panel, {'gold': gold, 'probeIds': probes})
    metrics = result['metrics']
    assert metrics['AA'] == pytest.approx(0.5) and metrics['forgetting'] == 0.0
    assert metrics['generalMacro'] == pytest.approx(0.5)
    assert metrics['generalCategoryScores']['general-knowledge']['probeScore'] == 1.0
    assert 'generalProbeMacro' not in metrics  # other categories have no probe item in this fixture
    assert metrics['items'] == len(panel) and metrics['itemFailures'] == 0
