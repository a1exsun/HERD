import json
import threading
from types import SimpleNamespace as NS

import openai
import pytest

from local_backend import MPSClient
from protocol import BASE_MODEL_ID


def fake_client(tmp_path, count=3):
    torch = pytest.importorskip('torch')

    class Inputs(dict):
        def to(self, device):
            assert device == 'mps'
            return self

    inputs = Inputs(input_ids=torch.zeros((1, count), dtype=torch.long))
    generations = []

    def generate(**kwargs):
        generations.append(kwargs)
        return torch.cat([inputs['input_ids'], torch.tensor([[9, 10]])], dim=1)

    client = MPSClient.__new__(MPSClient)
    client.lock, client.torch, client.cache = threading.Lock(), torch, tmp_path
    client.contract, client.eos = {'device': 'mps'}, [10]
    client.tokenizer = NS(apply_chat_template=lambda *a, **k: inputs, decode=lambda *a, **k: 'answer', pad_token_id=0)
    client.model = NS(generate=generate)
    return client, generations


def test_completed_calls_are_reused_without_generation(tmp_path):
    client, generations = fake_client(tmp_path)
    request = dict(model=BASE_MODEL_ID, messages=[{'role': 'user', 'content': 'q'}], temperature=0, max_tokens=4096)
    first = client.create(**request)
    second = client.create(**request)
    assert first.choices[0].message.content == second.choices[0].message.content == 'answer'
    assert first.cache_hit is False and second.cache_hit is True
    assert len(generations) == 1
    assert generations[0]['do_sample'] is False
    assert json.loads(next(tmp_path.rglob('*.json')).read_text())['outputTokenIds'] == [9, 10]


def test_input_ceiling_rejects_without_generation(tmp_path):
    client, generations = fake_client(tmp_path, 16385)
    with pytest.raises(openai.BadRequestError, match='maximum context length'):
        client.create(model=BASE_MODEL_ID, messages=[], temperature=0, max_tokens=4096)
    assert generations == []


def test_unavailable_mps_fails_instead_of_loading_on_cpu(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='MPS unavailable'):
        MPSClient(tmp_path, tmp_path)
