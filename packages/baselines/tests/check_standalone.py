"""Copy each baseline outside the repository and test it in a fresh locked env.

Only the pinned GPT-2 and Punkt data files are supplied to the copied runtime.
The canonical evaluator is used in a separate process to derive conformance
expectations; baseline processes cannot import it or any other project package.
"""
from __future__ import annotations

import argparse
import importlib.abc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

METHODS = ('base', 'static-harness', 'ace', 'hcl-stable', 'seq-lora', 'uniform-replay', 'treelora', 'moram',
           'sdft')


def cases():
    result = []
    def add(metric, prediction, answer='', metadata=None, prompt='p'):
        result.append({'metric': metric, 'predictions': [{'id': 'a', 'prediction': prediction}],
                       'goldItems': [{'id': 'a', 'answer': answer, 'prompt': prompt, 'metadata': metadata or {}}]})
    add('exact_match', 'Paris!', 'paris')
    add('strict_exact_match', 'Paris', 'paris')
    add('leading_choice_accuracy', ' A', 'A')
    add('edit_similarity', 'return <NUM_LIT>', 'return 0')
    add('rouge_l', 'a a b', 'a b', {'source': 'trace'})
    add('rouge_l', 'running', 'runs', {'source': 'superni'})
    add('rouge_l', '你好 世界', '你好 世界', {'sourceTaskId': '1219'})
    add('sari', 'The cat sat.', 'The cat sat.', prompt='Paragraph:\nThe small cat sat.\n\nSimplification:')
    add('final_number_match', 'Work: 1 + 1 = 2', '2')
    add('ifeval_strict', 'One sentence. Another sentence.', metadata={
        'instructionIdList': ['length_constraints:number_sentences'],
        'instructionKwargs': [{'num_sentences': 2, 'relation': 'at least'}]})
    add('ifeval_strict', 'oops, comma', metadata={
        'instructionIdList': ['punctuation:no_comma'], 'instructionKwargs': [{}]})
    add('ifeval_strict', 'THIS IS ENGLISH TEXT WITHOUT ANY COMMA IN IT', metadata={
        'instructionIdList': ['change_case:english_capital'], 'instructionKwargs': [{}]})
    return result


def worker(package: Path, expectations: Path):
    package = package.resolve()
    sys.path.insert(0, str(package))
    class NoProjectImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0].startswith(('hero_', 'dsh')):
                raise ImportError(f'Forbidden external project import: {fullname}')
    sys.meta_path.insert(0, NoProjectImports())
    import scoring
    for case, expected in zip(cases(), json.loads(expectations.read_text()), strict=True):
        actual = scoring.score_predictions(case['metric'], case['predictions'], case['goldItems'])
        assert actual == expected, (case, actual, expected)
    import run
    assert Path(run.__file__).resolve().parent == package
    if package.name == 'ace':
        import ace_adapter
        ace_adapter.activate_upstream()
        from ace import ACE
        assert ACE is not None
    import pytest
    status = pytest.main(['-q', '--import-mode=importlib', str(package / 'tests')])
    # Imported implementation modules must all reside in the copied directory.
    for name in ('run', 'protocol', 'scoring', 'scorer_assets', 'controls', 'sft',
                 'hcl_stable', 'treelora_cl', 'ace_adapter', 'ifeval',
                 'replay', 'sdft', 'moram_cl', 'upstream', 'local_backend', 'vllm_decode'):
        module = sys.modules.get(name)
        if module is not None:
            assert Path(module.__file__).resolve().is_relative_to(package), name
    raise SystemExit(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('methods', nargs='*', choices=METHODS)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--expectations', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.expectations)
        return
    baselines = Path(__file__).resolve().parents[1]
    repo = baselines.parents[1]
    selected = args.methods or METHODS
    with tempfile.TemporaryDirectory(prefix='baseline-isolation-') as temporary:
        directory = Path(temporary)
        assets = directory / 'assets'
        assets.mkdir()
        env = dict(os.environ, HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
        env.pop('PYTHONPATH', None)
        for variable, source in (
            ('TRACE_GPT2_TOKENIZER', 'tokenizers/gpt2/tokenizer.json'),
            ('TRACE_PUNKT_TOKENIZER', 'nltk/punkt/english.pickle'),
        ):
            path = assets / Path(source).name
            shutil.copy2(repo / 'experiments/data' / source, path)
            env[variable] = str(path)
        expectations = directory / 'expected.json'
        command = """import json,sys
sys.path.insert(0, sys.argv[1])
from hero_evaluation.scoring import score_predictions
print(json.dumps([score_predictions(c['metric'], c['predictions'], c['goldItems']) for c in json.load(sys.stdin)]))
"""
        # The base env has the evaluator's pinned third-party dependencies only.
        subprocess.run(['uv', 'sync', '--quiet', '--locked', '--project', str(baselines / 'base')], check=True)
        result = subprocess.run([str(baselines / 'base/.venv/bin/python'), '-I', '-c', command,
                                 str(repo / 'packages/evaluator/python/src')],
                                input=json.dumps(cases()), text=True, capture_output=True, check=True)
        expectations.write_text(result.stdout)
        runner = directory / 'check_standalone.py'
        shutil.copy2(__file__, runner)
        for name in selected:
            package = directory / name
            shutil.copytree(baselines / name, package, ignore=shutil.ignore_patterns('.venv', '__pycache__', '.pytest_cache', '*.pyc'))
            print(f'Isolated baseline: {name}', flush=True)
            subprocess.run(['uv', 'sync', '--quiet', '--locked', '--project', str(package)], check=True, env=env, cwd=directory)
            python = str(package / '.venv/bin/python')
            subprocess.run([python, '-I', str(runner), '--worker', str(package), '--expectations', str(expectations)],
                           check=True, env=env, cwd=package)
            subprocess.run([python, str(package / 'run.py'), '--help'], check=True, env=env, cwd=directory, stdout=subprocess.DEVNULL)
        print(f'{len(selected)} standalone baselines passed, including all {len(cases())} scoring conformance cases each.')


if __name__ == '__main__':
    main()
