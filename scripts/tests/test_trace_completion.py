import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('completion', Path(__file__).parents[1] / 'build-trace-plus-17.py')
completion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(completion)


class ResamplingTest(unittest.TestCase):
    def test_repeats_preserve_sources_and_have_unique_occurrence_ids(self):
        rows = [json.dumps({'id': str(i), 'prompt': f'question {i}', 'answer': str(i),
                            'references': [str(i)], 'metadata': {'sourceIndex': i}}) for i in range(3)]
        result = completion.resample_split(rows, 10, 'trace:NumGLUE-cm', 'train')
        self.assertEqual(result, completion.resample_split(rows, 10, 'trace:NumGLUE-cm', 'train'))
        self.assertEqual(result[:3], rows)
        records = [json.loads(row) for row in result]
        self.assertEqual(len({row['id'] for row in records}), 10)
        for row in records[3:]:
            source = records[int(row['metadata']['resampledFrom'])]
            for field in ('prompt', 'answer', 'references'):
                self.assertEqual(row[field], source[field])
        self.assertEqual(completion.resample_split(rows, 3, 'task', 'test'), rows)

    def test_empty_split_cannot_borrow_from_another_split(self):
        with self.assertRaises(ValueError):
            completion.resample_split([], 100, 'trace:NumGLUE-cm', 'validation')


if __name__ == '__main__':
    unittest.main()
