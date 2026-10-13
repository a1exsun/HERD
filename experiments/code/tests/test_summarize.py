from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('summary', Path(__file__).parents[1] / 'summarize.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryTest(unittest.TestCase):
    def test_metrics_and_identical_paired_interval(self):
        run = {'seed': 42, 'learned': {'a': {'x': 1.0}, 'b': {'y': 0.5}},
               'final': {'a': {'x': 0.5}, 'b': {'y': 0.5}},
               'generalBase': {'knowledge': {'z': 0.8}}, 'generalFinal': {'knowledge': {'z': 0.6}}}
        data = {'taskOrder': ['a', 'b'], 'runs': [run]}
        result = summary.summarize(data, data, draws=20)
        self.assertEqual(result['mean']['AA'], 0.5)
        self.assertEqual(result['mean']['acquisition'], 0.75)
        self.assertEqual(result['mean']['forgetting'], 0.5)
        self.assertEqual(result['mean']['BWT'], -0.5)
        self.assertAlmostEqual(result['mean']['deltaCMin'], -0.2)
        self.assertIsNone(result['sampleSd']['AA'])
        self.assertTrue(all(v['interval95'] == [0.0, 0.0] for v in result['pairedDifference'].values()))
        other = deepcopy(data)
        other['runs'][0]['final']['a'] = {'different-item': 0.5}
        with self.assertRaises(ValueError):
            summary.summarize(data, other, draws=20)

    def test_seed_spread_and_item_resampling(self):
        first = {'seed': 42, 'learned': {'a': {'x': 1, 'y': 1}},
                 'final': {'a': {'x': 0, 'y': 1}},
                 'generalBase': {'knowledge': {'z': 1}}, 'generalFinal': {'knowledge': {'z': 1}}}
        second = deepcopy(first)
        second['seed'] = 43
        second['final']['a']['x'] = 1
        data = {'taskOrder': ['a'], 'runs': [first, second]}
        result = summary.summarize(data, data, draws=100)
        self.assertEqual(result['mean']['AA'], 0.75)
        self.assertAlmostEqual(result['sampleSd']['AA'], 0.5 / 2**0.5)
        self.assertEqual(result['interval95']['AA'], [0.5, 1.0])
        self.assertEqual(result['pairedDifference']['AA']['interval95'], [0, 0])


if __name__ == '__main__':
    unittest.main()
