# Statistical analysis

`summarize.py` computes acquisition, AA, forgetting, signed BWT, G-test mean,
mean general change and worst-category change. It reports means over supplied
seeds, sample SD (null for one seed), and 95% percentile intervals from 4,000
item-bootstrap draws. Resampling is within task/category and paired across
stages, seeds and, when supplied, the comparison method. Seeds are kept fixed;
these intervals describe item sampling, not uncertainty from training seeds.

```sh
python3 experiments/code/summarize.py /path/scored-panels.json \
  --reference /path/baseline-panels.json --output output/comparison.json
python3 -m unittest discover -s experiments/code/tests
```

The input contains `taskOrder` (ordered task identifiers) and `runs` (one object
per seed). Each run has `seed` and four panels:

- `learned`: each task's scored test items immediately after learning it.
- `final`: the same test items after the final stage.
- `generalBase`: each G-test category's scored items under the base model.
- `generalFinal`: the same full G-test items after the final stage.

Each panel maps task/category identifiers to objects of `{item_id: score}`.
Use the shared evaluator's `itemScores`, retaining failed predictions with score
zero. Supply all 100 test items per task and all 200 per G-test category from
runs built with the same dataset manifest. For controls, reuse the fixed
predictor's task scores in both `learned` and `final`. Each run must use identical
item identities; the reference method must match those identities and task order.
The main experiment uses seeds 42, 43 and 44; a single-seed ablation is also accepted.

This utility consumes already-scored panels. Prediction decoding and scoring
remain in the method runners and shared evaluator. It does not load private
archives, invent missing cells or modify source predictions.
