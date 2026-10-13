# Scripts

Monorepo-level automation: the DSH host and smokes, dataset and G-test
construction. Experiment-specific
programs live in [`../experiments/code`](../experiments/code/README.md).

## DSH host and local model

| Script | Purpose |
| --- | --- |
| `dsh-host-web.mjs` | Installs the HERO bundle into the normal DSH Web profile and starts the host on `http://127.0.0.1:3080` (`pnpm dsh:web`). |
| `dsh-test-profile.mjs` | Creates the isolated test profile under `experiments/runtime/` (`pnpm dsh:config`). |
| `dsh-smoke.mjs` | Integration smoke of the HERO manifest and routes in the test profile (`pnpm dsh:smoke`). |
| `dsh-local-model-smoke.mjs` | End-to-end inference through DSH's `hero-qwen-local` provider (`pnpm model:smoke`). |

## Data

| Script | Purpose |
| --- | --- |
| `build-trace-plus-17.py` | Completes TRACE+17 from its selection stage into `experiments/data/trace-plus-17/` (`pnpm dataset:build`). |
| `archive-trace-source.mjs` | Downloads the official TRACE archive into `experiments/data/raw/trace/`. |
| `fetch-general-sources.py` | Fetches the pinned G-test sources into `experiments/data/raw/general-v2/`, checked against `datasets/general-test-v1.json`. |
| `prepare-general-test.py` | Builds the five-category G-test, probe, development and non-test panels and writes `datasets/general-test-v1.json`. |
| `prepare-general-probes.py` | Reserves the bounded non-test SuperNI general set and its disjoint held-out set and writes `datasets/general-probes-v1.json`. |
