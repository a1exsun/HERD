from __future__ import annotations

import argparse
import json

from .protocol import BASE_MODEL_ID, DATASET_NAME, DATASET_SEED, STAGE_COUNT


def main() -> None:
    parser = argparse.ArgumentParser(prog="hero-baselines")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("protocol", help="print the pinned matched-baseline protocol")
    args = parser.parse_args()
    if args.command == "protocol":
        print(json.dumps({
            "baseModel": BASE_MODEL_ID,
            "dataset": DATASET_NAME,
            "seed": DATASET_SEED,
            "stages": STAGE_COUNT,
            "evaluator": "hero_evaluation.scoring",
        }, indent=2))
