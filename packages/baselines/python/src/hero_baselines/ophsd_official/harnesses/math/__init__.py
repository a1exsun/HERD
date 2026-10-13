"""DeepMath Plan-Solve harness package.

Public API:
    PlanSolveHarness     — two-phase Plan → Solve harness (with optional GT injection).
    ZeroShotSolver       — single-call baseline (no scaffolding).
    extract_boxed        — pull the final ``\\boxed{...}`` answer from a generation.
    answers_match        — numerical / symbolic equivalence (Hendrycks + math_verify).
    evaluate_predictions — accuracy, abstention rate, per-benchmark breakdown.
    load_test_data       — parquet loader for the math eval set.
    load_train_data      — parquet loader for the math training set.
"""

from .plan_solve import PlanSolveHarness, ZeroShotSolver, build_clients

__all__ = [
    "PlanSolveHarness",
    "ZeroShotSolver",
    "build_clients",
]
