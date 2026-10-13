from __future__ import annotations

from dataclasses import dataclass, field
import json
import random
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class HarnessState:
    task_interface: Mapping[str, Any] = field(default_factory=dict)
    experience_memory: Mapping[str, Any] = field(default_factory=dict)
    capability_map: Mapping[str, Any] = field(default_factory=dict)
    adaptive_router: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "taskInterface": self.task_interface,
            "experienceMemory": self.experience_memory,
            "capabilityMap": self.capability_map,
            "adaptiveRouter": self.adaptive_router,
        }, sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    state: HarnessState
    current_score: float
    anchor_success: Mapping[str, bool]
    valid: bool = True


@dataclass(frozen=True)
class CommitDecision:
    committed: bool
    selected: CandidateEvaluation | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ComponentUpdateAudit:
    component: str
    deployed_score: float
    candidate_ids: tuple[str, ...]
    committed_candidate_id: str | None
    rejection_reasons: tuple[str, ...]


def historical_loss(
    deployed_anchor_success: Mapping[str, bool],
    candidate_anchor_success: Mapping[str, bool],
) -> int:
    if set(deployed_anchor_success) != set(candidate_anchor_success):
        raise ValueError("Deployed and candidate anchor sets must be identical.")
    return sum(
        1
        for anchor_id, was_solved in deployed_anchor_success.items()
        if was_solved and not candidate_anchor_success[anchor_id]
    )


def choose_stable_candidate(
    *,
    deployed_current_score: float,
    deployed_anchor_success: Mapping[str, bool],
    candidates: Sequence[CandidateEvaluation],
    minimum_current_gain: float,
    historical_budget: int = 0,
    seed: int = 42,
) -> CommitDecision:
    """Paper Eq. (9)-(13), with B_n=0 as the HCL-stable profile."""
    if historical_budget < 0:
        raise ValueError("historical_budget must be non-negative.")
    admissible: list[CandidateEvaluation] = []
    failures: list[str] = []
    for candidate in candidates:
        gain = candidate.current_score - deployed_current_score
        loss = historical_loss(deployed_anchor_success, candidate.anchor_success)
        if not candidate.valid:
            failures.append(f"{candidate.candidate_id}:invalid")
        elif gain + 1e-12 < minimum_current_gain:
            failures.append(f"{candidate.candidate_id}:insufficient-current-gain")
        elif loss > historical_budget:
            failures.append(f"{candidate.candidate_id}:historical-budget-exceeded")
        else:
            admissible.append(candidate)
    if not admissible:
        return CommitDecision(False, None, tuple(failures) or ("no-candidate",))

    best_score = max(candidate.current_score for candidate in admissible)
    tied = sorted(
        (candidate for candidate in admissible if candidate.current_score == best_score),
        key=lambda candidate: candidate.candidate_id,
    )
    selected = random.Random(seed).choice(tied)
    return CommitDecision(True, selected, ())


def build_component_optimizer_prompt(
    *,
    deployed: HarnessState,
    component: str,
    interaction_evidence: Mapping[str, Any],
) -> str:
    if component not in {"task_interface", "experience_memory", "capability_map", "adaptive_router"}:
        raise ValueError(f"Unknown HCL component: {component}")
    # Historical anchors are deliberately absent: the paper reserves them for the evaluator.
    return "\n\n".join([
        "Propose one isolated Harness Continual Learning component update.",
        f"Component: {component}",
        "Deployed harness:",
        deployed.to_json(),
        "Current interaction evidence (u, i, z, y, f):",
        json.dumps(interaction_evidence, sort_keys=True, ensure_ascii=False),
        "Return only the complete replacement component as JSON.",
    ])


def replace_component(state: HarnessState, component: str, replacement: Mapping[str, Any]) -> HarnessState:
    values = {
        "task_interface": state.task_interface,
        "experience_memory": state.experience_memory,
        "capability_map": state.capability_map,
        "adaptive_router": state.adaptive_router,
    }
    if component not in values:
        raise ValueError(f"Unknown HCL component: {component}")
    values[component] = dict(replacement)
    return HarnessState(**values)


def evolve_stable_harness(
    *,
    deployed: HarnessState,
    interaction_evidence: Mapping[str, Any],
    proposer: Callable[[HarnessState, str, str, int], Mapping[str, Any]],
    evaluator: Callable[[HarnessState], tuple[float, Mapping[str, bool], bool]],
    alternatives_per_component: int,
    minimum_current_gain: float,
    seed: int = 42,
) -> tuple[HarnessState, tuple[ComponentUpdateAudit, ...]]:
    """Sequential HCL component optimization with the stable B_n=0 gate.

    The proposer callback receives only the deployed harness, component name,
    optimizer prompt, and candidate index. Historical anchors exist solely
    inside the evaluator callback and are therefore unavailable to proposal.
    """
    if alternatives_per_component < 1:
        raise ValueError("alternatives_per_component must be positive.")
    state = deployed
    audits: list[ComponentUpdateAudit] = []
    components = ("task_interface", "experience_memory", "capability_map", "adaptive_router")
    for component_index, component in enumerate(components):
        deployed_score, deployed_anchors, deployed_valid = evaluator(state)
        if not deployed_valid:
            raise ValueError("The deployed HCL harness must pass the evaluator validity checks.")
        prompt = build_component_optimizer_prompt(
            deployed=state,
            component=component,
            interaction_evidence=interaction_evidence,
        )
        candidates: list[CandidateEvaluation] = []
        for candidate_index in range(alternatives_per_component):
            replacement = proposer(state, component, prompt, candidate_index)
            candidate_state = replace_component(state, component, replacement)
            score, anchors, valid = evaluator(candidate_state)
            candidates.append(CandidateEvaluation(
                candidate_id=f"{component}:{candidate_index}",
                state=candidate_state,
                current_score=score,
                anchor_success=anchors,
                valid=valid,
            ))
        decision = choose_stable_candidate(
            deployed_current_score=deployed_score,
            deployed_anchor_success=deployed_anchors,
            candidates=candidates,
            minimum_current_gain=minimum_current_gain,
            historical_budget=0,
            seed=seed + component_index,
        )
        if decision.committed and decision.selected is not None:
            state = decision.selected.state
        audits.append(ComponentUpdateAudit(
            component=component,
            deployed_score=deployed_score,
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
            committed_candidate_id=(decision.selected.candidate_id if decision.selected else None),
            rejection_reasons=decision.reasons,
        ))
    return state, tuple(audits)
