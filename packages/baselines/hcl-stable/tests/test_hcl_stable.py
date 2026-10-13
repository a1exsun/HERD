from hcl_stable import (
    CandidateEvaluation,
    HarnessState,
    build_component_optimizer_prompt,
    choose_stable_candidate,
    evolve_stable_harness,
)


def test_stable_profile_rejects_one_regressed_anchor():
    deployed = HarnessState()
    good = CandidateEvaluation("good", deployed, 0.8, {"a": True, "b": False})
    regressed = CandidateEvaluation("regressed", deployed, 0.9, {"a": False, "b": False})
    decision = choose_stable_candidate(
        deployed_current_score=0.7,
        deployed_anchor_success={"a": True, "b": False},
        candidates=[regressed, good],
        minimum_current_gain=0.05,
    )
    assert decision.committed
    assert decision.selected == good


def test_optimizer_prompt_never_receives_historical_anchors():
    prompt = build_component_optimizer_prompt(
        deployed=HarnessState(),
        component="experience_memory",
        interaction_evidence={"u": "new", "f": "success"},
    )
    assert "new" in prompt
    assert "historical anchor" not in prompt.lower()


def test_component_updates_are_sequential_and_anchor_private():
    seen_prompts = []

    def proposer(state, component, prompt, candidate_index):
        seen_prompts.append(prompt)
        return {"component": component, "candidate": candidate_index}

    def evaluator(state):
        filled = sum(bool(value) for value in (
            state.task_interface,
            state.experience_memory,
            state.capability_map,
            state.adaptive_router,
        ))
        return 0.5 + filled * 0.1, {"private-anchor": True}, True

    state, audits = evolve_stable_harness(
        deployed=HarnessState(),
        interaction_evidence={"u": "current-train", "f": "success"},
        proposer=proposer,
        evaluator=evaluator,
        alternatives_per_component=1,
        minimum_current_gain=0.05,
    )
    assert all((state.task_interface, state.experience_memory, state.capability_map, state.adaptive_router))
    assert len(audits) == 4
    assert all("private-anchor" not in prompt for prompt in seen_prompts)


def test_proposals_that_are_not_one_json_object_are_invalid_candidates():
    from run import INVALID, is_invalid, parse_component
    from hcl_stable import replace_component

    assert parse_component('```json\n{"rules": ["a"]}\n```') == {"rules": ["a"]}
    assert INVALID in parse_component("[1, 2]") and INVALID in parse_component("no json")
    assert is_invalid(replace_component(HarnessState(), "capability_map", parse_component("no json")))
    assert not is_invalid(replace_component(HarnessState(), "capability_map", {"rules": []}))


def test_invalid_candidates_keep_the_anchor_set_and_are_never_committed():
    from run import INVALID

    anchors = {"t1:a": True, "t1:b": False}

    def evaluator(state):
        if INVALID in state.task_interface or INVALID in state.experience_memory \
                or INVALID in state.capability_map or INVALID in state.adaptive_router:
            return 0.0, dict.fromkeys(anchors, False), False
        return 0.5, anchors, True

    state, audits = evolve_stable_harness(
        deployed=HarnessState(), interaction_evidence={"interactions": []},
        proposer=lambda *args: {INVALID: "not JSON"}, evaluator=evaluator,
        alternatives_per_component=1, minimum_current_gain=0.01)
    assert state == HarnessState()
    assert all(audit.rejection_reasons[0].endswith(":invalid") for audit in audits)


def test_evidence_follows_a_seeded_uniform_order_and_packs_while_the_prompt_fits():
    from run import evidence_order, pack_evidence

    order = evidence_order(50, 7)
    assert sorted(order) == list(range(50)) and order == evidence_order(50, 7) != evidence_order(50, 8)
    interactions = [{"u": "x" * size} for size in (5, 9, 2, 7, 1)]
    fits = lambda subset: sum(len(i["u"]) for i in subset) <= 10
    assert pack_evidence(interactions, fits) == [{"u": "x" * 5}, {"u": "x" * 2}, {"u": "x" * 1}]


def test_rendered_harness_round_trips_through_the_sealed_json():
    from run import render_harness, state_from_json

    state = HarnessState({"format": "Answer with one letter."}, {"lessons": ["x"]}, {}, {"route": "direct"})
    assert state_from_json(state.to_json()) == state
    assert "Answer with one letter." in render_harness(state)
