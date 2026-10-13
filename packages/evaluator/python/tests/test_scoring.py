import pytest
from hero_evaluation.scoring import score_predictions, score_one, sari, normalize_answer, final_number


def test_superni_punctuation_removal_and_stemming():
    assert normalize_answer("don't-stop!") == "dontstop"
    assert score_one("rouge_l", "running", "runs", {"source": "superni"}) == 1


def test_trace_case_and_first_character_are_not_repaired():
    assert score_one("strict_exact_match", "positive", "Positive", {}) == 0
    assert score_one("leading_choice_accuracy", " A", "A", {}) == 0
    assert score_one("leading_choice_accuracy", "a", "A", {}) == 0
    assert score_one("leading_choice_accuracy", "A explanation", "A", {}) == 1


def test_trace_rouge_uses_upstream_sentence_union():
    assert score_one("rouge_l", "a a b", "a b", {"source": "trace"}) == pytest.approx(1, abs=1e-7)
    assert score_one("rouge_l", "A", "a", {"source": "trace"}) == 0


def test_py150_literal_and_integer_percentage():
    assert score_one("edit_similarity", "return <NUM_LIT>", "return 0", {}) == 1
    assert score_one("edit_similarity", "ab", "abc", {}) == 0.8


def test_sari_official_multireference_example():
    value = sari("About 95 species are currently accepted .", "About 95 you now get in .", [
        "About 95 species are currently known .", "About 95 species are now accepted .", "95 species are now accepted ."
    ])
    assert value == pytest.approx(0.26953601953601954)


def test_empty_output_and_complete_coverage():
    gold = [{"id": "a", "prompt": "p", "references": ["a", "b"]}]
    assert score_predictions("exact_match", [{"id": "a", "prediction": "b"}], gold)["score"] == 1
    assert score_predictions("exact_match", [{"id": "a", "prediction": ""}], gold)["score"] == 0
    with pytest.raises(ValueError, match="coverage mismatch"):
        score_predictions("exact_match", [], gold)


def test_final_number_match_reads_the_last_number_not_the_whole_string():
    assert final_number("The answer is $1,234.") == 1234
    assert final_number("x=5, so y = -3.5") == -3.5
    assert final_number("no digits here") is None
    # GSM8K gold is the value after ####; the prediction is a chain of thought.
    assert score_one("final_number_match", "9 * 2 = $18 every day.", "18", {}) == 1
    assert score_one("final_number_match", "I think it is 71.", "72", {}) == 0
    assert score_one("final_number_match", "no digits", "72", {}) == 0


def test_ifeval_strict_requires_every_declared_instruction():
    def item(identifier, kwargs):
        return {"id": "a", "prompt": "Write without commas.",
                "metadata": {"instructionIdList": identifier, "instructionKwargs": kwargs}}

    gold = [item(["punctuation:no_comma"], [{}])]
    assert score_predictions("ifeval_strict", [{"id": "a", "prediction": "no commas here"}], gold)["score"] == 1
    assert score_predictions("ifeval_strict", [{"id": "a", "prediction": "oops, comma"}], gold)["score"] == 0
    assert score_predictions("ifeval_strict", [{"id": "a", "prediction": "   "}], gold)["score"] == 0
    # Conjunctive: one satisfied instruction is not enough.
    both = [item(["change_case:english_capital", "punctuation:no_comma"], [{}, {}])]
    assert score_predictions("ifeval_strict", [{"id": "a", "prediction": "THIS IS ENGLISH, WITH A COMMA IN IT"}], both)["score"] == 0
    assert score_predictions("ifeval_strict", [{"id": "a", "prediction": "THIS IS ENGLISH TEXT WITHOUT ANY COMMA IN IT"}], both)["score"] == 1


def test_ifeval_strict_rejects_malformed_instruction_metadata():
    gold = [{"id": "a", "prompt": "p", "metadata": {"instructionIdList": ["punctuation:no_comma"], "instructionKwargs": []}}]
    with pytest.raises(ValueError, match="instructionIdList"):
        score_predictions("ifeval_strict", [{"id": "a", "prediction": "x"}], gold)


def test_ifeval_upstream_random_default_is_repeatable_and_preserves_rng():
    import random
    from hero_evaluation.scoring import ifeval_strict
    metadata = {"instructionIdList": ["keywords:letter_frequency"], "instructionKwargs": [{"letter": "#", "let_frequency": 4, "let_relation": "at least"}]}
    random.seed(123)
    before = random.getstate()
    first = ifeval_strict("Use four hashtags.", "yyyy", metadata)
    assert random.getstate() == before
    random.seed(456)
    assert ifeval_strict("Use four hashtags.", "yyyy", metadata) == first
