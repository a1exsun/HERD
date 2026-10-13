import pytest

from run import require_valid_resume


def test_after_stage_three_requires_verified_resume_and_preserves_evidence(tmp_path):
    work = tmp_path / "training.tmp"
    require_valid_resume(work, 4, False)  # A new stage is permitted.
    work.mkdir()
    evidence = work / "started.json"
    evidence.write_text("started")
    with pytest.raises(RuntimeError, match="no retraining"):
        require_valid_resume(work, 4, False)
    assert evidence.read_text() == "started"
    require_valid_resume(work, 4, True)
    require_valid_resume(work, 3, False)
    checkpoint = work / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "progress.json").write_text("invalid")
    with pytest.raises(RuntimeError, match="failed verification"):
        require_valid_resume(work, 1, False)
    assert (checkpoint / "progress.json").read_text() == "invalid"
