from pathlib import Path

import pytest

from qwen_inference_lab.capability.builder import BatchSpec, build_capability_records


def inputs(verdicts):
    specs = [
        BatchSpec(Path(f"round{index}.json"), f"round{index}", 4 if index == 1 else 8, (f"R{index}",))
        for index in range(1, 4)
    ]
    question = {"id": "Q1", "question": "question", "tag": "tag", "math_type": "algebra", "difficulty": "medium"}
    runs = {
        f"R{index}": {"id": f"R{index}", "questionId": "Q1", "status": "completed"}
        for index in range(1, 4)
    }
    evaluations = {
        f"R{index}": {
            "runId": f"R{index}", "questionId": "Q1", "pipelineVersion": "evaluator-v1",
            "math": {"verdict": verdict, "level": 3, "method": "llm_semantic", "confidence": 0.9},
        }
        for index, verdict in enumerate(verdicts, 1)
    }
    return specs, {"Q1": question}, runs, evaluations


@pytest.mark.parametrize(
    ("verdicts", "correct", "incorrect", "pass_rate", "local_count"),
    [
        (["correct", "correct", "correct"], 3, 0, 1.0, 3),
        (["correct", "correct", "incorrect"], 2, 1, 0.6666667, 2),
        (["correct", "incorrect", "incorrect"], 1, 2, 0.3333333, 1),
        (["incorrect", "incorrect", "incorrect"], 0, 3, 0.0, 0),
    ],
)
def test_pass_rates(verdicts, correct, incorrect, pass_rate, local_count):
    record = build_capability_records(*inputs(verdicts))[0]
    assert record.math_correct == correct
    assert record.math_incorrect == incorrect
    assert record.math_pass_rate == pass_rate
    assert record.local_success_count == local_count
    assert record.local_success_rate == round(local_count / 3, 7)
    assert record.label_usable is True


def test_unknown_verdict_is_not_incorrect():
    record = build_capability_records(*inputs(["correct", "incorrect", "unresolved"]))[0]
    assert record.math_correct == 1
    assert record.math_incorrect == 1
    assert record.unresolved == 1
    assert record.gradable_attempts == 2
    assert record.math_pass_rate == 0.5
    assert record.label_usable is False


def test_duplicate_run_id_fails():
    specs, questions, runs, evaluations = inputs(["correct", "correct", "correct"])
    specs[1] = BatchSpec(Path("duplicate.json"), "round2", 8, ("R1",))
    with pytest.raises(ValueError, match="Duplicate runId"):
        build_capability_records(specs, questions, runs, evaluations)


def test_attempt_count_must_be_three():
    specs, questions, runs, evaluations = inputs(["correct", "correct", "correct"])
    with pytest.raises(ValueError, match="has 2 attempts"):
        build_capability_records(specs[:2], questions, runs, evaluations)


def test_missing_evaluation_fails():
    specs, questions, runs, evaluations = inputs(["correct", "correct", "correct"])
    evaluations.pop("R3")
    with pytest.raises(ValueError, match="Missing Evaluation"):
        build_capability_records(specs, questions, runs, evaluations)
