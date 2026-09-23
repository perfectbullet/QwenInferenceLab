import pytest
from qwen_inference_lab.evaluator import math_verify_adapter
from qwen_inference_lab.evaluator.math_verify_adapter import evaluate_single_answer

def test_fraction_equals_decimal():
    assert evaluate_single_answer(r"\boxed{\frac{1}{2}}", r"\boxed{0.5}").verdict == "correct"

@pytest.mark.parametrize(("gold", "prediction"), [
    (r"\boxed{(x+1)^2}", r"\boxed{x^2+2x+1}"),
    (r"\boxed{\{1,2\}}", r"\boxed{\{2,1\}}"),
    (r"\boxed{[0,1]}", r"\boxed{[0,1]}"),
    (r"\boxed{x=2}", r"\boxed{x=2}"),
    (r"\boxed{x>1}", r"\boxed{x>1}"),
])
def test_math_equivalence_cases(gold, prediction):
    assert evaluate_single_answer(gold, prediction).verdict == "correct"

def test_mismatch_is_incorrect():
    assert evaluate_single_answer(r"\boxed{2}", r"\boxed{3}").verdict == "incorrect"

def test_unparseable_prediction_is_unresolved():
    assert evaluate_single_answer(r"\boxed{2}", "there is no mathematical final answer here").verdict == "unresolved"

def test_verify_argument_direction(monkeypatch):
    parsed = iter([["GOLD"], ["PREDICTION"]])
    monkeypatch.setattr(math_verify_adapter, "parse", lambda *args, **kwargs: next(parsed))
    def fake_verify(gold, prediction, **kwargs):
        assert gold == ["GOLD"]
        assert prediction == ["PREDICTION"]
        return True
    monkeypatch.setattr(math_verify_adapter, "verify", fake_verify)
    assert evaluate_single_answer("gold", "prediction").verdict == "correct"
