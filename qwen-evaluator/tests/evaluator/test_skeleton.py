from qwen_inference_lab.evaluator import PIPELINE_VERSION


def test_pipeline_version_is_stable() -> None:
    assert PIPELINE_VERSION == "evaluator-v1"
