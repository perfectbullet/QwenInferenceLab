from qwen_inference_lab.evaluator.gold_adapter import build_gold_profile

def test_multi_part_routes_to_direct_judge():
    profile = build_gold_profile({"id": "Q", "question": "(1) 求值 (2) 证明", "reference_answer": "答案"})
    assert profile.status.value == "multi_part"
    assert profile.evaluation_route.value == "llm_multi_part"
