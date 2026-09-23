from qwen_inference_lab.evaluator.gold_adapter import build_gold_profile

def test_multi_part_routes_to_direct_judge():
    profile = build_gold_profile({"id": "Q", "question": "(1) 求值 (2) 证明", "reference_answer": "答案"})
    assert profile.status.value == "multi_part"
    assert profile.evaluation_route.value == "llm_multi_part"

def test_single_expression_routes_to_semantic_judge():
    profile = build_gold_profile({"id": "Q", "question": "求下列表达式的值", "reference_answer": "1"})
    assert profile.status.value == "resolved"
    assert profile.evaluation_route.value == "llm_semantic"
    assert profile.reason_code == "SINGLE_EXPRESSION_LLM_SEMANTIC"
