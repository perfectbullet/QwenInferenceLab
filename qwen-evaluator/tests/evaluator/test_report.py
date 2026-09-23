import json
from qwen_inference_lab.evaluator.report import build_audit_manifest

class Repository:
    def get_question(self, question_id): return {"question": "q", "reference_answer": "r", "math_type": "algebra", "difficulty": "easy", "tag": "tag"}
    def get_run(self, run_id): return {"answer": "a", "status": "completed"}

def test_manifest_leaves_human_fields_empty(tmp_path):
    output = tmp_path / "manifest.json"
    evaluation = {"runId": "R1", "questionId": "Q1", "runtime": {"status": "completed"}, "math": {"level": 1, "verdict": "correct"}, "extraction": {}, "judge": None}
    assert build_audit_manifest(Repository(), [evaluation], output, 50) == 1
    sample = json.loads(output.read_text())["samples"][0]
    assert sample["humanVerdict"] is None
    assert sample["humanNotes"] is None
