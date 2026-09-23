import json

from qwen_inference_lab.capability.report import write_csv, write_jsonl

from .test_builder import inputs
from qwen_inference_lab.capability.builder import build_capability_records


def test_exports_exclude_leakage_fields(tmp_path):
    records = build_capability_records(*inputs(["correct", "incorrect", "correct"]))
    jsonl = tmp_path / "dataset.jsonl"
    csv = tmp_path / "dataset.csv"
    write_jsonl(records, jsonl)
    write_csv(records, csv)
    payload = json.loads(jsonl.read_text(encoding="utf-8").strip())
    assert payload["question"] == "question"
    assert "reference_answer" not in payload
    assert "answer" not in payload
    assert "reasoning" not in payload
    assert csv.exists()
