#!/usr/bin/env python3
"""最小 OpenAI 兼容接口流式对话测试（仅依赖 Python 标准库）。"""

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://192.168.100.202:8015/v1").rstrip("/")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
MODEL = os.environ.get("MODEL")
PROMPT = os.environ.get("PROMPT", "用一句话介绍你自己。")


def request_json(url: str, payload: dict | None = None):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    with urlopen(Request(url, data=data, headers=headers), timeout=60) as response:
        return response if payload is not None else json.load(response)


def main() -> int:
    global MODEL
    if not MODEL:
        models = request_json(f"{BASE_URL}/models")
        available = models.get("data", [])
        if not available:
            raise RuntimeError(f"/models 未返回可用模型：{models}")
        MODEL = available[0]["id"]

    print(f"模型：{MODEL}")
    print(f"提问：{PROMPT}")
    print("回答：", end="", flush=True)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "temperature": 0.2,
    }
    response = request_json(f"{BASE_URL}/chat/completions", payload)
    with response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            event = line[6:]
            if event == "[DONE]":
                break
            try:
                delta = json.loads(event)["choices"][0].get("delta", {})
            except (json.JSONDecodeError, IndexError, KeyError):
                continue
            # 兼容普通文本与部分推理模型的 reasoning_content 字段。
            text = delta.get("content") or delta.get("reasoning_content") or ""
            print(text, end="", flush=True)
    print()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HTTPError, URLError, TimeoutError, RuntimeError) as error:
        print(f"\n请求失败：{error}", file=sys.stderr)
        raise SystemExit(1)
