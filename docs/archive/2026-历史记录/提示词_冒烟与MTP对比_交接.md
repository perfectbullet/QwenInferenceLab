# 提示词：Qwen3.8-27B 数学评测冒烟 + MTP 对比（归档）

> **⚠️ 2026-08-20 下午状态更新（最新，覆盖此前所有状态）**：**服务当前已停止**（GPU 让给同事使用）。恢复命令：`ssh zenking@192.168.8.231 'docker start vllm-qwen38'`，启动约 4.5 分钟（权重加载 + graph capture），**启动前必须 `nvidia-smi` 确认空闲显存 ≥30 GiB**（他人负载在跑时起不来）。服务配置为 **MTP@3 + CUDA Graph + 20K 窗口**（`--max-model-len 20480 --max-num-seqs 1`，无 enforce-eager），客户端实测 67.2 tok/s（短题）/ 131.5 tok/s（长输出稳态）。评测脚本已升级（默认采样=官方 Thinking 参数、max-tokens=4096、effort=medium、新增 TTFT/tokens/truncated/final_answer_raw 字段），直接按下方命令跑即可。`comfyui-h3` 仍处暂停，评测结束后按需 `docker start comfyui-h3`。详见 `总结_部署与提速实验_20260819.md` 实验 5-7。

> 使用方式：将下方分割线以内的全部内容复制给另一个 workspace 的 agent 执行。

---

## 背景

231 服务器（192.168.8.231，单卡 RTX 5090 32GB）上已经启动了 Qwen3.8-27B 的 vLLM no-spec 基线服务（2026-08-19 已验证可用，数学冒烟答题正确）。业务目标：**单请求高考数学推理，Pass@1 不降的前提下缩短端到端解题时间**。

你现在要完成两个阶段：

1. **阶段一**：用本地题库脚本对现有 no-spec 服务做 COT / TIR 小样本冒烟（各 10 题）。
2. **阶段二**：在 231 上重启服务启用内置 MTP（num_speculative_tokens=3），做同样冒烟，并与 no-spec 初步对比。

北极星指标（按顺序）：
1. Pass@1 不低于 no-spec 基线（硬门槛）；
2. 答对题的端到端耗时（Time-to-correct）更短；
3. 最后才看 Decode TPS / TTFT / 显存。

**禁止**：仅凭 token/s 宣布更好；在同一题集上一边调参一边报告最终正确率。

## 环境与资产（先逐项验证再动手）

| 项目 | 值 |
|---|---|
| 推理服务 | `http://192.168.8.231:8000/v1`（容器名 `vllm-qwen38`，模型名 `qwen38-27b`，max_model_len=32768） |
| SSH | `ssh zenking@192.168.8.231`（本机已配免密；用户名是 zenking，不是 zj） |
| 模型路径 | `/data/metahuman_work/models/Inferact/Qwen3.8-27B-NVFP4`（含 MTP 权重 `nvfp4_experts_mtp.safetensors`） |
| vLLM 镜像 | `vllm/vllm-openai:qwen38`（不要换 latest 或其他 tag） |
| 题库 | `/home/zj/math_model_deployment/math_eval_pipeline/data/math_qa_275_20260617.mineru.filled_reference_answer-v2.jsonl`（275 题，只读，勿改写） |
| 推理脚本 | `/home/zj/math_model_deployment/math_eval_pipeline/tir_math/run_model_tir_inference_concurrent.py` |
| 参考文档 | `/home/zj/aigc/dep_qwen38-27B_231/` 下 7 份 md（行动计划、路线图、MTP 原理清单等） |

服务器是共享的：动手前 `nvidia-smi` 确认显存空闲（comfyui-h3 容器常驻，占约 0.9GB 属正常）。

## 阶段一：no-spec 冒烟（COT 与 TIR 各 10 题）

> 脚本已于 2026-08-20 升级：默认采样=官方 Thinking 参数（temperature=1.0/top_p=0.95/top_k=20）、默认 max-tokens=4096、默认 reasoning-effort=medium；COT 走直连流式（真实 TTFT/token/finish_reason/思考长度），输出新增 `final_answer_raw`（boxed 提取）、`truncated`、`has_boxed_answer`、`decode_tps` 等字段。命令里**不必再传 --max-tokens**（用默认 4096 即可）。

```bash
cd /home/zj/math_model_deployment/math_eval_pipeline/tir_math

# COT 模式（直连路径，effort=medium 生效）
python run_model_tir_inference_concurrent.py \
  --input ../data/math_qa_275_20260617.mineru.filled_reference_answer-v2.jsonl \
  --output ../data/model_outputs/mtp3-cot-smoke10-$(date +%m%d%H%M).jsonl \
  --api_base http://192.168.8.231:8000/v1 \
  --model qwen38-27b \
  --workers 1 \
  --mode cot \
  --lang zh \
  --limit 10

# TIR 模式（agent 路径；effort 无法传递，实际为模型默认 xhigh，metadata 有注明）
python run_model_tir_inference_concurrent.py \
  --input ../data/math_qa_275_20260617.mineru.filled_reference_answer-v2.jsonl \
  --output ../data/model_outputs/mtp3-tir-smoke10-$(date +%m%d%H%M).jsonl \
  --api_base http://192.168.8.231:8000/v1 \
  --model qwen38-27b \
  --workers 1 \
  --mode tir \
  --lang zh \
  --limit 10
```

注意：
- **每次运行必须用新的输出文件名**——脚本会跳过同一输出文件中已有成功 `id` 的题目，复用旧文件会"看似完成、实际没推理"。
- 输出目录 `../data/model_outputs/` 不存在则先创建。

**冒烟检查清单**（COT、TIR 分别记录）：
1. 10 题中非空 `model_output` 数；`metadata.error=empty_model_output` 计数。
2. `final_answer_raw` 非空率（脚本已自动提取 `\boxed{}`）；提取失败样例单独列出。
3. 截断率：`metadata.truncated=true` 计数（**已知现象**：多问综合大题如 MATH-247/248 存在思考反刍循环，任何预算都会撞顶——如实记录即可，这是模型真实失败模式，详见总结文档实验 7）。
4. 每题 `metadata.latency_seconds`（这是单题端到端主耗时口径；`total_elapsed_seconds` 含排队，**不得**用于比较速度）。
5. 服务端日志：`ssh zenking@192.168.8.231 'docker logs vllm-qwen38 2>&1 | tail -50'`，确认无 OOM、无 5xx、无重启。
6. 记录服务是否连续 10 题无崩溃。

产出：一张 COT vs TIR 的冒烟对比表（非空输出数、boxed 提取率、平均单题耗时、截断/异常数），存到 `/home/zj/aigc/dep_qwen38-27B_231/` 或随报告给出。

## 阶段二：启用内置 MTP 并做同样冒烟

冒烟通过后，在 231 上把服务切换为 MTP 配置：

```bash
ssh zenking@192.168.8.231

# 1) 先看显存空闲
nvidia-smi

# 2) 停掉 no-spec 容器（保留备查）
docker stop vllm-qwen38 && docker rename vllm-qwen38 vllm-qwen38-nospec

# 3) 启动 MTP 容器（唯一区别：多了 --speculative-config）
docker run -d --gpus all --name vllm-qwen38-mtp \
  -v /data/metahuman_work/models/Inferact/Qwen3.8-27B-NVFP4:/model \
  -p 8000:8000 \
  --restart unless-stopped \
  vllm/vllm-openai:qwen38 \
  --model /model \
  --served-model-name qwen38-27b \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

启动与验证：
- 权重从 NFS 加载约需 5 分钟；轮询 `curl http://192.168.8.231:8000/v1/models` 直到返回 `qwen38-27b`。
- **必须从启动日志确认 MTP 真的加载**（搜索 `docker logs vllm-qwen38-mtp 2>&1 | grep -Ei "mtp|speculat"`，应能看到 speculative 配置与接受率相关输出），而不是静默回退到 no-spec。日志若显示 speculative_config=None 或被忽略，停下来排查，不要继续跑题。
- 若启动即 OOM：唯一降级顺序是先 `--gpu-memory-utilization 0.92`，再降 `--max-model-len 20480`；一次只改一个参数并记录。

MTP 就绪后，重复阶段一的 COT 冒烟（10 题、新输出文件名，如 `mtp3-cot-smoke10-*.jsonl`；TIR 若时间允许可加跑）。

**对比口径（COT 对 COT，勿混模式）**：
```text
每题：metadata.latency_seconds、输出长度、boxed 提取结果
汇总：非空输出数、boxed 提取率、平均/中位单题耗时、截断数
服务端：docker logs 中的 MTP acceptance / accepted draft tokens / mean acceptance length（如有）
GPU：nvidia-smi 显存占用（MTP 会略高于 no-spec）
```

初步判定（10 题只是冒烟，不下最终结论）：
- 若 MTP 平均耗时明显下降且输出正常 → 报告"值得进入 75 题 calibration"；
- 若 MTP 更慢或输出异常 → 报告现象与日志证据，不要自行改 k 值重试。

## 采样与 thinking 纪律（两阶段一致）

- 模型默认采样是 `temperature=1.0, top_k=20, top_p=0.95`（来自 generation_config.json）。所有对比运行必须显式固定同一组采样参数；如果脚本不传采样参数，就在报告里注明"使用服务端默认"并保证两阶段一致。
- 冒烟中记录模型思考段：本 vLLM 版本的思考文本在 **`message.reasoning`** 字段（注意：不是 `reasoning_content`，读错字段会误判为空）。服务默认开启思考模式（chat template 的 `enable_thinking` 不传即 true，effort 默认 xhigh）。统计口径：`content` 提取 `\boxed{}`，`reasoning` 统计思考量。后续正式实验需按文档测 reasoning on/low/off，本阶段只记录不展开。

## 边界（不要越界）

- 不下载任何新模型/镜像；不动 Unsloth 权重；不升级 vLLM/SGLang。
- 不改 `num_speculative_tokens`（固定 3）；不引入 DFlash/DSpark。
- 不修改题库 JSONL；不改脚本逻辑（缺 TTFT/finish_reason 等字段属于已知的脚本改造待办，只记录不实现）。
- 231 上的其他容器（comfyui-h3 等）不要动。

## 交付物

1. COT vs TIR 冒烟对比表（no-spec）。
2. MTP@3 冒烟结果 + 与 no-spec 的同模式对比。
3. MTP 是否真实加载的日志证据。
4. 异常清单（空输出/截断/工具错误/OOM）与初步建议。
5. 所有输出 JSONL 的路径清单（供复算）。
