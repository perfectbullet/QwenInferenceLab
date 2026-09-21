# RTX PRO 6000 Blackwell vLLM 启动排障记录（归档）

日期：2026-09-18
目标：在云服务器上启动 `Qwen3.8-27B-NVFP4` 的 vLLM OpenAI 兼容服务。

## 最终状态

- 服务已成功监听 `0.0.0.0:8000`。
- 服务模型名为 `qwen38-27b-nvfp4`。
- `/health` 返回 HTTP 200，`/v1/models` 可返回模型信息。
- 已调用 `/v1/chat/completions` 验证：问题“Calculate 17 times 23”返回 `391`；响应同时包含 `content` 与 `reasoning` 字段。
- 启动日志：`/root/autodl-tmp/aigc/dep_qwen38/vllm-qwen38.log`。
- 运行进程：启动时 API Server PID 为 `11328`（进程号会在重启后变化）。

## 服务器与模型环境

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU 计算能力 | SM120（12.0） |
| 驱动显示的最高 CUDA | 13.2 |
| PyTorch | 2.13.0+cu130 |
| vLLM | 0.29.0 |
| FlashInfer | 0.6.18 |
| 模型目录 | `/root/autodl-tmp/aigc/dep_qwen38/models/Inferact/Qwen3.8-27B-NVFP4` |
| Python 环境 | `/root/autodl-tmp/aigc/dep_qwen38/envs/qwen38-vllm` |

## 排查问题与处理

### 1. 系统默认 CUDA 编译器过旧

**现象**

FlashInfer 初始化 NVFP4 内核时报错：

```text
RuntimeError: No supported CUDA architectures found for major versions [12].
```

服务器默认的 `/usr/local/cuda` 指向 CUDA 11.8。该版本无法为 Blackwell 的 SM120 编译 NVFP4 内核。

**处理**

不使用系统 CUDA，而是将 `CUDA_HOME` 指到 Python 环境内的 CUDA 工具链：

```bash
CUDA_ROOT=/root/autodl-tmp/aigc/dep_qwen38/envs/qwen38-vllm/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
```

### 2. CUDA 编译器与头文件版本不一致

**现象**

改用环境内 CUDA 13.4 编译器后，FlashInfer 编译时报：

```text
CUDA compiler and CUDA toolkit headers are incompatible
```

原因是环境内运行时头文件仍为 CUDA 13.0。

**处理**

通过清华 PyPI 镜像将 CUDA Toolkit、nvcc、crt、runtime 统一为 13.4：

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple pip install --upgrade --force-reinstall \
  'cuda-toolkit==13.4.2' \
  'nvidia-cuda-nvcc==13.4.92' \
  'nvidia-cuda-crt==13.4.92' \
  'nvidia-cuda-runtime==13.4.92'
```

### 3. CUDA 13.0 无法汇编 NVFP4 所需的 PTX

**现象**

使用 CUDA 13.0 时，编译后报：

```text
ptxas fatal: Unsupported .version 9.4; current version is '9.0'
```

**处理**

保留统一后的 CUDA 13.4 工具链。它支持 SM120 目标和 PTX 9.4。

### 4. pip 版 CUDA 的库目录布局与 FlashInfer 预期不同

**现象**

NVFP4 内核对象文件均已完成编译，但链接时报：

```text
/usr/bin/ld: cannot find -lcudart
```

pip 版 CUDA 的动态库位于 `lib`，FlashInfer 默认查找 `lib64`；同时目录中只有 `libcudart.so.13`，没有供链接器使用的 `libcudart.so`。

**处理**

补充兼容链接，并让运行时加载该目录：

```bash
ln -sfn lib "$CUDA_ROOT/lib64"
ln -sfn libcudart.so.13 "$CUDA_ROOT/lib/libcudart.so"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"
```

### 5. `OMP_NUM_THREADS` 被设置为空值

**现象**

启动环境出现：

```text
libgomp: Invalid value for environment variable OMP_NUM_THREADS
```

**处理**

启动前清除该空变量：

```bash
unset OMP_NUM_THREADS
```

这只是 CPU OpenMP 线程数环境变量的配置问题，不是模型启动失败的根因。

## 当前有效启动方式

服务器上优先使用项目提供的脚本：

```bash
chmod +x scripts/qwen38_nvfp4_vllm_bjb1.sh
./scripts/qwen38_nvfp4_vllm_bjb1.sh start
./scripts/qwen38_nvfp4_vllm_bjb1.sh status
./scripts/qwen38_nvfp4_vllm_bjb1.sh logs
```

脚本会设置必需的 CUDA 环境变量、补齐 pip CUDA 的库链接，并将日志和 PID 分别写入 `logs/vllm-qwen38-nvfp4.log` 与 `logs/vllm-qwen38-nvfp4.pid`。

如需手动启动，等价命令如下：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/aigc/dep_qwen38/envs/qwen38-vllm

CUDA_ROOT=/root/autodl-tmp/aigc/dep_qwen38/envs/qwen38-vllm/lib/python3.12/site-packages/nvidia/cu13
unset OMP_NUM_THREADS
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"

vllm serve /root/autodl-tmp/aigc/dep_qwen38/models/Inferact/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen38-27b-nvfp4 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --mm-encoder-tp-mode data \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --host 0.0.0.0 \
  --port 8000
```

首次启动会编译 FlashInfer 的 SM120 NVFP4 内核、执行自动调优和 CUDA Graph 捕获，因此耗时明显更长。生成的缓存位于 `/root/.cache/flashinfer` 与 `/root/.cache/vllm`；保留缓存后，后续重启会快很多。

## 验证命令

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

简单推理测试：

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "qwen38-27b-nvfp4",
    "messages": [{"role": "user", "content": "Calculate 17 times 23. Give only the concise answer."}],
    "temperature": 0,
    "max_tokens": 128
  }'
```

## 仍需注意的事项

1. `torch 2.13.0+cu130` 的 pip 元数据固定依赖 CUDA Toolkit 13.0；为支持本机的 SM120/NVFP4 内核，实际使用了 CUDA Toolkit 13.4。因此 `pip` 可能提示元数据依赖冲突。
2. 当前 vLLM 实测能加载、推理和返回 reasoning 字段；但未来升级 PyTorch、vLLM 或 FlashInfer 时，应整体评估兼容性，不能只单独升级一个 CUDA 子包。
3. 当前 `--gpu-memory-utilization 0.95` 下，日志测得可用 KV Cache 约 60 GiB；262,144 token 上下文的理论最大并发约 6.46。实际生产并发应按请求长度、图像输入和稳定性留出余量。
4. 模型的 `generation_config.json` 会覆盖 vLLM 默认采样参数（日志显示 `temperature=1.0`、`top_k=20`、`top_p=0.95`）。如需统一由服务端控制采样默认值，可追加 `--generation-config vllm`，但这会改变当前模型默认行为，应先做效果测试。
