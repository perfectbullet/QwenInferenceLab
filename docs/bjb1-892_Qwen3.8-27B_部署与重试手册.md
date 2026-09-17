# bjb1-892：Qwen3.8-27B NVFP4 部署与重试手册

目标机器：`root@connect.bjb1.seetacloud.com:54492`（简称 bjb1-892）
模型：`Inferact/Qwen3.8-27B-NVFP4`
目标模型路径：`/root/autodl-tmp/aigc/dep_qwen38/models/Inferact/Qwen3.8-27B-NVFP4`
建议 Conda 环境名：`qwen38-vllm`，环境路径：`/root/autodl-tmp/aigc/dep_qwen38/envs/qwen38-vllm`

## 0. 先决检查（每次安装或启动前）

在 bjb1-892 执行：

```bash
ssh -p 54492 root@connect.bjb1.seetacloud.com
nvidia-smi
ls -l /dev/nvidia*
df -h /root/autodl-tmp
```

只有在安装、CUDA 验证或启动 vLLM 前，才必须看到一张 RTX PRO 6000、约 96 GB 显存，以及 `/dev/nvidia0` 等设备节点。bjb1-892 为节省 GPU 计费，空闲时可不挂载 GPU；此时 `nvidia-smi` 无 GPU 或 `/dev/nvidia*` 不存在是预期状态，并不表示实例异常。模型同步无需 GPU；准备部署时再在云平台挂载 GPU，之后才继续安装或启动服务。

## 1. 从 231 后台同步模型

以下两份脚本必须位于 231 的同一目录，且可执行：

```bash
chmod +x scripts/sync_qwen38_nvfp4_to_bjb1_892.sh scripts/start_qwen38_nvfp4_sync_231.sh
./scripts/start_qwen38_nvfp4_sync_231.sh
tail -f ~/autodl-tmp/aigc/dep_qwen38/logs/qwen38_nvfp4_rsync.log
```

脚本使用 `rsync --partial --append-verify`；同步中断后，直接再次执行启动命令即可从已验证的数据继续，绝不会使用 `--delete`。

检查进度和完成性：

```bash
ssh -p 54492 root@connect.bjb1.seetacloud.com \
  'du -sh /root/autodl-tmp/aigc/dep_qwen38/models/Inferact/Qwen3.8-27B-NVFP4; find /root/autodl-tmp/aigc/dep_qwen38/models/Inferact/Qwen3.8-27B-NVFP4 -type f | wc -l'

# 在 231：dry-run 应只显示极少或没有待传文件，且退出码为 0
rsync -aHn --delete --itemize-changes \
  -e 'ssh -p 54492' \
  /nfs-data/metahuman_work/models/Inferact/Qwen3.8-27B-NVFP4/ \
  root@connect.bjb1.seetacloud.com:/root/autodl-tmp/aigc/dep_qwen38/models/Inferact/Qwen3.8-27B-NVFP4/
```

上面的 `--delete` **只用于 dry-run 检查**，不会执行删除；不要去掉 `-n`。

## 2. 创建环境与安装

GPU 挂载确认后，在 bjb1-892 执行。环境放在高速、实例关机后仍保留的数据盘；镜像保存时不会随镜像带走，应保留本手册和安装命令。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
mkdir -p /root/autodl-tmp/aigc/dep_qwen38/envs
conda create -y -p /root/autodl-tmp/aigc/dep_qwen38/envs/qwen38-vllm python=3.12 pip
conda activate /root/autodl-tmp/aigc/dep_qwen38/envs/qwen38-vllm

# vLLM recipe requires 0.17.0+; install the current CUDA wheel first.
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  pip install -U pip vllm 'transformers>=5.8.0'
```

若清华镜像没有所需的最新 `vllm` CUDA wheel，才使用下列官方 PyPI 回退命令，并记录该例外：

```bash
PIP_INDEX_URL=https://pypi.org/simple pip install -U vllm 'transformers>=5.8.0'
```

安装后验证：

```bash
python -c 'import torch, vllm, transformers; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(vllm.__version__, transformers.__version__)'
```

## 3. 启动 vLLM（单 RTX PRO 6000，TP=1）

该配置来自 vLLM 的 Qwen3.8-27B NVFP4 配方：262K 原生上下文、FP8 KV cache、reasoning、自动工具调用、内置 MTP 推测解码。首次建议在 `tmux` 中运行。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/aigc/dep_qwen38/envs/qwen38-vllm

vllm serve /root/autodl-tmp/aigc/dep_qwen38/models/Inferact/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen38-27b-nvfp4 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --host 0.0.0.0 --port 8000
```

先做简单健康检查：

```bash
curl -sS http://127.0.0.1:8000/v1/models
curl -sS http://127.0.0.1:8000/health
```

若首次加载 OOM，依次尝试：把 `--gpu-memory-utilization 0.95` 降到 `0.90`，再降低 `--max-num-seqs 16`；不要先减少 262144 的上下文，除非业务本来不需要它。若 MTP 初始化有版本兼容问题，临时删除整行 `--speculative-config`，先验证基础服务。

## 4. 常用故障恢复

```bash
# 同步：查看 / 重新启动（231）
tail -100 ~/autodl-tmp/aigc/dep_qwen38/logs/qwen38_nvfp4_rsync.log
./scripts/start_qwen38_nvfp4_sync_231.sh

# GPU、磁盘、进程（bjb1-892）
nvidia-smi
df -h /root/autodl-tmp
ps -ef | rg 'vllm|VLLM'

# 端口被占用时定位
ss -ltnp | rg ':8000'

# 查看 Python/vLLM 版本
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/aigc/dep_qwen38/envs/qwen38-vllm
python -c 'import torch, vllm, transformers; print(torch.__version__, vllm.__version__, transformers.__version__)'
```

## 参考

- [vLLM Qwen3.8-27B recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [Inferact NVFP4 模型页](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4)
