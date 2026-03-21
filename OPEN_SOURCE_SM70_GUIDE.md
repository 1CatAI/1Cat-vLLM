# 1Cat-vLLM-0.0.2 开源发布指南

这份文档面向直接使用本仓库发布产物的用户，优先给出预编译安装方式。

如果你只是想把服务跑起来，请先看本文件，不要先从源码编译开始。

## 适用范围

- 目标平台：Linux x86_64
- Python：`3.12`
- CUDA：`12.8`
- 目标 GPU 架构：`SM70`
- 已验证机型：`Tesla V100 16GB`
- 当前发布默认走 `TRITON_ATTN` 路径

## 推荐安装方式：预编译 wheel

### 1. 创建环境

```bash
conda create -n 1Cat-vLLM-0.0.2 python=3.12 -y
conda activate 1Cat-vLLM-0.0.2

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. 安装 wheel

先从 GitHub Releases 下载 wheel，再本地安装。

当前本地构建产物示例文件名：

```bash
vllm-0.0.3.dev0+g55573923a.d20260321.cu128-cp312-cp312-linux_x86_64.whl
```

本地文件安装：

```bash
python -m pip install ./vllm-0.0.3.dev0+g55573923a.d20260321.cu128-cp312-cp312-linux_x86_64.whl
```

如果你把 wheel 作为 GitHub Release 资产发布，也可以直接用 URL 安装：

```bash
python -m pip install "https://github.com/1CatAI/1Cat-vLLM/releases/download/v0.0.2/vllm-0.0.3.dev0+g55573923a.d20260321.cu128-cp312-cp312-linux_x86_64.whl"
```

说明：

- 当前 wheel 是 `cp312`，请使用 Python 3.12。
- 如果你的发布文件名变了，只替换 wheel 文件名即可。
- 推荐先装 `cu128` 的 PyTorch，再装 wheel，避免被错误拉取到 CPU 版或其他 CUDA 版本的 `torch`。

### 3. 快速确认安装成功

```bash
python - <<'PY'
import vllm
print(vllm.__version__)
PY
```

## Docker 部署：推荐走 wheel 运行时镜像

如果你要给别人分发一个可直接部署的容器，推荐用本仓库提供的
SM70 wheel-runtime 镜像入口，而不是在 Docker 里重新从源码编译。

### 1. 构建镜像

```bash
docker build \
  -f docker/Dockerfile.sm70-wheel \
  -t 1cat-vllm-sm70:0.0.2 \
  .
```

第一次 `docker build` 会下载数 GB 的 PyTorch / CUDA 运行时层。仓库的
Docker build context 已经尽量收紧，但 Docker 镜像层本身仍然会写到主机
当前的 Docker Root 目录，除非你已经自己迁移过 Docker 数据目录。

这个 Dockerfile 明确使用 `python:3.12-slim-trixie`。原因很直接：
当前 SM70 wheel 需要 `glibc >= 2.38`，并且运行时镜像里还要保留
`gcc/g++`，因为 Triton 首次启动时会现场编译一个很小的辅助模块。

这个镜像默认会安装当前 `v0.0.2` release wheel，并且自带以下公开默认项：

- `--skip-mm-profiling`
- `--limit-mm-per-prompt '{"image":0,"video":0}'`
- `--attention-backend TRITON_ATTN`
- `--compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}'`

如果你想把运行期缓存放到 1T 盘之类的大盘上，可以在下面的 `docker run`
命令后面额外补这些参数：

- `-v /path/to/1t-cache/hf:/cache/hf -e HF_HOME=/cache/hf`
- `-v /path/to/1t-cache/triton:/cache/triton -e TRITON_CACHE_DIR=/cache/triton`
- `-v /path/to/1t-cache/torchinductor:/cache/torchinductor -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor`
- `-v /path/to/1t-cache/tmp:/cache/tmp -e TMPDIR=/cache/tmp`

这条 Docker runtime 路径已经在双 `16G` V100 + `Qwen3.5-35B-A3B-AWQ`
上做过真实验证：`/health` 正常返回，首个 `2+2` 请求返回 `4`。

### 2. 双 `16G` V100 运行 `Qwen3.5-27B-AWQ`

```bash
docker run --rm \
  --gpus '"device=1,2"' \
  --ipc=host \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e VLLM_MODEL=/models/Qwen3.5-27B-AWQ \
  -e VLLM_SERVED_MODEL_NAME=Qwen3.5-27B-AWQ \
  -e VLLM_TENSOR_PARALLEL_SIZE=2 \
  -e VLLM_GPU_MEMORY_UTILIZATION=0.90 \
  -e VLLM_MAX_MODEL_LEN=262144 \
  -e VLLM_MAX_NUM_SEQS=4 \
  -e VLLM_MAX_NUM_BATCHED_TOKENS=2048 \
  1cat-vllm-sm70:0.0.2
```

### 3. 双 `16G` V100 运行 `Qwen3.5-35B-A3B-AWQ`

```bash
docker run --rm \
  --gpus '"device=1,2"' \
  --ipc=host \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e VLLM_MODEL=/models/Qwen3.5-35B-A3B-AWQ \
  -e VLLM_SERVED_MODEL_NAME=Qwen3.5-35B-A3B-AWQ \
  -e VLLM_TENSOR_PARALLEL_SIZE=2 \
  -e VLLM_GPU_MEMORY_UTILIZATION=0.90 \
  -e VLLM_MAX_MODEL_LEN=262144 \
  -e VLLM_MAX_NUM_SEQS=2 \
  -e VLLM_MAX_NUM_BATCHED_TOKENS=2048 \
  1cat-vllm-sm70:0.0.2
```

### 4. 四卡 `16G` V100 运行 `Qwen3.5-35B-A3B-AWQ`

```bash
docker run --rm \
  --gpus '"device=1,2,3,4"' \
  --ipc=host \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e VLLM_MODEL=/models/Qwen3.5-35B-A3B-AWQ \
  -e VLLM_SERVED_MODEL_NAME=Qwen3.5-35B-A3B-AWQ \
  -e VLLM_TENSOR_PARALLEL_SIZE=4 \
  -e VLLM_GPU_MEMORY_UTILIZATION=0.90 \
  -e VLLM_MAX_MODEL_LEN=65536 \
  -e VLLM_MAX_NUM_SEQS=1 \
  -e VLLM_MAX_NUM_BATCHED_TOKENS=2048 \
  1cat-vllm-sm70:0.0.2
```

### 5. 容器内快速验证

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.5-27B-AWQ",
    "messages": [{"role": "user", "content": "只回答最终结果：2+2等于几？"}],
    "temperature": 0,
    "max_completion_tokens": 16,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

## 源码编译方式：保留，但不推荐

只在下面几种情况才建议源码编译：

- 你要修改 CUDA / Triton / kernel 代码
- 你要重新打自己的 wheel
- 你要针对自己的机器重新做构建验证

不推荐源码编译给普通用户的原因很简单：

- 本仓库是 CUDA 12.8 + SM70 定向构建，直接装 wheel 成本最低
- 源码编译会受本机 `torch`、`cmake`、CUDA 路径、旧缓存和内存压力影响
- `MAX_JOBS` 过高时很容易把内存打爆

精简版源码编译命令如下：

```bash
conda create -n 1Cat-vLLM-0.0.2 python=3.12 -y
conda activate 1Cat-vLLM-0.0.2

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/cuda.txt
python -m pip install cmake build

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export VLLM_TARGET_DEVICE=cuda
export VLLM_MAIN_CUDA_VERSION=12.8
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS=20
export NVCC_THREADS=1

rm -rf build vllm.egg-info
rm -rf .deps/*-build .deps/*-subbuild

python -m build --wheel --no-isolation --outdir dist-cu128-sm70
```

## 运行前建议

当前开源发布的默认建议：

- Qwen3.5 系列优先走 `TRITON_ATTN`
- 双 `16G` V100 开视觉时上下文会明显变短，公开默认建议优先跑 text-only
- 总显存 `64 GB` 及以上时，多模态上下文压力会小很多，可以按自己的场景调整
- 需要长上下文时，先关闭视觉：

```bash
--limit-mm-per-prompt '{"image":0,"video":0}'
```

### 这两个参数建议当成公开文档默认值

下面所有完整命令都会直接带上这两项：

```bash
--attention-backend TRITON_ATTN
--compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}'
```

这只是开源文档里的推荐默认值，不是说仓库本身已经修改了 CLI 全局默认行为。

### 模型名写法

为了让命令更短，下面示例直接写：

- `Qwen3.5-27B-AWQ`
- `Qwen3.5-35B-A3B-AWQ`

实际使用时：

- 如果这是你的本地模型目录名，可以直接用
- 如果你模型不在当前目录，请替换成你自己的本地路径
- 如果你用 Hugging Face repo id，也可以直接替换成 repo id

通用的健康检查命令：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "Qwen3.5-27B-AWQ",
    "messages": [{"role": "user", "content": "用一句话回答，2+2等于几？"}],
    "temperature": 0,
    "max_completion_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

只要首个请求能返回 `2+2 等于 4。`，就说明服务基本正常。

## V100 16G 参考配置

### 结论先看

| 场景 | 推荐配置 | 口径 |
|---|---|---|
| 默认发布推荐 | `Qwen3.5-27B-AWQ TP=2` | 兼顾单并发和少量多并发 |
| 双 16G 跑 35B | `Qwen3.5-35B-A3B-AWQ TP=2` | 同样走 text-only，配置更保守 |
| 四卡 `16G` 跑 27B | `Qwen3.5-27B-AWQ TP=4` | 更适合做吞吐型部署 |
| 四卡 `16G` 跑 35B | `Qwen3.5-35B-A3B-AWQ TP=4` | 35B 的更优 4 卡口径 |
| 最大上下文口径 | `27B/35B TP=2 max_model_len=262144` | 仅验证“启动 + 首请求正常” |

### 推荐默认值

#### 统一发布默认值

如果你希望同一套配置同时兼顾：

- 单并发
- 少量多并发
- 尽量短的启动命令

建议把下面这组值写进开源文档：

| 主机 | 模型 | TP | `max_model_len` | `max_num_seqs` | `max_num_batched_tokens` | 说明 |
|---|---|---:|---:|---:|---:|---|
| 双 `16G` V100 | `Qwen3.5-27B-AWQ` | 2 | `262144` | `4` | `2048` | 双卡默认推荐 |
| 双 `16G` V100 | `Qwen3.5-35B-A3B-AWQ` | 2 | `262144` | `2` | `2048` | 更保守，适合双卡 |
| 四卡 `16G` V100 | `Qwen3.5-27B-AWQ` | 4 | `65536` | `2` | `2048` | 更适合吞吐部署 |
| 四卡 `16G` V100 | `Qwen3.5-35B-A3B-AWQ` | 4 | `65536` | `1` | `2048` | 35B 的更优 4 卡口径 |

说明：

- 这组值适合“统一对外文档”
- 它兼顾单并发和少量短请求并发
- 如果你真的要跑超长 prompt，仍建议临时把 `max_num_seqs` 降到 `1`
- `262144` 这个值是当前“最大上下文启动成功 + 首请求正确”的验证口径，不是 256K 满载压测口径

#### 1. Qwen3.5-27B-AWQ：双 16G 默认推荐

这是当前最适合写进开源文档的默认方案。

```bash
export CUDA_VISIBLE_DEVICES=0,1

python -m vllm.entrypoints.openai.api_server \
  --model Qwen3.5-27B-AWQ \
  --quantization awq \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --tensor-parallel-size 2 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 2048 \
  --skip-mm-profiling \
  --attention-backend TRITON_ATTN \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}' \
  --host 0.0.0.0 \
  --port 8000
```

发布建议：

- 双 16G 默认就写这一条
- 这是“兼顾单并发和少量多并发”的稳定值
- 如果你的用户主要跑超长上下文，再补一句：超长 prompt 时建议把 `--max-num-seqs` 调成 `1`

#### 2. Qwen3.5-35B-A3B-AWQ：双 16G 保守推荐

35B 在双 16G 上也能跑，但默认建议保守一些。

```bash
export CUDA_VISIBLE_DEVICES=0,1

python -m vllm.entrypoints.openai.api_server \
  --model Qwen3.5-35B-A3B-AWQ \
  --quantization awq \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --tensor-parallel-size 2 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 2048 \
  --skip-mm-profiling \
  --attention-backend TRITON_ATTN \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}' \
  --host 0.0.0.0 \
  --port 8000
```

发布建议：

- 如果开源文档必须覆盖 35B 双 16G，就写这条
- 35B 比 27B 更吃资源，所以默认把少量多并发收成 `max_num_seqs=2`
- 如果你主要跑极长上下文，还是建议临时降成 `max_num_seqs=1`

### 256K 上下文说明

下面这个结论可以写进文档，但必须写清口径：

- `Qwen3.5-27B-AWQ TP=2 text-only`：`max_model_len=262144` 可启动，且首个短请求返回正常
- `Qwen3.5-35B-A3B-AWQ TP=2 text-only`：`max_model_len=262144` 可启动，且首个短请求返回正常
- 这不是“256K 满 prompt 压测通过”
- 这是“服务能拉起 + 第一个请求正确返回”的验证口径

参考命令：

```bash
export CUDA_VISIBLE_DEVICES=0,1

python -m vllm.entrypoints.openai.api_server \
  --model Qwen3.5-27B-AWQ \
  --quantization awq \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --tensor-parallel-size 2 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --skip-mm-profiling \
  --attention-backend TRITON_ATTN \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}' \
  --host 0.0.0.0 \
  --port 8000
```

35B 只需要把模型名换掉即可。

## 模型运行参考命令

### 27B：单机双卡 text-only 推荐

见上面的 `Qwen3.5-27B-AWQ TP=2` 默认方案。

### 35B：单机双卡 text-only 推荐

见上面的 `Qwen3.5-35B-A3B-AWQ TP=2` 保守方案。

### 27B：如果你有 4 张卡，更推荐 TP=4

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m vllm.entrypoints.openai.api_server \
  --model Qwen3.5-27B-AWQ \
  --quantization awq \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 65536 \
  --tensor-parallel-size 4 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 2048 \
  --skip-mm-profiling \
  --attention-backend TRITON_ATTN \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}' \
  --host 0.0.0.0 \
  --port 8000
```

### 35B：如果你有 4 张卡，更推荐 TP=4

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m vllm.entrypoints.openai.api_server \
  --model Qwen3.5-35B-A3B-AWQ \
  --quantization awq \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 65536 \
  --tensor-parallel-size 4 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --skip-mm-profiling \
  --attention-backend TRITON_ATTN \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}' \
  --host 0.0.0.0 \
  --port 8000
```

## 可选实验项

### FP8 KV cache

当前建议写成“实验功能”，不要当默认配置：

- `fp8_e4m3`：V100 / SM70 上不推荐，当前不可用
- `fp8_e5m2`：可以启动并返回正确结果
- 但当前不要配 `--calculate-kv-scales`

实验命令示例：

```bash
--kv-cache-dtype fp8_e5m2
```

## 相关说明

- 本文件已经包含开源发布所需的安装、运行和双 16G 推荐配置。
- 目录中的开发阶段分析记录已清理，不再作为发布文档保留。
