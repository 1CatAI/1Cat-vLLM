# 1Cat-vLLM-0.0.2

> 一猫之下始终相信，V100 不该在今天的大模型浪潮里被轻易宣判“过时”。
> `1Cat-vLLM-0.0.2` 不是一次简单的适配更新，而是一次面向
> **SM70 / Tesla V100** 的系统性工程重构。我们围绕 AWQ、注意力后端、
> 长上下文稳定性、运行时默认值和部署路径做了成体系的打磨，极大提升了
> V100 的模型使用上限，让更多原本“难以跑起来、难以跑稳定、难以跑得快”
> 的现代模型场景，真正变得可用、好用、能持续部署。
>
> 在我们聚焦和验证过的 V100 场景里，这个版本不仅显著抬升了上下文能力与
> 部署稳定性，也带来了业界领先的推理速度表现。对还在使用 V100 的个人开发者、
> 工作室和团队来说，这意味着老卡依然有很强的生命力，依然值得被继续挖掘。
> 我们真心希望 V100 开源社区越来越好，也希望把一猫之下自己的工程经验、
> 优化成果和热情，实实在在地贡献给社区。感谢每一位关注、使用、反馈和支持
> 一猫之下的朋友。你们的支持，是我们继续把这件事做深、做久、做好的动力。

`1Cat-vLLM-0.0.2` is a formal `0.0.2` release of the **Tesla V100 / SM70** vLLM fork for
**AWQ 4-bit inference on Volta GPUs**.

Upstream vLLM AWQ kernels normally require **SM75+** in the default path.
This branch integrates **lmdeploy TurboMind SM70 WMMA kernels** and a set of
SM70-specific runtime fixes so that V100 can serve modern AWQ models,
especially **Qwen3.5 dense and MoE models**.

Compared with the earlier `0.0.1` line, `0.0.2` is a major practical upgrade:
the install path is simpler, the public runtime defaults are more stable, and
the validated V100 coverage is much broader for real Qwen3.5 deployments.

## Recommended model providers

- `tclf90/Qwen3.5-27B-AWQ`
- `tclf90/Qwen3.5-35B-A3B-AWQ`
- `tclf90/Qwen3.5-122B-A10B-AWQ` for larger 4-GPU setups

The launch commands below use short model names such as
`Qwen3.5-27B-AWQ` and `Qwen3.5-35B-A3B-AWQ`.

This assumes one of the following is true:

- you have local model directories with exactly these names
- you replace `--model` with your real local path
- you replace `--model` with the full Hugging Face repo id

## What this branch adds

- AWQ 4-bit support for **SM70 / Tesla V100**
- Dense and MoE AWQ execution paths on V100
- Reuse of SM70 AWQ kernels for selected compressed-tensors MoE paths
- SM70-specific Triton attention and MLA/GDN runtime fixes
- Compatibility with `torch.compile` and CUDA graphs
- OpenAI-compatible API serving through standard vLLM entrypoints

## What is new in 0.0.2

- A substantial release step forward over `0.0.1` in deployability,
  long-context stability, and tested V100 serving coverage
- A **prebuilt wheel first** installation path for `Python 3.12 + CUDA 12.8`
- Public runtime defaults now center on:
  - `--attention-backend TRITON_ATTN`
  - `--compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}'`
- V100 `16 GB` reference configs for:
  - dual-card systems
  - 4-card systems
  - `Qwen3.5-27B-AWQ`
  - `Qwen3.5-35B-A3B-AWQ`
- A documented **max-context TP2 text-only profile** with
  `max_model_len=262144`
- Experimental `fp8_e5m2` KV cache support is available on V100, but it is
  not the default public recommendation

## Reference hardware platforms

`0.0.2` was validated for the product directions shown below.
This release is intended to be a much stronger public drop for both the
dual-card and 4-card V100 all-in-one platforms.

| 4-card V100 all-in-one workstation | Dual-card V100 all-in-one workstation |
| --- | --- |
| [![1CatAI-QX620A 4-card V100 all-in-one workstation](docs/微信图片_20260321113511_19_9.png)](docs/微信图片_20260321113511_19_9.png) | [![DX521A dual-card V100 all-in-one workstation](docs/微信图片_20260321113510_18_9.png)](docs/微信图片_20260321113510_18_9.png) |

- `1CatAI-QX620A`: 4-card V100 water-cooled workstation
- `DX521A`: dual-card V100 all-in-one workstation

## Benchmarks / Effort figures

For a compact set of benchmark figures and charts, see:

- [`OPEN_SOURCE_SM70_GUIDE.md`](OPEN_SOURCE_SM70_GUIDE.md)
- [`effort/README.md`](effort/README.md)

### Qwen3.5 local test charts

The repository also includes local throughput charts under
[`测试结果/`](测试结果), covering both **prefill** and **decode** speed
against prompt length on V100.

These figures are useful as reference data for this branch, but they should be
read together with the runtime notes below:

- first-request warmup on V100 is slow and is not representative
- long-context throughput depends strongly on `TP`, `max_num_seqs`, and the
  attention backend
- the public runtime defaults in this README prioritize stable serving over
  peak single-case benchmark numbers

| Qwen3.5-27B-AWQ TP2 | Qwen3.5-27B-AWQ TP4 |
| --- | --- |
| [![Qwen3.5-27B-AWQ TP2](测试结果/Qwen3.5-27B-AWQ/tp2/Tesla_V100-16G_x4,1Cat-vLLM-0.0.2,Qwen3.5-27B-AWQ,20260321_103111.png)](测试结果/Qwen3.5-27B-AWQ/tp2/Tesla_V100-16G_x4,1Cat-vLLM-0.0.2,Qwen3.5-27B-AWQ,20260321_103111.png) | [![Qwen3.5-27B-AWQ TP4](测试结果/Qwen3.5-27B-AWQ/tp4/Tesla_V100-16G_x4,1Cat-vLLM-0.0.2,Qwen3.5-27B-AWQ,20260321_104628.png)](测试结果/Qwen3.5-27B-AWQ/tp4/Tesla_V100-16G_x4,1Cat-vLLM-0.0.2,Qwen3.5-27B-AWQ,20260321_104628.png) |

| Qwen3.5-35B-A3B-AWQ TP2 | Qwen3.5-35B-A3B-AWQ TP4 |
| --- | --- |
| [![Qwen3.5-35B-A3B-AWQ TP2](测试结果/Qwen3.5-35B-A3B-AWQ/tp2/Tesla_V100-16G_x2,1Cat-vLLM-0.0.2,Qwen3.5-35B-A3B-AWQ,20260321_110213.png)](测试结果/Qwen3.5-35B-A3B-AWQ/tp2/Tesla_V100-16G_x2,1Cat-vLLM-0.0.2,Qwen3.5-35B-A3B-AWQ,20260321_110213.png) | [![Qwen3.5-35B-A3B-AWQ TP4](测试结果/Qwen3.5-35B-A3B-AWQ/tp4/Tesla_V100-16G_x4,1Cat-vLLM-0.0.2,Qwen3.5-35B-A3B-AWQ,20260321_105548.png)](测试结果/Qwen3.5-35B-A3B-AWQ/tp4/Tesla_V100-16G_x4,1Cat-vLLM-0.0.2,Qwen3.5-35B-A3B-AWQ,20260321_105548.png) |

At a glance:

- `Qwen3.5-27B-AWQ TP2`: decode is about `56.94 tok/s` at `1024` tokens and
  `39.85 tok/s` at `31744` tokens
- `Qwen3.5-27B-AWQ TP4`: decode is about `85.11 tok/s` at `1024` tokens and
  `64.37 tok/s` at `31744` tokens
- `Qwen3.5-35B-A3B-AWQ TP2`: decode is about `103.31 tok/s` at `1024` tokens
  and `83.96 tok/s` at `31744` tokens
- `Qwen3.5-35B-A3B-AWQ TP4`: decode is about `121.85 tok/s` at `1024` tokens
  and `95.70 tok/s` at `31744` tokens

## 微信交流群

请扫描下方二维码加入微信群组：

![1Cat-vllm 开源交流群1](docs/微信图片_20260321115313_20_9.png)

## Validated stack

The commands in this README were validated on the following setup:

- OS: `Ubuntu 24.04.4 LTS`
- Python: `3.12.13`
- CUDA toolkit: `12.8`
- PyTorch: `2.9.1+cu128`
- Triton: `3.5.1`
- Driver: `570.211.01`
- GPU: `Tesla V100-SXM2-16GB`

Both dual-card and 4-card V100 setups were used during validation.

## Runtime notes you should read first

- The **first real request is not representative** of steady-state speed.
  On V100, the first request may spend **1 to 3 minutes** compiling kernels,
  building graphs, and warming up execution paths.
- On dual `16 GB` V100, turning on vision leaves very limited context
  headroom. For Qwen3.5 on smaller cards, the public recommendation is
  text-only serving.
- If total VRAM is `64 GB` or above, multimodal use is much less constrained
  and you can choose your own context and vision balance more freely.
- For Qwen3.5 text-only serving on smaller cards, the recommended public
  defaults are:
  - `--skip-mm-profiling`
  - `--limit-mm-per-prompt '{"image":0,"video":0}'`
  - `--attention-backend TRITON_ATTN`
  - `--compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}'`
- `fp8_e4m3` KV cache is **not supported** on V100 in the current Triton path.
- `fp8_e5m2` KV cache can be used experimentally, but **do not** combine it
  with `--calculate-kv-scales`.

## Quick start

### 1. Install CUDA 12.8

Use the official NVIDIA repository on Ubuntu 24.04:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8
```

If the machine also has CUDA 13.x installed, force build-time and runtime CUDA
to 12.8:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
hash -r
nvcc -V
```

### 2. Create the conda environment

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda create -y -n 1Cat-vLLM-0.0.2 python=3.12
conda activate 1Cat-vLLM-0.0.2

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Recommended install path: prebuilt wheel

Use the release wheel if you only want to run the project.

Install from a local wheel file:

```bash
python -m pip install ./dist-cu128-sm70/vllm-*.whl
```

Or install from a GitHub release asset:

```bash
python -m pip install "https://github.com/1CatAI/1Cat-vLLM/releases/download/v0.0.2/vllm-0.0.3.dev0+g55573923a.d20260321.cu128-cp312-cp312-linux_x86_64.whl"
```

Notes:

- This is the **recommended** installation path for public users.
- The wheel already contains the compiled extension, so runtime installation
  from the wheel does not require the `lmdeploy` source tree.
- Use `Python 3.12` and `CUDA 12.8`.

### 4. Verify the environment

```bash
python - <<'PY'
import torch, triton, vllm, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("triton", triton.__version__)
print("vllm", vllm.__version__)
PY
```

## Docker deployment

Docker deployment is now supported for this fork. The recommended container
path is a runtime image built from the released SM70 wheel, not a full
source-build image.

### 1. Build the recommended SM70 runtime image

```bash
docker build \
  -f docker/Dockerfile.sm70-wheel \
  -t 1cat-vllm-sm70:0.0.2 \
  .
```

The first Docker build will download several gigabytes of PyTorch and CUDA
runtime layers. The build context for this repository is already trimmed, but
the Docker image store still lives under the host Docker root directory unless
you have moved it yourself.

This Dockerfile intentionally uses `python:3.12-slim-trixie`. The current
SM70 wheel needs `glibc >= 2.38`, and the runtime image also keeps `gcc/g++`
installed because Triton compiles a small helper module on first startup.

This image is pinned to:

- `Python 3.12`
- `Debian trixie / glibc 2.41`
- `torch 2.9.1`
- `torchvision 0.24.1`
- `torchaudio 2.9.1`
- `gcc/g++` for Triton first-run compilation
- the current `v0.0.2` release wheel

The image entrypoint already includes these public defaults:

- `--skip-mm-profiling`
- `--limit-mm-per-prompt '{"image":0,"video":0}'`
- `--attention-backend TRITON_ATTN`
- `--compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}'`

If you want runtime caches to stay on a large disk, add these options to the
`docker run` commands below:

- `-v /path/to/1t-cache/hf:/cache/hf -e HF_HOME=/cache/hf`
- `-v /path/to/1t-cache/triton:/cache/triton -e TRITON_CACHE_DIR=/cache/triton`
- `-v /path/to/1t-cache/torchinductor:/cache/torchinductor -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor`
- `-v /path/to/1t-cache/tmp:/cache/tmp -e TMPDIR=/cache/tmp`

This runtime path was validated on dual `16 GB` V100 with
`Qwen3.5-35B-A3B-AWQ`: `/health` returned successfully and the first
`2+2` request returned `4`.

### 2. Run on dual `16 GB` V100 with `Qwen3.5-27B-AWQ`

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

### 3. Run on dual `16 GB` V100 with `Qwen3.5-35B-A3B-AWQ`

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

### 4. Run on four `16 GB` V100 with `Qwen3.5-35B-A3B-AWQ`

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

### 5. Quick API check

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

### 6. Container source build

Container source build is still available through the upstream-style
multi-stage [docker/Dockerfile](/media/test/软件/vllm-0.0.2/vllm/docker/Dockerfile), but
it is not the recommended first path for public users.

For this fork, the recommended public Docker path is still the released wheel
image above.

## Source build

Source build is still supported, but it is **not** the recommended first
install path for public users.

Only use it if:

- you want to modify CUDA or Triton code
- you want to rebuild your own wheel
- you are doing development on this fork

### 1. Bundled `lmdeploy` source dependency

This repository already includes the validated `lmdeploy` source tree needed
for the SM70 AWQ build path.

```bash
cd /path/to/vllm
test -d lmdeploy
```

### 2. Install build dependencies

```bash
cd /path/to/vllm
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate 1Cat-vLLM-0.0.2

python -m pip install -r requirements/build.txt
python -m pip install -r requirements/cuda.txt
python -m pip install -r requirements/common.txt
python -m pip install cmake build
```

### 3. Build from source

The current validated `0.0.2` source build uses `CUDA 12.8`, `SM70`, and
`MAX_JOBS=20`.

```bash
cd /path/to/vllm
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate 1Cat-vLLM-0.0.2

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=20
export NVCC_THREADS=1

rm -rf build vllm.egg-info
rm -rf .deps/*-build .deps/*-subbuild

python -m build --wheel --no-isolation --outdir dist-cu128-sm70
```

If you want an editable source install instead of a wheel build:

```bash
python -m pip install -e . --no-build-isolation
```

## Public runtime defaults for V100 16 GB reference systems

These are the public `0.0.2` reference configs we recommend writing into
deployment docs.

| Host | Model | TP | `max_model_len` | `max_num_seqs` | `max_num_batched_tokens` | Use case |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dual `16 GB` V100 | `Qwen3.5-27B-AWQ` | 2 | `262144` | `4` | `2048` | public default, text-only, single request plus small short-request concurrency |
| dual `16 GB` V100 | `Qwen3.5-35B-A3B-AWQ` | 2 | `262144` | `2` | `2048` | public default, text-only, more conservative on memory |
| 4-card `16 GB` V100 | `Qwen3.5-27B-AWQ` | 4 | `65536` | `2` | `2048` | higher throughput reference profile |
| 4-card `16 GB` V100 | `Qwen3.5-35B-A3B-AWQ` | 4 | `65536` | `1` | `2048` | better fit for 35B on 4 cards |

Important wording:

- `262144` is the **max-context declaration** validated for the dual-card
  text-only profile.
- This means **startup plus first request was validated**.
- It does **not** mean full `256K` prompt stress testing at higher concurrency.
- The 4-card profiles here intentionally lean toward serving throughput.
- If you plan to feed very long prompts, reduce `--max-num-seqs` to `1`.

## Launch examples

All commands below are written as full runnable commands.

Run them from the repository root or from any environment where `vllm` has
already been installed.

### Qwen3.5-27B-AWQ, TP2, public dual-16G default

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

If your actual workload is a very long prompt rather than small short-request
concurrency, use the same command but change:

```text
--max-num-seqs 1
```

### Qwen3.5-35B-A3B-AWQ, TP2, public dual-16G default

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

Again, if your main workload is a truly long prompt, reduce:

```text
--max-num-seqs 1
```

### Qwen3.5-27B-AWQ, TP4, 4-card V100 reference

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

### Qwen3.5-35B-A3B-AWQ, TP4, 4-card V100 reference

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

## OpenAI-compatible request example

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer EMPTY' \
  -d '{
    "model": "Qwen3.5-27B-AWQ",
    "messages": [{"role": "user", "content": "用一句话回答，2+2等于几？"}],
    "temperature": 0,
    "max_completion_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

If the first request returns `2+2 等于 4。`, the service is basically healthy.

## Optional experimental feature: FP8 KV cache

This is not the default public recommendation, but it is worth documenting.

- `fp8_e4m3` is not usable on V100 in the current Triton path
- `fp8_e5m2` can be used experimentally
- do **not** add `--calculate-kv-scales`

Example:

```bash
--kv-cache-dtype fp8_e5m2
```

## Known limits

- This branch is optimized for **SM70 / Tesla V100**, not for all hardware.
- The public `262144` profile is a **compatibility and startup-validated**
  profile, not a proof of full `256K` stress stability at higher concurrency.
- On dual `16 GB` V100, multimodal and vision workloads reduce context
  headroom quickly, so this README defaults to text-only serving.
- Once total VRAM reaches `64 GB` or above, multimodal deployment is much less
  constrained and you can tune the context budget more freely.
- If you want guaranteed headroom for very long prompts, lower
  `--max-num-seqs` before increasing any other knob.

## Repository notes

- The upstream project is **vLLM**
- This fork focuses on **SM70 AWQ support and V100-oriented runtime tuning**
- The public `0.0.2` README prioritizes:
  - prebuilt wheel installation
  - short model names in commands
  - full runnable `python -m vllm.entrypoints.openai.api_server` commands

## Acknowledgements

- [vLLM](https://github.com/vllm-project/vllm)
- [lmdeploy / TurboMind](https://github.com/InternLM/lmdeploy)

## License

This repository follows the upstream vLLM license model. See [LICENSE](LICENSE).
