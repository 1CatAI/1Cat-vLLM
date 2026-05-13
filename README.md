# 1Cat-vLLM 1.0.0

> 一猫之下始终相信，V100 不该在今天的大模型浪潮里被轻易宣判“过时”。
> `1Cat-vLLM 1.0.0` 不是一次简单的适配更新，而是一次面向
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

`1Cat-vLLM 1.0.0` is the recommended public release of the
**Tesla V100 / SM70** vLLM fork for
**AWQ 4-bit inference on Volta GPUs,and FlashAttn-2!!**.

Upstream vLLM AWQ kernels normally require **SM75+** in the default path.
This branch integrates **lmdeploy TurboMind SM70 WMMA kernels**,
**FLASH_ATTN_V100**, and a set of SM70-specific runtime fixes so that V100 can
serve modern AWQ models, especially **Qwen3.5 / Qwen3.6 dense and MoE models**.

Compared with the earlier `0.0.x` line, `1.0.0` focuses on the new V100
attention backend, Qwen3.5/Qwen3.6 model coverage, FP8 KV cache support,
MTP serving, output-quality stability fixes, and a cleaner public wheel
installation path. The validated default path now centers on the prebuilt
`v1.0.0` wheels and `FLASH_ATTN_V100` instead of source builds or the older
Triton attention fallback.

## Recommended model providers

- `tclf90/Qwen3.6-27B-AWQ`
- `tclf90/Qwen3.6-35B-A3B-AWQ`
- `tclf90/Qwen3.5-122B-A10B-AWQ` for larger 4-GPU setups

The launch commands below use short model names such as
`Qwen3.5-27B-AWQ` and `Qwen3.6-35B-A3B-AWQ`.

This assumes one of the following is true:

- you have local model directories with exactly these names
- you replace `--model` with your real local path
- you replace `--model` with the full Hugging Face repo id

## What this branch adds

- AWQ 4-bit support for **SM70 / Tesla V100**
- Dense and MoE AWQ execution paths on V100
- Reuse of SM70 AWQ kernels for selected compressed-tensors MoE paths
- `FLASH_ATTN_V100` decode and prefill backend for Volta GPUs
- Qwen3.5 / Qwen3.6 model and config support, including MoE and MTP paths
- SM70-specific MLA/GDN runtime fixes
- Compatibility with `torch.compile` and CUDA graphs
- OpenAI-compatible API serving through standard vLLM entrypoints

## What is new in 1.0.0

- A release step forward over `0.0.3` for **V100-flash-attention**, Qwen3.5/Qwen3.6
  coverage, public packaging, and output-quality stability
- A **two-wheel** installation path for `Python 3.12 + CUDA 12.8`
  (`flash_attn_v100` plus `vllm`)
- FP8 KV cache support for the V100 FA path, with `fp8_e5m2` documented as the
  current experimental V100 option
- MTP speculative decoding support for Qwen3.6-class models
- Tool-calling and OpenAI-compatible API fixes for Cherry Studio, OpenClaw, and
  similar OpenAI API clients
- DFlash is included as an experimental path for continued validation
- Public runtime defaults now center on:
  - `--attention-backend FLASH_ATTN_V100`
  - `--max-model-len 262144`
  - explicit low-concurrency serving limits such as `--max-num-seqs` and
    `--max-num-batched-tokens`
- V100 `32 GB` reference configs for 4-card systems:
  - `Qwen3.5-27B-AWQ`
  - `Qwen3.6-35B-A3B-AWQ`
  - `Qwen3.5-122B-A10B-AWQ`
- Long-prompt chunk budget for `FLASH_ATTN_V100` on 32 GB V100 defaults to
  `max_num_batched_tokens=16384`
- Direct paged prefill remains experimental and is not the public default

## Reference hardware platforms

`1.0.0` is validated primarily on 4-card V100 systems. The recommended public
commands below assume **4 x V100 32 GB** and text-generation workloads.

| Public reference host | Notes |
| --- | --- |
| 4 x Tesla PG503 / V100 32 GB | Recommended target for Qwen3.5/Qwen3.6 AWQ serving |

- `Qwen3.5-27B-AWQ`: supported on TP1/TP2/TP4, with TP4 recommended for this README
- `Qwen3.6-35B-A3B-AWQ`: TP4 recommended for the public command
- `Qwen3.5-122B-A10B-AWQ`: TP4 only in the public command

## Benchmarks / Effort figures

The following local `1.0.0` regression charts were generated on a 4-card V100
32 GB system. First-request warmup is not included as steady-state throughput.

### Local test charts

| `Qwen3.5-27B-AWQ` | `Qwen3.6-35B-A3B-AWQ` | `Qwen3.5-122B-A10B-AWQ` |
| --- | --- | --- |
| [![Qwen3.5-27B-AWQ](docs/test-table/Tesla_PG503-32G_x4,1Cat-vLLM-0.0.3,Qwen3.5-27B-AWQ,20260501_001139.png)](docs/test-table/Tesla_PG503-32G_x4,1Cat-vLLM-0.0.3,Qwen3.5-27B-AWQ,20260501_001139.png) | [![Qwen3.6-35B-A3B-AWQ](docs/test-table/Tesla_PG503-32G_x4,1Cat-vLLM-0.0.3,Qwen3.6-35B-A3B-AWQ,20260501_002046.png)](docs/test-table/Tesla_PG503-32G_x4,1Cat-vLLM-0.0.3,Qwen3.6-35B-A3B-AWQ,20260501_002046.png) | [![Qwen3.5-122B-A10B-AWQ](docs/test-table/Tesla_PG503-32G_x4,1Cat-vLLM-0.0.3,Qwen3.5-122B-A10B-AWQ,20260501_004543.png)](docs/test-table/Tesla_PG503-32G_x4,1Cat-vLLM-0.0.3,Qwen3.5-122B-A10B-AWQ,20260501_004543.png) |

- first-request warmup on V100 is slow and is not representative
- long-context throughput depends strongly on `TP`, `max_num_seqs`, and the
  attention backend
- the public runtime defaults in this README prioritize stable serving over
  peak single-case benchmark numbers

### Reproducible 27B decode baseline

The `1.0.0` 27B speed baseline is measured as **incremental decode TPS**:

```text
incremental_decode_tps =
  (decode64_output_tokens - decode1_output_tokens) /
  (decode64_median_latency - decode1_median_latency)
```

This removes prefill/TTFT from the measurement. It is stricter than API
streaming throughput and should not be compared directly with browser-side
OpenAI streaming numbers.

Reference result on `4 x Tesla PG503 / V100 32 GB`:

| Model | Backend | TP | Custom all-reduce | Short-context incremental decode | 8K-context incremental decode |
| --- | --- | ---: | --- | ---: | ---: |
| `Qwen3.5-27B-AWQ` | `FLASH_ATTN_V100` | 4 | enabled | `86.31 tok/s` | `79.04 tok/s` |

Strict reproduction command. This speed-only harness intentionally keeps
`max_model_len=12288` to match the historical model-side regression test. The
public serving commands below default to 256K context with
`max_model_len=262144`.

```bash
export ONECAT_VLLM_REPO=/path/to/1Cat-vLLM/vllm
cd /tmp

CUDA_VISIBLE_DEVICES=0,1,2,3 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python "$ONECAT_VLLM_REPO/tools/vllm_v100_backend_regression.py" \
  --child \
  --backend FLASH_ATTN_V100 \
  --model /path/to/Qwen3.5-27B-AWQ \
  --dtype float16 \
  --kv-cache-dtype auto \
  --max-model-len 12288 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.88 \
  --tensor-parallel-size 4 \
  --prompt-style qwen35-chat \
  --disable-thinking \
  --disable-mm \
  --quality-max-tokens 1 \
  --long-prompt-tokens 8202 \
  --speed-warmup 3 \
  --speed-iters 5 \
  --skip-quality \
  --child-output /tmp/qwen35_27b_fa2_baseline.json
```

Expected key latencies:

- `batch1_prefill512_decode1`: about `0.179 s`
- `batch1_prefill512_decode64`: about `0.909 s`
- incremental decode: about `86 tok/s`

Do **not** add `--disable-custom-all-reduce` for the 27B TP4 baseline. On the
same hardware this drops short-context incremental decode from about
`86.31 tok/s` to about `75.91 tok/s`.

## 微信交流群

**群聊：** 1Cat-vLLM 开源交流群2

请使用微信扫描下方二维码加入群组：

![1Cat-vLLM 微信交流群二维码](docs/assets/wechat-group-qr.png)

> 提示：微信群二维码通常 7 天内有效。若扫描失败或提示过期，请重新打开本页查看最新图片，或关注仓库更新。

## Validated stack

The commands in this README were validated on the following setup:

- OS: `Ubuntu 24.04.4 LTS`
- Python: `3.12.13`
- CUDA toolkit: `12.8`
- PyTorch: `2.9.1+cu128`
- Triton: `3.5.1`
- Driver: `570.211.01`
- GPU: `4 x Tesla V100 32 GB` public reference profile

The public launch commands below are written for 4-card V100 32 GB systems.

## Runtime notes you should read first

- The **first real request is not representative** of steady-state speed.
  On V100, the first request may spend **1 to 3 minutes** compiling kernels,
  building graphs, and warming up execution paths.
- The public commands in this README are text-generation profiles. Vision or
  multimodal workloads should be tuned separately.
- For Qwen3.5/Qwen3.6 text-only serving on V100 32 GB, the recommended public
  commands explicitly set only the serving choices that change behavior:
  - `--attention-backend FLASH_ATTN_V100`
  - `--max-model-len 262144`
  - `--max-num-seqs` and `--max-num-batched-tokens`
  - `--enable-prefix-caching` for the MTP + prefix-cache profile
- `--gpu-memory-utilization` is an upper bound for the model executor. By
  default, 1Cat-vLLM trims the final KV cache allocation to about
  `1.05 * max_model_len * max_num_seqs`, so single-request 256K serving does
  not preallocate KV capacity for many extra full-length requests. Set
  `--kv-cache-auto-trim-ratio 0` to keep upstream vLLM's "use all requested
  memory for KV cache" behavior, or use `--kv-cache-memory-bytes` for an exact
  per-GPU KV cache size.
- `VLLM_SM70_ENABLE_DENSE_F16_FASTPATH=1` is experimental. Keep it disabled for
  the public 35B/122B MoE commands.
- Direct paged prefill can be forced with `VLLM_FLASH_V100_ENABLE_PAGED_PREFILL=1`,
  but it is not the quality-safe default.

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
conda create -y -n 1Cat-vLLM-1.0.0 python=3.12
conda activate 1Cat-vLLM-1.0.0

python -m pip install --upgrade pip setuptools wheel
```

### 3. Recommended install path: prebuilt wheel

Use the release wheel if you only want to run the project. This is the
recommended installation path. Source builds are for kernel development and are
not recommended for normal deployment.

The wheel install pulls the matching `torch==2.9.1+cu128` runtime from the
PyTorch CUDA 12.8 index. `--no-cache-dir` is recommended because the CUDA
runtime wheels are large.

Install from a local wheel file:

```bash
python -m pip install --prefer-binary --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  ./dist-cu128-sm70-1.0.0/flash_attn_v100-*.whl \
  ./dist-cu128-sm70-1.0.0/vllm-*.whl
```

Or install from a GitHub release asset:

```bash
python -m pip install --prefer-binary --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  "https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.0.0/flash_attn_v100-1.0.0-cp312-cp312-linux_x86_64.whl" \
  "https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.0.0/vllm-1.0.0-cp312-cp312-linux_x86_64.whl"
```

Notes:

- This is the **recommended first installation path** for public users.
- `flash_attn_v100` is a separate wheel and should be installed together with
  the vLLM wheel.
- Runtime installation from the wheels does not require the `lmdeploy` source
  tree.
- Use `Python 3.12` and `CUDA 12.8`.
- If your shell has a broken local proxy configured, unset it before installing:
  `env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy ...`.
- After installing from wheels, run `python -m vllm...` from a directory outside
  this source checkout, such as `cd ~` or `cd /tmp`. Running inside the cloned
  repository makes Python import the local `vllm/` source tree, which does not
  contain the wheel-installed CUDA extension files such as `vllm/_C.abi3.so`.

### 4. Verify the environment

```bash
python - <<'PY'
import torch, triton, vllm, sys
import flash_attn_v100_cuda, paged_kv_utils
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("triton", triton.__version__)
print("vllm", vllm.__version__)
print("flash_attn_v100", "ok")
PY
```

## Docker deployment

Docker deployment follows the same wheel-first approach. This release
does not include a dedicated `1.0.0` wheel-runtime Dockerfile yet, so use the
conda wheel path above for final local validation.

### 1. Build the recommended SM70 runtime image

```bash
# No dedicated 1.0.0 wheel-runtime Dockerfile is included in this tree yet.
# Use the conda wheel install path above, or adapt docker/Dockerfile for source build.
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
- the current `v1.0.0` release wheel

The runtime entrypoint should include these public defaults:

- `FLASH_ATTN_V100` as the V100 attention backend
- `--max-model-len 262144`
- explicit `max_num_seqs` and `max_num_batched_tokens` limits for the target
  model

If you want runtime caches to stay on a large disk, add these options to the
`docker run` commands below:

- `-v /path/to/1t-cache/hf:/cache/hf -e HF_HOME=/cache/hf`
- `-v /path/to/1t-cache/triton:/cache/triton -e TRITON_CACHE_DIR=/cache/triton`
- `-v /path/to/1t-cache/torchinductor:/cache/torchinductor -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor`
- `-v /path/to/1t-cache/tmp:/cache/tmp -e TMPDIR=/cache/tmp`

Final Docker validation data will be added after the wheel-runtime image is
rebuilt for `1.0.0`.

### 2. Run on four `32 GB` V100 with `Qwen3.5-27B-AWQ`

```bash
docker run --rm \
  --gpus '"device=0,1,2,3"' \
  --ipc=host \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100 \
  -e VLLM_MODEL=/models/Qwen3.5-27B-AWQ \
  -e VLLM_SERVED_MODEL_NAME=Qwen3.5-27B-AWQ \
  -e VLLM_TENSOR_PARALLEL_SIZE=4 \
  -e VLLM_GPU_MEMORY_UTILIZATION=0.88 \
  -e VLLM_MAX_MODEL_LEN=262144 \
  -e VLLM_MAX_NUM_SEQS=1 \
  -e VLLM_MAX_NUM_BATCHED_TOKENS=16384 \
  1cat-vllm-sm70:1.0.0
```

### 3. Run on four `32 GB` V100 with `Qwen3.6-35B-A3B-AWQ`

```bash
docker run --rm \
  --gpus '"device=0,1,2,3"' \
  --ipc=host \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100 \
  -e VLLM_MODEL=/models/Qwen3.6-35B-A3B-AWQ \
  -e VLLM_SERVED_MODEL_NAME=Qwen3.6-35B-A3B-AWQ \
  -e VLLM_TENSOR_PARALLEL_SIZE=4 \
  -e VLLM_GPU_MEMORY_UTILIZATION=0.88 \
  -e VLLM_MAX_MODEL_LEN=262144 \
  -e VLLM_MAX_NUM_SEQS=1 \
  -e VLLM_MAX_NUM_BATCHED_TOKENS=8192 \
  1cat-vllm-sm70:1.0.0
```

### 4. Run on four `32 GB` V100 with `Qwen3.5-122B-A10B-AWQ`

```bash
docker run --rm \
  --gpus '"device=0,1,2,3"' \
  --ipc=host \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100 \
  -e VLLM_MODEL=/models/Qwen3.5-122B-A10B-AWQ \
  -e VLLM_SERVED_MODEL_NAME=Qwen3.5-122B-A10B-AWQ \
  -e VLLM_TENSOR_PARALLEL_SIZE=4 \
  -e VLLM_GPU_MEMORY_UTILIZATION=0.88 \
  -e VLLM_MAX_MODEL_LEN=262144 \
  -e VLLM_MAX_NUM_SEQS=1 \
  -e VLLM_MAX_NUM_BATCHED_TOKENS=8096 \
  1cat-vllm-sm70:1.0.0
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
multi-stage [`docker/Dockerfile`](docker/Dockerfile), but it is not the
recommended first path for public users.

For this fork, the recommended public Docker path is still the released wheel
image above.

## Source build

Source build is still supported, but it is **not recommended** for public
runtime deployment. Install the release wheels first unless you are changing
CUDA/C++/Triton code.

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
conda activate 1Cat-vLLM-1.0.0

python -m pip install -r requirements/build.txt
python -m pip install -r requirements/cuda.txt
python -m pip install -r requirements/common.txt
python -m pip install cmake build
```

### 3. Build from source

The current validated `1.0.0` source build uses `CUDA 12.8`, `SM70`, and
`MAX_JOBS=12`.

```bash
cd /path/to/vllm
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate 1Cat-vLLM-1.0.0

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=12
export NVCC_THREADS=1

rm -rf build vllm.egg-info
rm -rf .deps/*-build .deps/*-subbuild

pushd flash-attention-v100
python -m build --wheel --no-isolation --outdir ../dist-cu128-sm70-1.0.0
popd

export VLLM_VERSION_OVERRIDE="1.0.0"
python -m build --wheel --no-isolation --outdir dist-cu128-sm70-1.0.0
```

If you want an editable source install instead of a wheel build:

```bash
python -m pip install -e . --no-build-isolation
```

## Public runtime defaults for V100 32 GB reference systems

These are the public `1.0.0` reference configs we recommend writing into
deployment docs.

| Host | Model | TP | `max_model_len` | `max_num_seqs` | `max_num_batched_tokens` | Use case |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 4-card `32 GB` V100 | `Qwen3.5-27B-AWQ` | 4 | `262144` | `1` | `16384` | stable public default |
| 4-card `32 GB` V100 | `Qwen3.6-27B-AWQ` + MTP | 4 | `262144` | `4` | `8192` | MTP + prefix-cache API serving |
| 4-card `32 GB` V100 | `Qwen3.6-35B-A3B-AWQ` | 4 | `262144` | `1` | `8192` | stable public default for MoE |
| 4-card `32 GB` V100 | `Qwen3.5-122B-A10B-AWQ` | 4 | `262144` | `1` | `8096` | long-context large-model default |

Important wording:

- `FLASH_ATTN_V100` is the recommended attention backend for V100 in `1.0.0`.
- Public baseline launch commands in this README default to 256K context
  (`max_model_len=262144`). If you publish or compare a new baseline, add its
  exact launch command to this README.
- Keep `max_num_seqs=1` for the baseline public commands until your workload
  has been profiled locally. The MTP + prefix-cache profile intentionally uses
  `max_num_seqs=4`.
- On 32 GB V100 with `FLASH_ATTN_V100`, the baseline API
  server default is also capped at `max_num_seqs=1` to avoid upstream's
  high-concurrency default preallocating unnecessary KV cache and
  sampler/CUDAGraph buffers.
- Do not pass `--disable-custom-all-reduce` for the 27B TP4 decode baseline.
- `122B` uses a small prefill chunk budget to leave room for SM70 MoE
  temporary workspace during long-context serving.
- `VLLM_SM70_ENABLE_DENSE_F16_FASTPATH=1` is not recommended for the 35B/122B
  MoE public commands.

## Launch examples

All commands below are written as full runnable commands. When using the
prebuilt wheels, run them outside the source checkout, for example after
`cd ~`, so Python loads the installed wheel package and its CUDA extensions.

The commands assume the `1Cat-vLLM` wheel is already installed in your active
Python environment. Use `CUDA_VISIBLE_DEVICES=0,1,2,3` only when you need to
select a specific four-card V100 set.

### Qwen3.5-27B-AWQ, TP4, public 4-card default

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.5-27B-AWQ \
  --served-model-name Qwen3.5-27B-AWQ \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 16384 \
  --host 0.0.0.0 \
  --port 8000
```

### Qwen3.6-27B-AWQ, TP4, MTP + prefix cache, public 4-card default

This is the recommended MTP serving profile for the `Qwen3.6-27B-AWQ` model on
4 x V100 32 GB. It keeps the public 256K context default, enables prefix cache
for repeated prompts, and uses `num_speculative_tokens=4`, which was the stable
public MTP profile used for Cherry Studio/OpenClaw-style API testing.

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.6-27B-AWQ \
  --served-model-name qwen3.6-27b-awq-mtp \
  --trust-remote-code \
  --dtype float16 \
  --quantization awq \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-auto-trim-ratio 0.0 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --skip-mm-profiling \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1,2,4,8,9,18]}' \
  --host 0.0.0.0 \
  --port 8000
```

Do not remove these MTP-profile options unless you are deliberately changing
runtime behavior:

- `--kv-cache-auto-trim-ratio 0.0` disables 1Cat-vLLM's default KV auto-trim
  (`1.05`) and leaves more room for retained prefix-cache blocks.
- `--mamba-cache-mode align` is required for the Qwen3.6 hybrid Mamba path when
  prefix caching and MTP are used together.
- `--mm-processor-cache-gb 0`, `--skip-mm-profiling`, and
  `--limit-mm-per-prompt '{"image":0,"video":0}'` keep this public profile
  text-only and avoid unnecessary multimodal processor cache/profiling.
- `--compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1,2,4,8,9,18]}'`
  captures the MTP decode shapes used by this TP4 profile. Omitting it can
  reduce CUDA graph hit rate and noticeably lower decode throughput.

Do not set `VLLM_SM70_ENABLE_DENSE_F16_FASTPATH=1` for this public MTP profile.
That dense fast path is experimental and should be benchmarked separately from
the stable serving command.

For speed-only experiments without prefix cache or tool calling, use
`--max-num-seqs 1`, remove `--enable-prefix-caching`,
`--enable-auto-tool-choice`, and `--tool-call-parser`, and benchmark
`num_speculative_tokens` in `{2,4,6,8}` locally.

### Qwen3.6-35B-A3B-AWQ, TP4, public 4-card default

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.6-35B-A3B-AWQ \
  --served-model-name Qwen3.6-35B-A3B-AWQ \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --host 0.0.0.0 \
  --port 8000
```

### Qwen3.5-122B-A10B-AWQ, TP4, long-context 4-card default

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.5-122B-A10B-AWQ \
  --served-model-name Qwen3.5-122B-A10B-AWQ \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8096 \
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
- Public launch commands default to 256K context with
  `max_model_len=262144`.
- The public 27B command keeps `max_num_batched_tokens=16384`.
- The public 35B and 122B commands use smaller prefill chunk budgets to leave
  room for MoE and long-context workspace.
- Multimodal and vision workloads are not the default public profile for this
  release.
- If you want guaranteed headroom for very long prompts, keep
  `--max-num-seqs 1` before increasing any other knob.

## Repository notes

- The upstream project is **vLLM**
- This fork focuses on **SM70 AWQ support and V100-oriented runtime tuning**
- The public `1.0.0` README prioritizes:
  - prebuilt wheel installation
  - short model names in commands
  - `FLASH_ATTN_V100` as the recommended V100 attention backend
  - full runnable `python -m vllm.entrypoints.openai.api_server` commands

## Acknowledgements

- [vLLM](https://github.com/vllm-project/vllm)
- [lmdeploy / TurboMind](https://github.com/InternLM/lmdeploy)

## License

This repository follows the upstream vLLM license model. See [LICENSE](LICENSE).
