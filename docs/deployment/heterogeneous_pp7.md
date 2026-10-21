# Heterogeneous PP deployment dependencies

The seven-stage deployment uses independently built SM70, SM86, and SM89 native
kernels, plus architecture-specific FlashAttention libraries. Source changes
alone do not supply those binaries. Build the native dependencies for every
architecture used by the pipeline; do not copy a virtual environment or model
weights into Git. The optional gated norm extension has separate
[build instructions](pp7_fusion.md).

## PP decode batching and startup warmup

`--pp-async-queue-depth` defaults to the physical PP stage count, preserving the
previous behavior. The validated seven-stage deployment uses a depth of four;
all seven stages still execute the model. This setting supplies one shared bound
for scheduler eligibility, in-flight batches, and the delayed feedback ring.
Do not independently change those components. An explicit depth requires the
asynchronous multiprocessing executor and Model Runner V2, and must be between
two and the physical stage count.

Add these options to an existing, otherwise complete MTP serving configuration:

```bash
PP_OPTIONS=(
  --pipeline-parallel-size 7
  --pp-async-queue-depth 4
  --additional-config '{"mtp_tail_warmup":true}'
)
# Include "${PP_OPTIONS[@]}" in the existing vllm serve command.
```

`mtp_tail_warmup` is opt-in. It warms single-tail and two-tail decode shapes using
the actual allocated KV cache, with at most eight requests and two partial tails
per batch. All ranks use the same capacity-limited plan. The validated MTP4
deployment warmed 102 profiles and added about six minutes to startup. This
bounded coverage does not guarantee that arbitrary unseen shapes avoid JIT
compilation; three or more simultaneous partial tails are outside its coverage.

## TileLang current-device selection

The deployed TileLang copy selected CUDA ordinal zero in three architecture
queries. A worker assigned to another GPU could therefore JIT for the wrong
architecture on a mixed Volta/Ampere/Ada server. The deployment applied
[`tilelang-current-device.patch`](patches/tilelang-current-device.patch) to the
dependency: the target query and both compute-version fallbacks now use
`torch.cuda.current_device()`. vLLM must set its worker device before these calls.

This patch is for the dependency version that still has the shown ordinal-zero
code. Check the installed source before applying it. Newer TileLang versions may
already implement the fix; do not apply it twice. The original files used for
the validated patch had these SHA256 values:

| File | Original SHA256 |
| --- | --- |
| `tilelang/utils/target.py` | `db2ccfe9caadcde93db40b1c222beed693f6555081c8c773fdf7d99532ac6b75` |
| `tilelang/contrib/nvcc.py` | `f11bc0b40ee29b9c0c1c339c1e7525ad576e3539bb48d3d2a49a102ad0574dee` |

Use an isolated environment for the deployment and review the dry run first:

```bash
REPO_ROOT="$PWD"
TILELANG_SITE="$(.venv/bin/python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("tilelang").origin).resolve().parent.parent)')"
patch --dry-run -p0 -d "$TILELANG_SITE" \
  < "$REPO_ROOT/docs/deployment/patches/tilelang-current-device.patch"
patch -p0 -d "$TILELANG_SITE" \
  < "$REPO_ROOT/docs/deployment/patches/tilelang-current-device.patch"
```

This does not require importing TileLang or creating a GPU context. Keep the
original dependency files available if the environment needs to be restored.
The patch is a deployment prerequisite, not an automatic package-install hook.

## MTP and vision checkpoints

Pipeline-parallel MTP uses a standalone draft on the final stage. Its checkpoint
must contain `model.embed_tokens.weight` in addition to the draft/expert weights;
the final stage cannot share the first stage's target embedding. The loader now
rejects a missing private embedding before inference. This repository does not
include checkpoints or conversion outputs.

Multimodal Qwen4Exp GGUF requires the matching dense F16/F32/BF16 mmproj. The
adapter validates every expected tensor and reconstructs the temporal patch
convolution from its two GGUF slices. The first pipeline stage owns the vision
tower and must contain all deepstack injection layers. Downstream and CPU PLE
offload workers skip the tower.

The CPU regressions exercise the actual checked-out methods. Use a checkout
with its normal native dependencies built; these checks need no GPU context.
Each command keeps collection inside its focused conftest boundary:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  tests/models/qwen4_exp/test_mtp_embedding_guard.py \
  tests/models/qwen4_exp/test_mm_pipeline_stage.py \
  --confcutdir=tests/models/qwen4_exp -q

CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  tests/v1/worker/test_mtp_tail_warmup.py \
  --confcutdir=tests/v1/worker -q
```
