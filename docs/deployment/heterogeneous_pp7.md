# Heterogeneous PP deployment dependencies

The seven-stage deployment uses independently built SM70, SM86, and SM89 native
kernels, plus architecture-specific FlashAttention libraries. Source changes
alone do not supply those binaries. Build the native dependencies for every
architecture used by the pipeline; do not copy a virtual environment or model
weights into Git. The optional gated norm extension has separate
[build instructions](pp7_fusion.md).

## Isolated native build and startup verification

`tools/pp7_native.py` separates a complete build from an inventory of existing
binaries. It never installs into an existing destination, modifies loader source,
or patches the active Python environment. Keep the serving checkout and virtual
environment available throughout validation and rollback.

Use the deployment virtual environment's Python with its CUDA-enabled PyTorch.
The tool requires CMake, Ninja, GCC/G++, `patch`, `patchelf`, and the CUDA toolkit's
`nvcc` and `cuobjdump`. First apply and verify the TileLang patch described below in an
isolated environment. Dependency paths must identify prepared source trees:
the matching CUTLASS and Triton kernels, the patched Volta FA2 tree used by this
repository, and the pinned upstream RTX FA2 tree. No dependency is silently
downloaded or updated. A Git checkout records its commit, recursive submodules,
dirty diff hash and complete source tree hash; a copied tree records its content
hash and explicitly has no known commit.

Prepare a fresh Volta FA2 checkout using the pinned dependency and patch order
from `cmake/external_projects/vllm_flash_attn.cmake`. A source directory copied
from an older deployment may have omitted its CUTLASS submodule, even when that
deployment has working precompiled libraries. The build checks required headers
and patched sources before compiling.

```bash
# Run from the vLLM checkout. Both destinations must be new directories.
REPO_ROOT="$PWD"
FA2_VOLTA=/path/to/new-volta-fa2
FA2_RTX=/path/to/new-rtx-fa2
git clone --no-checkout https://github.com/zhinianqin/flash-attention-v100.git "$FA2_VOLTA"
git -C "$FA2_VOLTA" checkout --detach c2eda5e6115b98c3ba4bfd181570668742eece22
git -C "$FA2_VOLTA" submodule update --init --recursive
for name in pipeline splitkv3 k_pingpong gqa_arch; do
  patch --batch --forward -p1 -l -d "$FA2_VOLTA" \
    -i "$REPO_ROOT/cmake/patches/sm70_flash_attn_d256_${name}.patch"
done
git clone --no-checkout https://github.com/vllm-project/flash-attention.git "$FA2_RTX"
git -C "$FA2_RTX" checkout --detach bce29425653ec0fbc579d329883030e832d15ada
git -C "$FA2_RTX" submodule update --init --recursive
```

Use the matching CUTLASS and Triton source revisions specified by this checkout's
CMake dependency declarations. Their supplied content is snapshotted; the tool
does not substitute a different revision from the network during the build.
The Triton input is the complete repository, not its Python package subdirectory.
The build clears an inherited `TRITON_KERNELS_SRC_DIR` so that it cannot change
the snapshot's CMake vendoring layout.

```bash
# All output, dependency snapshots and build directories go to a new stage.
.venv/bin/python tools/pp7_native.py --cuda-home /usr/local/cuda-12.8 build \
  --stage /path/to/new-pp7-stage --jobs 4 \
  --cutlass /path/to/cutlass-source \
  --triton /path/to/triton-kernels-source \
  --fa2-sm70 /path/to/prepared-volta-fa2-source \
  --fa2-rtx /path/to/pinned-rtx-fa2-source

# Verify before launching workers; this command does not create a GPU context.
.venv/bin/python tools/pp7_native.py --cuda-home /usr/local/cuda-12.8 verify \
  --root /path/to/new-pp7-stage/bundle

# Materialize a separate deployment candidate. Existing destinations are refused.
.venv/bin/python tools/pp7_native.py --cuda-home /usr/local/cuda-12.8 install \
  --root /path/to/new-pp7-stage/bundle --destination /path/to/new-candidate
```

The build covers `_C`, `_C_gguf`, `_C_stable_libtorch`, `_moe_C`, Triton kernels,
the CMake SM70 companion modules, host allocator/spinloop modules, separate
Volta/RTX FA2 libraries, FlashAttention V100 and FlashQLA. The optional
gated norm extension remains a separate opt-in build. Build commands and source
identities are preserved; the manifest records runtime Python/Torch/CUDA/C++ ABI,
compiler versions, ELF architectures, binary hashes, and TileLang patch hashes.
Fresh builds materialize the complete `flash_attn_v100` package at the bundle
root, including both native extensions and package resources. Built manifests
require local V100 and FlashQLA SM70 extensions and record their import layout.
Verification resolves the candidate's package and extension origins without
importing their code or loading CUDA, and repeats this check after installation.
It cannot silently satisfy a built manifest with older packages in the virtual
environment. This checks the candidate with its own root first on the import
path; activate that same root in the actual serving command, for example
`PYTHONPATH=/path/to/new-candidate .venv/bin/python -m vllm.entrypoints.cli.main ...`.
Verification rejects changed source or binaries, incomplete architecture coverage,
runtime ABI mismatch and dependency patch drift. It does not establish numerical
correctness: run focused GPU tests and serving validation before switching services.
The process is reproducible from recorded inputs, not a promise of byte-identical
output across different toolchains. Full rebuild success must be reported separately
from inventory verification.

To audit a prepared existing checkout without rebuilding:

```bash
.venv/bin/python tools/pp7_native.py --cuda-home /usr/local/cuda-12.8 record \
  --root /path/to/prepared-candidate --external-json /path/to/native-externals.json
```

This writes an `observed` manifest and never claims to know the binary build
provenance. Its ABI fields describe the recording environment; only a build
manifest ties them to compiler inputs. It refuses to overwrite a manifest. Native
symlinks are checked by content; installation into a new directory materializes
their files. Inventory covers native libraries under `vllm`, `flash-attention-v100`,
`flash_attn_v100` and `flash_qla` in the specified root. Observed manifests can
describe existing external packages instead of requiring the fresh-build layout;
they do not certify that the bundle is self-contained. Explicit external
dependencies can be listed
in JSON as `[{"label":"Flash-V100","module":"flash_attn_v100","files":["/absolute/path/to/__init__.py","/absolute/path/to/extension.so"]}]`.
The optional `module` field checks the actual import origin and installed
distribution version; all listed files are hashed and ELF architectures recorded.
Include the selected FlashQLA prebuilt library or JIT cache if it is outside the
bundle. Only explicitly listed dependencies are covered. A JIT cache hash does
not guarantee a later launch will not recompile it; verify again after startup,
or bind the validated `FLASH_QLA_SM70_PREBUILT_EXTENSION_PATH` in the serving
configuration. The manifest is an integrity record
for trusted local artifacts, not a digital signature.
Run `verify` again with the actual serving environment before every activation.

The qualified RTX FA2 file is selected only for SM80, SM86 and SM89. Other
architectures retain their ordinary library, even when an RTX file is installed.
The filename alone is not a compatibility guarantee. This FA2 build emits SM80
cubins, which also run on SM86/SM89 under NVIDIA's
[same-major, higher-minor binary compatibility rule](https://docs.nvidia.com/cuda/archive/12.8.0/ada-compatibility-guide/index.html).
The manifest requires SM80 ELF coverage for this library and explicit SM70/86/89
coverage for the four core extensions; it does not assume cross-major support.

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

For an opt-in policy that separates a two-request decode tail at feedback
depth two, see [PP decode staggering](../design/pp_decode_stagger.md). It
requires a separate performance comparison; it is disabled by default.

## Optional decode transport and feedback packing

Two independent boolean options allow a staged rollout of the communication
changes. Both default to `false`; merge them into the existing JSON argument:

```bash
--additional-config '{"mtp_tail_warmup":true,"pp_decode_static_transfer":true,"pp_packed_feedback":true}'
```

`pp_decode_static_transfer` supports Model Runner V2, Qwen4Exp FP16, PP greater
than one, TP/DP equal to one, and MTP with one to four draft tokens. It requires
SP, DBO, and microbatching to be disabled. All pipeline ranks negotiate the
model-owned intermediate tensor schema once. Complete, uniform MTP decode
batches then send only real token rows without per-batch CPU metadata messages.
New/resumed requests, prefill, encoder work, and partial decode tails keep the
dynamic protocol. Local graph padding is initialized separately. The receive
pool has one buffer per in-flight batch, with CUDA stream/event ordering guarding
allocation, consumption, and reuse. Startup and first-hit logs identify whether
the path is enabled and exercised; enabling it does not make every batch eligible.

`pp_packed_feedback` combines sampled tokens, next-step drafts, and accepted/
rejected counts into one broadcast on the existing independent communication
group. Integer widths and payload size are preserved: eight requests with MTP4
use 640 bytes. Packet buffers and events are reused at the configured feedback
depth. Request-generation masks still reject stale feedback after cancellation
or preemption. Source sampler/draft tensors must remain immutable until their
broadcast-stream reads finish; allocation lifetime alone does not guard writes.

Validate static transfer first with packed feedback disabled, then enable packed
feedback and repeat the same requests. Include request cancellation and immediate
replacement, prefix-cache cold/warm runs, mixed tail lengths, and image/video
requests. Compare warmed common-decode throughput separately from first-run
prefill latency. These options do not enable NCCL P2P or change GPU topology.

Forced prefix-cache resets use each scheduled batch's monotonic step to reject
old results. Rejected draft tokens and empty prefill outputs do not represent a
number of pending output batches; counting token placeholders could discard
valid results after resumption. The step cutoff filters old local PP/MTP results
before they can change rejection accounting, encoder state, or generated tokens.
This does not establish generation isolation for optional external KV connectors
or global routed-expert recording; those paths are outside this validation.

Focused tests (native dependencies must already be built):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  tests/v1/worker/test_pp_decode_transfer.py \
  tests/v1/worker/test_pp_feedback_packet.py \
  tests/v1/worker/test_pp_delayed_mamba.py \
  --confcutdir=tests/v1/worker -q

# Exercise the actual gather, Mamba state-copy, and draft-scatter kernels.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/worker/test_pp_delayed_mamba.py \
  --confcutdir=tests/v1/worker -q
```

`tests/v1/core/test_pp_reset_prefix_cache.py` adds real asynchronous scheduler
regressions for forced reset, resumption, cancellation, empty prefill, multiple
in-flight batches, and partial MTP acceptance at PP feedback depths 2, 4, and 7.

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

`--language-model-only` retains the Qwen4Exp language-model wrapper prefix and
skips mmproj discovery, metadata, and payload loading. A text-only GGUF directory
therefore does not need a vision projector. CPU adapter tests write and read
small actual GGUF files, including missing/extra/invalid tensors and temporal
patch reconstruction:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  tests/models/gguf/test_qwen3_vision_adapter.py \
  --confcutdir=tests/models/gguf -q
```

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
