# PP NCCL lifecycle regression

These tests use the real `PPHandler`, `StaticPPDecodeTransfer`,
`AsyncIntermediateTensors`, and `GroupCoordinator` with CUDA streams and NCCL.
Only model metadata/input batches are synthetic. No model weights or network
downloads are needed. Tensor payloads are small; each process still creates a
normal CUDA context and NCCL communicators.

Run from an existing built vLLM environment on Linux:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m pytest \
  tests/v1/worker/test_pp_nccl_lifecycle.py -v -s

CUDA_VISIBLE_DEVICES=2,0,1,4,5,6,3 VLLM_TEST_PP_NCCL_WORLD_SIZE=7 \
  .venv/bin/python -m pytest tests/v1/worker/test_pp_nccl_lifecycle.py -v -s
```

Two ranks test feedback depth D=2. Four or more ranks test D=2 and D=4,
respecting the production requirement that D cannot exceed PP size. Both the
legacy two-broadcast and packed one-broadcast paths run with varying request
counts and sample widths, skipped prefills, finishing rows, cancellation and
immediate request-index reuse. Delayed GPU consumers, repeated ring wraps, and
allocator pressure exercise source and destination lifetimes. Checks copy data
to the CPU only after the case finishes, preserving asynchronous overlap.

Static transfer alternates with the metadata-bearing protocol and uses different
local graph padding on adjacent ranks. The first receive reuses cached storage
with an outstanding write on its allocation stream; pointer checks ensure this
condition actually occurred. Payload and zero-padding checks validate ownership.
The launcher selects the native CUDA allocator for this regression only.

Each NCCL/Gloo collective has a 45-second timeout; the pytest launcher kills the
whole torchrun process group after 240 seconds, including stuck descendants.
Override the outer limit with `VLLM_TEST_PP_NCCL_TIMEOUT`. A CPU-only timeout test
checks that a descendant cannot continue after cleanup. Child output is retained
in pytest's temporary `pp-nccl-lifecycle.log`; successful runs emit per-rank JSON
counts. Existing `NCCL_*` settings, including P2P policy, are preserved.

For manual diagnostics the worker can also be launched directly:

```bash
PYTORCH_CUDA_ALLOC_CONF=backend:native .venv/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  tests/v1/worker/pp_nccl_lifecycle_worker.py --steps 64 --collective-timeout 45
```

The direct command has collective timeouts but lacks the pytest launcher's outer
process-tree deadline. These small-tensor regressions complement model-serving
tests; they do not establish model correctness or end-to-end speedups.
