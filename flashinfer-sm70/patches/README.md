# FlashInfer SM70 Patch Series

Patches in this directory apply only to the pinned FlashInfer `v0.6.13`
source prepared by `tools/prepare-flashinfer-sm70.sh`.  The canonical
FlashInfer backend and its SM75+ build remain unchanged.

The intended series is:

1. register the dedicated SM70 JIT/AOT target without weakening the canonical
   backend capability gate;
2. add SM70 prefill and decode dispatch entries;
3. connect the Volta-specific MMA, shared-layout, and software-pipeline
   implementation;
4. expose planner/workspace metadata through the dedicated
   `FLASHINFER_SM70` vLLM backend.

Every patch must retain the exact upstream commit in its header and must be
independently removable.  A capability-check-only patch is not a usable
backend and may not be reported as progress beyond build plumbing.
