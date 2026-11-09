# torch backports for the pinned torch 2.10.0

1Cat pins torch 2.10.0 (the last cu128 wheel with Volta, sm_70). Fixes that landed
in later torch releases and that this stack needs are kept here as patches and
applied to the installed torch with `apply.sh`. Each patch names its upstream
source. `apply.sh` is idempotent and refuses files it does not recognise.

| patch | upstream | why |
|---|---|---|
| 0001 aot_compile_types.py | pytorch/pytorch #173556 (main dffe73e2, in 2.11+) | AOT compile artifacts (`VLLM_USE_AOT_COMPILE`, compile cache on) reference Triton kernels through indices of a process-local side table; a fresh process has an empty table. Loading such an artifact fails with an empty assertion (recompile) or, seen with Qwen3.8-Flash-Next on an RTX 8000 stage, resolves a wrong kernel and dies with an illegal memory access. The fix serialises the table into the artifact and restores it on load; old artifacts without a table are still read the old way, so delete `torch_aot_compile/` under the cache root once after applying. |

Usage after installing or reinstalling torch in a venv:

    tools/torch_patches/apply.sh /path/to/venv/bin/python
