# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report packaged dense H3 capabilities without allocating a CUDA context.

Studio selects this profile only on a dedicated TP4 V100 group. Native CLI
options remain explicit; this does not enable sparse attention or step caches.
"""

from importlib import import_module

PROFILE = "sm70-dense-v1"


def studio_capabilities():
    try:
        projection = import_module("vllm._h3_w8a16_C")
        attention = import_module("vllm._h3_flashattn_C")
        reduction = import_module("vllm._sm70_exact_reduce_C")
    except (ImportError, OSError, RuntimeError):
        return {"profile": PROFILE, "available": False}
    return {
        "profile": PROFILE,
        "available": bool(
            hasattr(projection, "scaled_add_")
            and hasattr(projection, "ColumnMajorGemmPlan")
            and "query_tile"
            in (getattr(getattr(attention, "forward", None), "__doc__", "") or "")
            and all(
                hasattr(reduction, name)
                for name in ("allocate", "open_handle", "release", "run")
            )
        ),
    }
