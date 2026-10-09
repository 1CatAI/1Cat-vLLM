# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage bindings shared by execution and offline path explanations.

These declarations describe native semantics, not model admission. Layout and
model-specific gates remain in the existing route selector/weight preparation.
"""

STAGE_BINDINGS = {
    "w13": {
        "compact": (
            "single_token_compact_dense_w13",
            "input+W13",
            "compact active",
            "TM FP16",
        ),
        "indexed": (
            "single_token_indexed_dense_w13",
            "input+W13",
            "indexed active",
            "TM FP16",
        ),
        "active_dense": ("single_token_dense_w13", "input+W13", "active", "TM FP16"),
        "indexed_prefill": ("indexed_dense_w13", "W13", "indexed input", "TM FP16"),
        "active_grouped": (
            "active_dense_stage",
            "W13",
            "active expert offsets",
            "TM FP16",
        ),
        "dense": ("dense_stage", "W13", "dense expert offsets", "TM FP16"),
        "batched": ("gemm", "W13", "dense expert offsets", "TM grouped"),
        "per_expert_dispatch": (
            "gemm",
            "W13",
            "dense expert offsets",
            "TM per expert dispatch",
        ),
    },
    "w2": {
        "indexed": (
            "single_token_indexed_dense_stage",
            "W2",
            "indexed active",
            "TM FP16",
        ),
        "active_dense": ("single_token_dense_stage", "W2", "active", "TM FP16"),
        "active_grouped": (
            "active_dense_stage",
            "W2",
            "active expert offsets",
            "TM FP16",
        ),
        "dense": ("dense_stage", "W2", "dense expert offsets", "TM FP16"),
        "batched": ("gemm", "W2", "dense expert offsets", "TM grouped"),
        "per_expert_dispatch": (
            "gemm",
            "W2",
            "dense expert offsets",
            "TM per expert dispatch",
        ),
    },
}


def native_binding(family: str, stage: str, mode: str) -> str:
    suffix = STAGE_BINDINGS[stage][mode][0]
    tail = "_per_expert_dispatch_out" if mode == "per_expert_dispatch" else "_out"
    return f"{family.lower()}_moe_{suffix}_sm70{tail}"
