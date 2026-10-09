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


# (format, stage, effective mode): native name, covered stages, layout, arithmetic.
# Execution, coverage generation and route explanations use this same table.
FP4_STAGE_BINDINGS = {
    ("nvfp4", "w13", "active_grouped"): (
        "nvfp4_grouped_w13_sm70_out",
        "w13+activation",
        "shared compact route groups",
        "split4/8; original fused SwiGLU rounding",
    ),
    ("nvfp4", "w2", "active_grouped"): (
        "nvfp4_grouped_w2_sm70_out",
        "w2+reduce",
        "shared compact route groups",
        "original grouped W2 reduction",
    ),
    ("nvfp4", "w2", "grouped_batch_reduce"): (
        "nvfp4_grouped_w2_batch_reduce_sm70_out",
        "w2+reduce",
        "shared MTP5 route groups",
        "original grouped batch reduction",
    ),
    ("nvfp4", "w13", "dense"): (
        "nvfp4_moe_dense_stage_sm70_out",
        "w13",
        "expert offsets",
        "TurboMind FP16 boundary",
    ),
    ("nvfp4", "w13", "qpn"): (
        "nvfp4_moe_qpn_m1_sm70_out",
        "w13",
        "route slots",
        "FP32 accumulation; FP16 boundary",
    ),
    ("nvfp4", "w13", "qpn_mtp"): (
        "nvfp4_moe_qpn_mtp5_sm70_out",
        "w13",
        "M5 route slots",
        "explicit split-K; FP16 boundary",
    ),
    ("nvfp4", "w13", "qpn_raw"): (
        "nvfp4_moe_qpn_raw_scale_sm70_out",
        "w13",
        "raw E4M3 + global scale",
        "explicit split-K; FP16 boundary",
    ),
    ("nvfp4", "w2", "dense"): (
        "nvfp4_moe_dense_stage_sm70_out",
        "w2",
        "expert offsets",
        "TurboMind FP16 boundary",
    ),
    ("nvfp4", "w2", "qpn"): (
        "nvfp4_moe_qpn_m1_sm70_out",
        "w2",
        "route slots",
        "FP32 accumulation; FP16 boundary",
    ),
    ("nvfp4", "w2", "qpn_mtp"): (
        "nvfp4_moe_qpn_mtp5_sm70_out",
        "w2",
        "M5 route slots",
        "explicit split-K; FP16 boundary",
    ),
    ("nvfp4", "w2", "qpn_raw"): (
        "nvfp4_moe_qpn_raw_scale_sm70_out",
        "w2",
        "raw E4M3 + global scale",
        "explicit split-K; FP16 boundary",
    ),
    ("mxfp4", "w13", "dense"): (
        "mxfp4_moe_dense_stage_sm70_out",
        "w13",
        "expert offsets",
        "TurboMind FP16 boundary",
    ),
    ("mxfp4", "w13", "qpn"): (
        "mxfp4_moe_qpn_m1_sm70_out",
        "w13",
        "route slots",
        "FP32 accumulation; FP16 boundary",
    ),
    ("mxfp4", "w2", "dense"): (
        "mxfp4_moe_dense_stage_sm70_out",
        "w2",
        "expert offsets",
        "TurboMind FP16 boundary",
    ),
    ("mxfp4", "w2", "qpn"): (
        "mxfp4_moe_qpn_m1_sm70_out",
        "w2",
        "route slots",
        "FP32 accumulation; FP16 boundary",
    ),
    ("nvfp4", "w13", "fused_qpn"): (
        "nvfp4_qwen38_w13_fused_swiglu_out",
        "w13+activation",
        "interleaved M1",
        "preserves FP16 intermediate rounding",
    ),
    ("nvfp4", "w13", "fused_batch_qpn"): (
        "nvfp4_moe_qpn_w13_swiglu_batch_sm70_out",
        "w13+activation",
        "M4/8/16 route slots",
        "preserves FP16 intermediate rounding",
    ),
    ("nvfp4", "w13", "fused_batch_qpn_raw"): (
        "nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out",
        "w13+activation",
        "M4/8/16 raw scales",
        "preserves FP16 intermediate rounding",
    ),
    ("nvfp4", "w13", "indexed_prefill"): (
        "nvfp4_moe_indexed_dense_stage_sm70_out",
        "w13",
        "indexed input",
        "preserves FP16 intermediate rounding",
    ),
    ("nvfp4", "w13", "indexed_fused"): (
        "nvfp4_moe_indexed_fused_swiglu_sm70_out",
        "w13+activation",
        "indexed interleaved",
        "preserves FP16 intermediate rounding",
    ),
    ("nvfp4", "w13", "indexed_split_fused"): (
        "nvfp4_moe_indexed_fused_swiglu_sm70_out",
        "w13+activation",
        "N256 + N64 indexed slices",
        "preserves FP16 intermediate rounding",
    ),
    ("nvfp4", "w13", "glm_qpn"): (
        "nvfp4_glm53_moe_q8_qpn_sm70_out",
        "w13",
        "GLM TP8 M8 sorted rows",
        "preserves FP16 intermediate rounding",
    ),
    ("nvfp4", "w2", "direct_reduce"): (
        "nvfp4_qwen38_w2_direct_reduce_out",
        "w2+reduce",
        "Qwen M1",
        "fused weighted reduction; original rounding",
    ),
    ("nvfp4", "w2", "batch_reduce"): (
        "nvfp4_moe_qpn_w2_reduce_sm70_out",
        "w2+reduce",
        "Qwen batch",
        "fused weighted reduction; original rounding",
    ),
    ("nvfp4", "w2", "batch_reduce_raw"): (
        "nvfp4_moe_qpn_raw_w2_reduce_sm70_out",
        "w2+reduce",
        "Qwen batch raw scales",
        "fused weighted reduction; original rounding",
    ),
    ("mxfp4", "w13", "prepare_w13"): (
        "mxfp4_moe_single_token_prepare_w13_sm70_out",
        "route+w13",
        "direct top6",
        "TurboMind FP16 boundary",
    ),
}


def fp4_native_binding(family: str, stage: str, mode: str) -> str:
    return FP4_STAGE_BINDINGS[family, stage, mode][0]
