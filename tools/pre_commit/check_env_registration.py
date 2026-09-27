# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject new VLLM_* environment reads that bypass vllm/envs.py.

`envs.compile_factors()` hashes every registered variable into the
torch.compile cache key. A switch read straight from `os.environ` is invisible
to it: flipping such a switch can load an AOT artifact compiled for the other
setting. Registering the variable in `vllm/envs.py` keeps it in the key; a
variable that never changes compiled code also belongs in `ignored_factors`
there, so it does not invalidate the cache.
"""

import ast
import sys

import regex as re

ENVS_FILE = "vllm/envs.py"

_READ_PATTERN = re.compile(
    r"""(?:os\.getenv|os\.environ\.get|os\.environ\.setdefault)\(\s*"""
    r"""["'](VLLM_[A-Z0-9_]+)["']"""
    r"""|os\.environ\[\s*["'](VLLM_[A-Z0-9_]+)["']\s*\](?!\s*=[^=])"""
)

# Direct reads that existed when this check was added. Register them in
# vllm/envs.py (or delete them) and drop them from this list over time; do not
# add new entries.
BASELINE: frozenset[str] = frozenset(
    {
        "VLLM_CPU_CI_ENV",
        "VLLM_CPU_SIM_MULTI_NUMA",
        "VLLM_CUTLASS_SRC_DIR",
        "VLLM_DFLASH_DDTREE_ATTN_COMPACT_BATCH",
        "VLLM_DFLASH_DDTREE_CONV_KERNEL",
        "VLLM_DFLASH_DDTREE_DISABLE_GDN_FAST_BUILD",
        "VLLM_DFLASH_DDTREE_DISABLE_GDN_FAST_BUILD_CACHE",
        "VLLM_DFLASH_DDTREE_ENABLE_GDN_FAST_BUILD",
        "VLLM_DFLASH_DDTREE_ENABLE_GDN_FAST_BUILD_CACHE",
        "VLLM_DFLASH_DDTREE_ENGINE_PROFILE",
        "VLLM_DFLASH_DDTREE_FAST_BUILD_DEBUG",
        "VLLM_DFLASH_DDTREE_FORCE_MAMBA_COMPACT",
        "VLLM_DFLASH_DDTREE_GDN_FAST_BUILD_TRITON",
        "VLLM_DFLASH_DDTREE_GDN_SHARED_COMMON",
        "VLLM_DFLASH_DDTREE_GPU_SAMPLER",
        "VLLM_DFLASH_DDTREE_LINEAR_GDN",
        "VLLM_DFLASH_DDTREE_MAMBA_COMPACT_BATCH",
        "VLLM_DFLASH_DDTREE_METADATA_PROFILE",
        "VLLM_DFLASH_DDTREE_PATH_PROBE",
        "VLLM_DFLASH_DDTREE_PATH_PROBE_LAYER",
        "VLLM_DFLASH_DDTREE_PATH_PROBE_MAX_REPORTS",
        "VLLM_DFLASH_DDTREE_PATH_PROBE_NODE_LIMIT",
        "VLLM_DFLASH_DDTREE_PROFILE",
        "VLLM_DFLASH_DDTREE_QLA_GDN",
        "VLLM_DFLASH_DDTREE_SERIAL_GDN",
        "VLLM_DFLASH_DDTREE_SKIP_MAMBA_COMPACT",
        "VLLM_DFLASH_DDTREE_STOCHASTIC_TOPK_LOGITS",
        "VLLM_DFLASH_DDTREE_TARGET_FORWARD_NVTX",
        "VLLM_DFLASH_DDTREE_TARGET_FORWARD_PROFILER_STEP",
        "VLLM_DFLASH_DDTREE_TRACE_JSONL",
        "VLLM_DFLASH_DDTREE_TRACE_KV_CACHE_DIFF",
        "VLLM_DFLASH_DDTREE_TRITON_SAMPLER",
        "VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE",
        "VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE_CONTEXT",
        "VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE_TOPK",
        "VLLM_DFLASH_DDTREE_WORKER_PROFILE",
        "VLLM_DFLASH_DEBUG_COORD_TRACE",
        "VLLM_DFLASH_DEBUG_PP_AUX_DUMP_LIMIT",
        "VLLM_DFLASH_DEBUG_PROPOSAL_STAGES",
        "VLLM_DFLASH_DEBUG_TARGET_LAYER_TRACE",
        "VLLM_DFLASH_DEBUG_TARGET_LOGITS",
        "VLLM_DFLASH_DEBUG_TARGET_TRACE_MIN_POSITION",
        "VLLM_DFLASH_DEBUG_TENSOR_DUMP_DIR",
        "VLLM_DFLASH_DEBUG_TENSOR_DUMP_LIMIT",
        "VLLM_DIST_IDENT",
        "VLLM_FLASH_V100_DEBUG_ROUTE_SUMMARY",
        "VLLM_FLASH_V100_SMALLQ_DECODE_USE_XQA",
        "VLLM_FLASH_V100_SMALLQ_DECODE_XQA_MIN_SEQ_LEN",
        "VLLM_FLASH_V100_XQA_E4M3_G6_P512_BEGIN",
        "VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO",
        "VLLM_FLASH_V100_XQA_E4M3_G6_WAVE_PARTITIONS",
        "VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA",
        "VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE",
        "VLLM_KPOOL_SKIP_DECODE_WRITE",
        "VLLM_KPOOL_SKIP_TAIL_CACHE",
        "VLLM_QWEN35_MTP_KEEP_QUANT",
        "VLLM_QWEN35_MTP_SHARE_IO_WEIGHTS",
        "VLLM_SM70_DFLASH2_BF16_EMULATION",
        "VLLM_SM70_DUMP_AWQ_MOE_BUFFERS",
        "VLLM_SM70_DUMP_AWQ_MOE_LABELS",
        "VLLM_SM70_DUMP_COMPILE_GRAPH_INPUT_DIR",
        "VLLM_SM70_DUMP_COMPILE_GRAPH_INPUT_STEPS",
        "VLLM_SM70_DUMP_GDN_CORE_DIR",
        "VLLM_SM70_DUMP_GDN_CORE_ENABLE_FILE",
        "VLLM_SM70_DUMP_GDN_CORE_LAYER_IDS",
        "VLLM_SM70_DUMP_GDN_CORE_MAX_DUMPS",
        "VLLM_SM70_DUMP_GDN_GRAPH_BUFFERS",
        "VLLM_SM70_DUMP_GDN_GRAPH_DIR",
        "VLLM_SM70_DUMP_GDN_GRAPH_ENABLE_FILE",
        "VLLM_SM70_DUMP_GDN_GRAPH_LABELS",
        "VLLM_SM70_DUMP_GDN_GRAPH_LAYER_IDS",
        "VLLM_SM70_DUMP_GDN_GRAPH_METADATA",
        "VLLM_SM70_DUMP_GDN_GRAPH_SHAPES",
        "VLLM_SM70_DUMP_GDN_GRAPH_STATE_INDICES",
        "VLLM_SM70_DUMP_GDN_GRAPH_STEPS",
        "VLLM_SM70_DUMP_GDN_PROJ_DIR",
        "VLLM_SM70_DUMP_GDN_PROJ_ENABLE_FILE",
        "VLLM_SM70_DUMP_GDN_PROJ_LAYER_IDS",
        "VLLM_SM70_DUMP_GDN_PROJ_MAX_DUMPS",
        "VLLM_SM70_DUMP_GDN_STATE_TABLE_SEQS",
        "VLLM_SM70_DUMP_QWEN_LAYER_COUNTS",
        "VLLM_SM70_DUMP_QWEN_LAYER_DIR",
        "VLLM_SM70_DUMP_QWEN_LAYER_DIRECT_SAVE",
        "VLLM_SM70_DUMP_QWEN_LAYER_ENABLE_FILE",
        "VLLM_SM70_DUMP_QWEN_LAYER_GRAPH_BUFFERS",
        "VLLM_SM70_DUMP_QWEN_LAYER_GRAPH_STEPS",
        "VLLM_SM70_DUMP_QWEN_LAYER_IDS",
        "VLLM_SM70_DUMP_QWEN_LAYER_LABELS",
        "VLLM_SM70_DUMP_QWEN_LAYER_MAX_DUMPS",
        "VLLM_SM70_DUMP_QWEN_LAYER_MAX_TOKENS",
        "VLLM_SM70_DUMP_QWEN_MLP_INTERNALS",
        "VLLM_SM70_DUMP_SAMPLER_LOGITS_ENABLE_FILE",
        "VLLM_SM70_DUMP_TOP_TOKEN_MARGIN_PROBE_TOKENS",
        "VLLM_SM70_FLASHQLA_DIRECT_OUTPUT",
        "VLLM_SM70_FLASHQLA_INDEXED_PREFILL",
        "VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL",
        "VLLM_SM70_FP8_BATCH_PRESCALED",
        "VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_COMPARE",
        "VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_COMPARE_REPORTS",
        "VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_DECOMPOSED",
        "VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_EXACT_LAYOUT",
        "VLLM_SM70_FP8_MOE_LEGACY_SINGLE_TOKEN_COMPACT_NATIVE_UNPERMUTE",
        "VLLM_SM70_GDN_PREFILL_PROFILE",
        "VLLM_SM70_GDN_PREFILL_PROFILE_MAX_LOGS",
        "VLLM_SM70_GDN_PREFILL_PROFILE_MAX_PER_STAGE",
        "VLLM_SM70_GDN_STATE_CONTRACT_ASSERT",
        "VLLM_SM70_GLM53_EXACT_KDA_GEMV",
        "VLLM_SM70_GLM53_FP16_GEMV_LIBRARY",
        "VLLM_SM70_INDEXER_DECODE_CUBLAS",
        "VLLM_SM70_INDEXER_DECODE_CUBLAS_MIN_KEYS",
        "VLLM_SM70_INDEXER_FUSED_LOGITS",
        "VLLM_SM70_INDEXER_PREFILL_CUBLAS",
        "VLLM_SM70_INDEXER_PREFILL_TILE_MB",
        "VLLM_SM70_INDEXER_RELU",
        "VLLM_SM70_KDA_PREFILL_SCHEDULE",
        "VLLM_SM70_MTP_DUMP_TENSOR_MAX",
        "VLLM_SM70_QSA_GROUPED_PAD_FIX",
        "VLLM_SM70_QSA_GROUPED_PAGE4",
        "VLLM_SM70_QSA_INDEXER_CUBLAS",
        "VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_ROWS",
        "VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS",
        "VLLM_SM70_QSA_INDEXER_SCORE_TILE_MB",
        "VLLM_SM70_QSA_MTP_TOPK",
        "VLLM_SM70_QSA_TOPK_LIBRARY",
        "VLLM_SM70_QSA_XQA_PAGE4",
        "VLLM_SM70_QSA_XQA_PAGE4_MIN_ROWS",
        "VLLM_SM70_QWEN38_QPN_ROUTE_DEBUG",
        "VLLM_SM70_QWEN_GDN_ASSERT_NO_ACTIVE_SPEC_STANDARD",
        "VLLM_SM70_SPEC_TARGET_FORWARD_NVTX",
        "VLLM_SM70_SPEC_TARGET_FORWARD_PROFILER_STEP",
        "VLLM_SM70_TURBOQUANT_COMPARE_DUMP_DIR",
        "VLLM_SM70_TURBOQUANT_COMPARE_LOG_PATH",
        "VLLM_SM70_TURBOQUANT_FLASH_V100_DECODE",
        "VLLM_SM70_TURBOQUANT_FLASH_V100_PREFILL",
        "VLLM_SM70_TURBOQUANT_RESERVE_WORKSPACE",
        "VLLM_SPEC_DUMP_ALIGNMENT_DIR",
        "VLLM_SPEC_DUMP_ALIGNMENT_TAG",
    }
)


def registered_variables() -> set[str]:
    with open(ENVS_FILE, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        else:
            continue
        if (
            isinstance(target, ast.Name)
            and target.id == "environment_variables"
            and isinstance(node.value, ast.Dict)
        ):
            return {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    raise RuntimeError(f"environment_variables not found in {ENVS_FILE}")


def scan_file(path: str, known: set[str]) -> int:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    returncode = 0
    for match in _READ_PATTERN.finditer(content):
        name = match.group(1) or match.group(2)
        if name in known or name in BASELINE:
            continue
        line_num = content[: match.start()].count("\n") + 1
        print(
            f"{path}:{line_num}: \033[91merror:\033[0m {name} is read from "
            f"os.environ but not registered in {ENVS_FILE}. Register it there "
            "(and add it to ignored_factors if it never changes compiled code)."
        )
        returncode = 1
    return returncode


def main() -> int:
    known = registered_variables()
    returncode = 0
    for filename in sys.argv[1:]:
        if filename == ENVS_FILE:
            continue
        returncode |= scan_file(filename, known)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
