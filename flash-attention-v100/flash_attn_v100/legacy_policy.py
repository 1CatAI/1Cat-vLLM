# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent no-policy entry points; configured engines do not call these."""

import os


def partition_size():
    return os.getenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE")


def dynamic_partitions():
    return os.getenv("VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS", "1") != "0"


def staged_pv():
    return os.getenv("VLLM_FLASH_V100_XQA_STAGED_PV", "0") == "1"


def share_workspace():
    return os.getenv("VLLM_FLASH_V100_SHARE_DECODE_WORKSPACE", "1") != "0"


def scalar_fast():
    return (
        os.getenv(
            "VLLM_FLASH_V100_E4M3_SCALAR_FAST",
            os.getenv("VLLM_FLASH_V100_TP2_E4M3_SCALAR_FAST", "1"),
        )
        == "1"
    )


def padded_smem():
    return os.getenv("VLLM_FLASH_V100_XQA_PADDED_SMEM", "1") != "0"


def dual_cta():
    return os.getenv("VLLM_FLASH_V100_XQA_G6_DUAL_CTA", "0") == "1"


def batch_xqa():
    return os.getenv("VLLM_FLASH_V100_E4M3_BATCH_XQA", "1") == "1"
