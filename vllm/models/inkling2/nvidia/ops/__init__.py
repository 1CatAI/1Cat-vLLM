# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools

import torch


@functools.cache
def use_sm100_kernels() -> bool:
    """Whether to run the Blackwell kernels (tml_fa4, tcgen05 attention,
    TRT-LLM-gen expert GEMMs, e2m1 conversions). Hopper uses Humming experts;
    SM12x uses Triton experts and does not support tcgen05."""
    from vllm.platforms import current_platform

    return current_platform.is_device_capability_family(100)


@functools.cache
def is_sm70() -> bool:
    """Whether the target CUDA device is pre-Ampere (Volta SM70 / Turing SM75)."""
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        return False
    return not current_platform.has_device_capability(80)


@functools.cache
def model_act_dtype() -> torch.dtype:
    """Primary activation/weight dtype for Inkling2 on this device.

    Volta (SM70) has no native BF16 tensor cores, so unquantized weights and
    activations run in FP16 while AttnRes/loop/LSE accumulations stay FP32.
    """
    return torch.float16 if is_sm70() else torch.bfloat16


@functools.cache
def use_pdl() -> bool:
    """Whether the Triton kernels use programmatic dependent launch
    (griddepcontrol) and TMA descriptors, both SM90+. Ampere and Ada (SM8x)
    run the same kernels without them, and Humming experts like Hopper."""
    from vllm.platforms import current_platform

    return current_platform.has_device_capability(90)


@functools.cache
def use_native_conv_mix() -> bool:
    """Whether the csrc/inkling2 conv/mix kernels run here: they are built
    for Hopper and SM10x only."""
    from vllm.platforms import current_platform

    return any(current_platform.is_device_capability_family(f) for f in (90, 100))


@functools.cache
def use_native_e4m3() -> bool:
    """Whether Triton converts e4m3 (fp8e4nv) here, SM89+. Below, kernels
    take e4m3 tensors as uint8 and convert in software (`ops/e4m3.py`)."""
    from vllm.platforms import current_platform

    return current_platform.has_device_capability(89)
