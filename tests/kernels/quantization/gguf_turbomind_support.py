# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single capability gate for the TurboMind GGUF tests.

The production code only prepares TurboMind GGUF storage on a device with
compute capability (7, 0) (``GGUFPreparedProjection._prepare``; the loader's
canonical expert storage and the SM70 prepare/restore ops behave alike). On
other devices the ops refuse loudly (``requires SM70``) and the loader never
selects this path, so the tests that exercise it are skipped there.
"""

import pytest

from vllm.platforms import current_platform

REQUIRES_SM70_REASON = (
    "TurboMind GGUF storage requires compute capability (7, 0) on the test device"
)


def turbomind_gguf_supported() -> bool:
    """Mirror the production check on the device the tests allocate on."""
    return current_platform.is_cuda() and current_platform.get_device_capability() == (
        7,
        0,
    )


requires_turbomind_gguf = pytest.mark.skipif(
    not turbomind_gguf_supported(), reason=REQUIRES_SM70_REASON
)
