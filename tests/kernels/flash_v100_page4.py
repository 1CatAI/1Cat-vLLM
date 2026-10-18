# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device gate for tests of the Flash-V100 grouped sparse page4 kernels."""

import pytest
import torch


def require_flash_v100_page4():
    """Return the Flash-V100 extension if its build runs page4 on this device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    # Import through the package like the QSA routes do. The top-level
    # flash_attn_v100_cuda name may resolve to another tree's build, and loading
    # both registers the extension's pybind types twice.
    interface = pytest.importorskip("flash_attn_v100.flash_attn_interface")
    extension = interface.flash_attn_v100_cuda
    supported = getattr(extension, "grouped_sparse_page4_device_supported", None)
    if supported is None or not supported(torch.accelerator.current_device_index()):
        pytest.skip("this Flash-V100 build does not run grouped page4 on this device")
    return extension
