# SPDX-License-Identifier: Apache-2.0
"""Per-device dispatch; upstream architecture guards remain intact."""


def is_sm70(tensor):
    if tensor.device.type != "cuda":
        return False
    import torch

    return torch.cuda.get_device_capability(tensor.device) == (7, 0)
