# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.distributed.device_communicators.cuda_communicator import (
    _all_gather_existing_pynccl,
)


@pytest.mark.parametrize(
    "shape,dim", [((2, 3), 0), ((2, 3), -1), ((2, 3, 4), 1), ((0, 3), 0)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int64])
def test_existing_communicator_preserves_concat_layout(shape, dim, dtype):
    value = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).to(dtype)
    if value.ndim == 3:
        value = value.transpose(0, 2)
    chunks = [value + rank * 8 for rank in range(4)]

    class Communicator:
        def all_gather(self, output, input_):
            assert input_.is_contiguous()
            output.copy_(torch.stack(chunks))

    actual = _all_gather_existing_pynccl(value, Communicator(), 4, dim)
    torch.testing.assert_close(actual, torch.cat(chunks, dim), rtol=0, atol=0)
