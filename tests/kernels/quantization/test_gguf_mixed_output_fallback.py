# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.gguf_turbomind import (
    apply_prepared_gguf_projections,
    mixed_projection_capabilities,
)


def test_uncanonicalized_format_keeps_original_projection_dispatch():
    class Projection:
        def __init__(self, source, kernel, offset):
            self.source_type = source
            self.kernel = kernel
            self.offset = offset

        def __call__(self, x):
            return x + self.offset

    # Q8_1 is absent from canonical family declarations. A mixed projection
    # must retain its imported fallback rather than fail family classification.
    projections = [Projection(12, object(), 1), Projection(9, None, -1)]
    x = torch.arange(8, dtype=torch.float16).reshape(2, 4)
    assert mixed_projection_capabilities(projections) == ()
    expected = torch.cat((x + 1, x - 1), dim=-1)
    torch.testing.assert_close(
        apply_prepared_gguf_projections(x, projections), expected, rtol=0, atol=0
    )
