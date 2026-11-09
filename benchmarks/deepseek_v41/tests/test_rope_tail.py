# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.models.deepseek_v41.quant_rope import apply_rope, rope_table

torch.manual_seed(0)
for compressed in (False, True):
    small = rope_table(compressed=compressed)  # 16384 rows + freq row
    full = rope_table(600000, compressed=compressed)  # reference big table
    x = torch.randn(4096, 1, 512, device="cuda").half()
    for lo in (0, 12288, 16380, 100000, 520000):
        pos = torch.arange(lo, lo + 4096, device="cuda")
        a = apply_rope(x, pos, small)
        b = apply_rope(x, pos, full)
        print(
            "compressed=%s pos %d..: bit-exact=%s maxdiff=%.3g"
            % (
                compressed,
                lo,
                torch.equal(a, b),
                (a.float() - b.float()).abs().max().item(),
            )
        )
