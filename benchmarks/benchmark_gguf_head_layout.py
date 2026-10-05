# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare input permutation and load-time head restoration on real GGUF rows."""

import argparse
import json
from pathlib import Path

import gguf
import numpy as np
import torch

from vllm.model_executor.layers.quantization.gguf_layout import GGUFHeadTilingLayout
from vllm.model_executor.layers.quantization.gguf_transcode import transcode_affine
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    apply_prepared_gguf_projections,
    prepare_gguf_projections,
)
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("output")
a = p.parse_args()
torch.set_num_threads(1)
reader = GGUFReader(a.model)
ts = [
    t
    for t in reader.tensors
    if t.name.endswith("ssm_out.weight") and int(t.tensor_type) == 14
][:6]
assert len(ts) == 6
layout = GGUFHeadTilingLayout(3, 128)
old = []
new = []
errors = []
for t in ts:
    raw = np.asarray(t.data)
    assert int(t.tensor_type) == 14
    raw = layout.shard_weight(
        torch.from_numpy(raw.copy()),
        dim=1,
        logical_size=int(t.shape[0]),
        block_size=256,
        tp_rank=0,
        tp_size=4,
    ).numpy()
    w = torch.from_numpy(raw).cuda()
    old.append(prepare_gguf_projections([(w, 14)], torch.float16, True, 256))
    new.append(
        prepare_gguf_projections(
            [(w, 14)], torch.float16, True, 256, input_layout=layout
        )
    )
    assert new[-1][0].input_layout_restored
    c = transcode_affine(raw, 14)
    oracle = gguf.quants.dequantize(raw, t.tensor_type)
    d = c.dequantize() - oracle
    errors.append(
        {
            "tensor": t.name,
            "k": c.codes.shape[1],
            "n": c.codes.shape[0],
            "canonical_max_abs": float(abs(d).max()),
            "canonical_relative_l2": float(np.linalg.norm(d) / np.linalg.norm(oracle)),
        }
    )
rows = []
for m in [1, 5, 20]:
    torch.manual_seed(20261005 + m)
    x = (torch.randn(m, 1536, device="cuda") * 0.125).half()
    diff = []
    for i, t in enumerate(ts):
        y = apply_prepared_gguf_projections(layout.input_to_gguf(x), old[i])
        z = apply_prepared_gguf_projections(x, new[i])
        torch.testing.assert_close(y, z, rtol=0.003, atol=0.0002)
        diff.append(
            {
                "max_abs": float((y.float() - z.float()).abs().max()),
                "relative_l2": float((y.float() - z.float()).norm() / y.float().norm()),
            }
        )
    graphs = []
    for projs, shuffle in [(old, True), (new, False)]:
        for i in range(6):
            apply_prepared_gguf_projections(
                layout.input_to_gguf(x) if shuffle else x, projs[i]
            )
        torch.accelerator.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(6):
                apply_prepared_gguf_projections(
                    layout.input_to_gguf(x) if shuffle else x, projs[i]
                )
        graphs.append(g)
    samples = [[], []]
    for epoch in range(8):
        for arm in [0, 1] if epoch % 2 == 0 else [1, 0]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                graphs[arm].replay()
            end.record()
            end.synchronize()
            samples[arm].append(start.elapsed_time(end) * 1000 / 600)
    rows.append(
        {
            "m": m,
            "old_us": float(np.median(samples[0])),
            "new_us": float(np.median(samples[1])),
            "samples_us": samples,
            "projection_difference": diff,
        }
    )
report = {
    "weight_banks": 6,
    "source_types": [int(t.tensor_type) for t in ts],
    "canonical_errors": errors,
    "rows": rows,
    "qualification": (
        "Graph microbenchmark, six distinct real TP4 GDN output weights; "
        "not full-model timing."
    ),
}
Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
