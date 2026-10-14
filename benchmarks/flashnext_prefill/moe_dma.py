# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare full-extent and occupied-slot DMA in the actual prefill pipeline.

The eight-layer cache, residency, staging allocation, transcode, and GEMMs are
identical in both modes. Initial and fragmented post-decode maps are timed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from moe_streams import build_cache
from moe_tiles import EXPERTS, HIDDEN, TOPK, artifact_identity, measure

from vllm.model_executor.layers.quantization.gguf_expert_cache import SHARDS, _native
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader


class LaneProbe:
    def __init__(self, cache):
        self.cache, self.lane = cache, cache.lane
        self.mode = "bulk"

    def __getattr__(self, name):
        return getattr(self.lane, name)

    def stage_cold(self, i, shard, dst, stream):
        if self.mode == "sparse":
            return self.lane.stage_cold(i, shard, dst, stream)
        cache = self.cache
        layer = cache.layers[i]
        rows = cache.tm.extent[i % cache.tm.VIEWS]
        if rows:
            _native().copy_h2d(
                dst,
                cache.host[layer, SHARDS[shard]].data_ptr(),
                rows * cache.sizes[layer, SHARDS[shard]],
                stream,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(9683)
    cache = build_cache(GGUFReader(args.model))
    cache.lane = LaneProbe(cache)
    x = (torch.randn(7344, HIDDEN, device="cuda") * 0.5).half()
    scores, ids = torch.randn(7344, EXPERTS, device="cuda").topk(TOPK, dim=1)
    weights = scores.softmax(dim=1)
    native = Path(_native().__file__)
    report = {
        "artifact": artifact_identity(),
        "cache_native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "cases": [],
        "maps": {},
    }

    def forward(gap=0, count=8):
        outputs = []
        for layer in cache.layers[:count]:
            outputs.append(cache.prefill_moe(layer, x, ids, weights))
            if gap:
                torch.cuda._sleep(gap)
        return outputs

    def check():
        for actual, expected in zip(forward(), reference):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        assert cache.tm.error.item() == 0

    try:
        reference = [out.cpu() for out in forward()]
        for phase in ("initial", "after_decode"):
            if phase == "after_decode":
                for _ in range(10):
                    for layer in cache.layers:
                        cache.resolve(
                            layer, torch.randperm(EXPERTS, device="cuda")[:50]
                        )
                torch.accelerator.synchronize()
            report["maps"][phase] = []
            for i in range(len(cache.layers)):
                host = cache.host_slot_of[i][cache.slot_of[i] < 0]
                report["maps"][phase].append(
                    {"cold": host.numel(), "extent": int(host.max()) + 1}
                )
            for gap in (0, 20000000):
                for mode in ("bulk", "sparse", "bulk"):
                    torch.accelerator.synchronize()
                    cache.lane.mode = mode
                    cache.tm.prefetched.clear()
                    cache.tm.start()
                    check()
                    forward(gap, count=4)
                    check()  # Restart while the interrupted pass has pending copies.
                    row = {
                        "phase": phase,
                        "mode": mode,
                        "gap_cycles": gap,
                        **measure(lambda gap=gap: forward(gap), 2),
                        "exact_output_and_restart": True,
                    }
                    report["cases"].append(row)
                    print(json.dumps(row), flush=True)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["cache_stats"] = cache.stats()
        assert report["cache_stats"]["errors"] == 0
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        torch.accelerator.synchronize()
        cache.close()


if __name__ == "__main__":
    main()
