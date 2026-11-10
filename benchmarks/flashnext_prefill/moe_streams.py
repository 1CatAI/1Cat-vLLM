# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen prefill stream dependencies with real weights and the native cache.

Uses eight logical layers (four real TP4 rank-zero layers repeated) to wrap
the residency-view ring. All modes share kernels, weights, expert residency,
and bank/staging capacity. Optional GPU sleep models a between-layer delay,
not concurrent model compute. This is not an end-to-end throughput result.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from moe_tiles import EXPERTS, HIDDEN, INTERMEDIATE, TOPK, artifact_identity, read_layer

from vllm.model_executor.layers.quantization.gguf_expert_cache import (
    SHARDS,
    ExpertCache,
    TurboMindPrefill,
)
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader


class StreamProbe(TurboMindPrefill):
    """Benchmark-only policies; production _issue/_locate/compute are reused."""

    mode = "shared"

    def start(self):
        super().start()
        device = self.cache.device
        self.cold_by_key = {
            "gu": self.cold_stream,
            "dn": torch.cuda.Stream(device, priority=-1),
        }
        self.resident_by_key = {
            "gu": self.resident_stream,
            "dn": torch.cuda.Stream(device, priority=-1),
        }
        self.located = [torch.cuda.Event() for _ in range(self.VIEWS)]
        self.view_done = [torch.cuda.Event() for _ in range(self.VIEWS)]
        for event in self.view_done:
            event.record()

    def _locate(self, i):
        p = i % self.VIEWS
        if self.mode == "split":
            # Down may still use this view after an interrupted pass.
            self.resident_stream.wait_event(self.view_done[p])
        super()._locate(i)
        self.located[p].record(self.resident_stream)

    def _issue(self, key, i):
        self.cold_stream = self.cold_by_key["gu" if self.mode == "shared" else key]
        self.resident_stream = self.resident_by_key[
            key if self.mode == "split" else "gu"
        ]
        if key == "dn" and self.mode == "split":
            self.resident_stream.wait_event(self.located[i % self.VIEWS])
        super()._issue(key, i)
        if key == "dn":
            self.view_done[i % self.VIEWS].record(self.resident_stream)


def build_cache(reader):
    real_layers = [0, 1, 17, 47]
    sources = [read_layer(reader, layer) for layer in real_layers]
    banks = []
    for layer in range(8):
        raw, gate_type, down_type, left = sources[layer % len(sources)]
        for shard, part in zip(SHARDS, ("gate", "up", "down")):
            source = (
                (down_type, INTERMEDIATE, left)
                if part == "down"
                else (gate_type, HIDDEN, 0)
            )
            banks.append((layer, shard, raw[part].shape, 0, source))
    cache = ExpertCache(banks, EXPERTS, 0.8, 8, 0.05, None, "cuda:0", "turbomind", 4096)
    for layer in cache.layers:
        raw = sources[layer % len(sources)][0]
        for shard, part in zip(SHARDS, ("gate", "up", "down")):
            cache.store_bank(layer, shard, torch.from_numpy(raw[part]))
    cache.tm.__class__ = StreamProbe
    cache.start()
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=7344)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--gap-cycles", type=int, nargs="+", default=[0, 20000000])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens < 1 or args.iterations < 1 or min(args.gap_cycles) < 0:
        parser.error("invalid tokens, iterations, or gap cycles")
    torch.manual_seed(9683)
    cache = build_cache(GGUFReader(args.model))
    gen = torch.Generator(device="cuda").manual_seed(9683)
    x = (torch.randn(args.tokens, HIDDEN, device="cuda", generator=gen) * 0.5).half()
    scores, ids = torch.randn(args.tokens, EXPERTS, device="cuda", generator=gen).topk(
        TOPK, dim=1
    )
    weights = scores.softmax(dim=1)
    report = {
        "artifact": artifact_identity(),
        "tokens": args.tokens,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "resident_experts": cache.resident0,
        "host_slots": cache.host_slots,
        "canonical_and_staging_bytes": cache.tm.storage_bytes(),
        "cases": [],
    }

    def forward(gap, count=8):
        outputs = []
        for layer in cache.layers[:count]:
            outputs.append(cache.prefill_moe(layer, x, ids, weights))
            if gap:
                torch.cuda._sleep(gap)
        return outputs

    def select(mode):
        torch.accelerator.synchronize()
        cache.tm.prefetched.clear()
        cache.tm.mode = mode
        cache.tm.start()

    def compare(outputs, reference):
        # Stream changes preserve arithmetic: exact output is the contract.
        for actual, expected in zip(outputs, reference):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        assert cache.tm.error.item() == 0

    try:
        # Warm native tuning, Triton, allocation pools and every source format.
        reference = [out.cpu() for out in forward(0)]
        for gap in args.gap_cycles:
            for mode in ("shared", "cold_split", "split", "shared"):
                select(mode)
                compare(forward(gap), reference)
                # Restart without draining side-stream work from this pass.
                forward(gap, count=4)
                compare(forward(gap), reference)
                samples = []
                for _ in range(5):
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(args.iterations):
                        forward(gap)
                    end.record()
                    end.synchronize()
                    samples.append(begin.elapsed_time(end) / args.iterations)
                case = {
                    "mode": mode,
                    "gap_cycles": gap,
                    "median_ms": statistics.median(samples),
                    "samples_ms": samples,
                    "exact_output_and_restart": True,
                }
                report["cases"].append(case)
                print(json.dumps(case), flush=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
        # Exercise real decode misses, host write-backs, and a new snapshot.
        for layer in cache.layers:
            cache.resolve(layer, torch.arange(480, 512, device="cuda"))
        torch.accelerator.synchronize()
        for mode in ("shared", "cold_split", "split"):
            select(mode)
            compare(forward(0), reference)
        report["after_decode_residency_change_exact"] = True
        report["cache_stats"] = cache.stats()
        assert report["cache_stats"]["errors"] == 0
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["cache_stats"]), flush=True)
    finally:
        torch.accelerator.synchronize()
        cache.close()


if __name__ == "__main__":
    main()
