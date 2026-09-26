"""Forward-hook probe: per-layer hidden norms written to a file.

Run from the repo directory (sys.path[0] must be the repo so the
workers import the repo vllm, not site-packages). Requires
VLLM_ALLOW_INSECURE_SERIALIZATION=1 for the callable-class RPC.
"""
import torch
from vllm import LLM, SamplingParams

STATS_PATH = "/tmp/mhc_layer_stats.txt"

llm = LLM(model='/home/nvidia/Dev/model/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw',
          tensor_parallel_size=4, max_model_len=256,
          max_num_batched_tokens=256, max_num_seqs=4,
          gpu_memory_utilization=0.95, enforce_eager=True,
          disable_log_stats=True, kv_cache_dtype='fp8')


class Probe:
    """Register file-writing hooks on every decoder layer."""

    def __call__(self, worker):
        model = worker.model_runner.model
        layers = model.model.layers
        rank = worker.rank if hasattr(worker, "rank") else -1
        f = open(f"/tmp/mhc_layer_stats_r{rank}.txt", "w")

        def make_hook(idx):
            def hook(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                n = o.float().norm().item()
                f.write(f"{idx} {n:.6f}\n")
                f.flush()
            return hook

        handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(len(layers))]
        return f"hooked {len(layers)} layers on rank {rank}"


results = llm.collective_rpc(Probe())
print("HOOKS:", results[0])

sp = SamplingParams(max_tokens=1, temperature=0.0)
gen = llm.generate(['The capital of France is'], sp)
print('OUT:', repr(gen[0].outputs[0].text[:40]))
print('STATS:', STATS_PATH)
