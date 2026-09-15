import torch
from vllm import LLM, SamplingParams
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

llm = LLM(model='/home/nvidia/Dev/model/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw',
          tensor_parallel_size=4, max_model_len=256,
          max_num_batched_tokens=256, max_num_seqs=4,
          gpu_memory_utilization=0.95, enforce_eager=True,
          disable_log_stats=True, kv_cache_dtype='fp8')

# Register hooks on the decoder layers via the collective_rpc path
from vllm.distributed.parallel_state import get_tp_group

class Probe:
    def __init__(self):
        self.stats = []

    def __call__(self, worker):
        model = worker.model_runner.model
        layers = model.model.layers
        def make_hook(idx):
            def hook(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                self.stats.append((idx, o.float().norm().item()))
            return hook
        self.handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(len(layers))]
        return f"hooked {len(layers)} layers"

probe = Probe()
results = llm.collective_rpc(probe)
print("HOOKS REGISTERED:", results[0])

sp = SamplingParams(max_tokens=1, temperature=0.0)
gen = llm.generate(['The capital of France is'], sp)
print('OUT:', repr(gen[0].outputs[0].text[:40]))
