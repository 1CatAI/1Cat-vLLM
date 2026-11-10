# Flash-Next GGUF on 4× V100 (SM70)

A reproducible OpenAI-compatible service for
Qwen3.8-Flash-Next-GSQ-RCO **IQ3_S** GGUF with its **MTP4** draft on four
V100-SXM2-32GB GPUs (TP4).

```bash
MODEL=/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
DRAFT=/models/mtp \
./serve.sh            # http://127.0.0.1:8000/v1, model name "flash-next"
```

`bench_client.py` measures TTFT, prefill and decode throughput and MTP
acceptance through the API:

```bash
python bench_client.py --model flash-next --lengths 1000,8000,30000 \
    --concurrency 1 --output c1.json
```

## What `serve.sh` enables

| Area | Setting | Why |
| --- | --- | --- |
| Speculation | MTP, 4 greedy draft tokens | ~4.9 emitted tokens per verification round |
| HC boundaries | `sm70_hcx` (fused TP4 HC chain) | full-mesh one-hop exchange |
| HC output projection | `sm70_hcx_output_projection=false` | the fused o_proj is **0.6 ms/round slower** at C1 on this model |
| QSA verify | `sm70_qsa_shared_key`, `sm70_qsa_device_history` | five verify queries share key reads; history read directly |
| KV history | `qsa_host_kv` FP16 (target and draft), hot cache = `MAX_MODEL_LEN` tokens/layer | `KV_PLACEMENT=host` (default) pinned host memory; `device` keeps it on GPU |
| PLE n-gram tables | file-backed (disk) | SM70 Qwen3.8 default; no 26 GiB resident table |
| Startup | `device_transcode`, persistent compile/Triton/GEMM caches | see below |

FP16 history keeps verification outputs bit-identical to device FP16 KV.

## Startup

Measured on 4× V100 (`Model loading took` / engine ready, warm caches):

| | main before #1145 | this configuration |
| --- | ---: | ---: |
| Weight loading | 706 s | 113 s |
| Engine ready | 909 s | 249 s |

The first start also compiles graphs and tunes GEMMs into `CACHE_DIR`; later
starts reuse them.

## Measured latency

Host 54633: 4× V100-SXM2-32GB, full NVLink, 1530 MHz application clock,
185 W power cap, driver 580.173.02, CUDA 12.8, Torch 2.10.

Verification round time (`benchmarks/benchmark_flashnext_acceptance.py
--probe`, 8192-token prompt, 256 tokens, one stream, device FP16 history):

| Configuration | C1 ms/round |
| --- | ---: |
| HCX only (historical 17.4 ms contract; 17.4 was measured on another host) | 19.36 |
| + HCX local schedule (#1129) | 18.92 |
| + QSA shared key and direct device history | 18.29 |
| all of the above | **18.18** |
| all of the above, history in host memory | 19.24 |
| + fused HCX output projection | 20.00 (not enabled) |

All configurations emit identical C1 tokens (4.89 tokens per round).

Service (`serve.sh` defaults, host-memory history, 32K hot cache, one stream,
256 generated tokens, `bench_client.py`):

| Prompt tokens | TTFT | Prefill tok/s | Decode tok/s |
| ---: | ---: | ---: | ---: |
| 988 | 0.6 s | 1,690 | 120–170 |
| 7,801 | 4.5 s | 1,740 | 106–158 |
| 29,222 | 17.3 s | 1,690 | 108–111 |

Decode speed follows MTP acceptance, which depends on the prompt.

## Long prompts and the hot cache

With host-memory history, each QSA layer keeps `qsa_host_kv_hot_tokens`
FP16 K/V entries on the GPU. Prompts longer than that re-stage history from
host memory for every 32-query tile and prefill turns superlinear. `serve.sh`
therefore sizes the hot cache to `MAX_MODEL_LEN` (1 KiB/token/layer, about
0.4 GiB per GPU at 32K). Single-request time to first token, 4× V100:

| Prompt tokens | hot cache 8192 | hot cache 32768 |
| ---: | ---: | ---: |
| 8,192 | 4.6 s | 4.6 s |
| 16,384 | 30.2 s | 9.3 s |
| 30,000 | 101.3 s | 16.9 s |

## Notes and limits

- Requires a build whose `KernelConfig` has `sm70_hcx_local_schedule` for
  `HCX_LOCAL_SCHEDULE=1` (#1129); without it leave the switch at 0.
- `KV_PLACEMENT=host` disables the direct device-history read, which costs
  about 1 ms/round versus `device`; use `device` when GPU memory allows.
- Clocks and power limits change absolute numbers; record
  `nvidia-smi -q -d CLOCK,POWER` with any measurement.
- Known issues:
    - `PREFILL_CHUNK=8192` runs out of memory on 32 GB V100s with this model;
    keep 2048.
    - The host-history hot cache is allocated after the KV budget is profiled
    (not counted by `--gpu-memory-utilization`); `GPU_UTIL=0.92` leaves room
    for a 32K hot cache.
    - The first requests at new batch shapes trigger Triton JIT compiles
    (seconds each); send a short warmup burst after startup.
    - Several concurrent long prompts serialize behind chunked prefill: with
    four 8K prompts the last stream waits for the earlier prefills, and
    decode of running streams slows while prefill chunks are mixed in.
