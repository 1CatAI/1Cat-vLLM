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
| KV history | `qsa_host_kv` FP16 (target and draft), 8192 hot tokens/layer | `KV_PLACEMENT=host` (default) pinned host memory; `device` keeps it on GPU |
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

See the tables filled in by the validation PR (C1: one stream, C4: four
streams).

## Notes and limits

- Requires a build whose `KernelConfig` has `sm70_hcx_local_schedule` for
  `HCX_LOCAL_SCHEDULE=1` (#1129); without it leave the switch at 0.
- `KV_PLACEMENT=host` disables the direct device-history read, which costs
  about 1 ms/round versus `device`; use `device` when GPU memory allows.
- Clocks and power limits change absolute numbers; record
  `nvidia-smi -q -d CLOCK,POWER` with any measurement.
