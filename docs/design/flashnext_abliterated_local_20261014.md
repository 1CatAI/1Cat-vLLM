# Local Flash-Next IQ3_S abliterated deployment

This deployment combines the latest integration base `6f3557c973`, the decode
implementation from PR #1204 at `46f5c8c197`, and the qualified prefill
implementation from PR #1205 at `3faaa58d08`. Later prefill experiments remain
outside this deployment.

The model is
[SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF),
IQ3_S, revision `eec4e10a2c29b440a2ae85591fa11569ce752963`. Both GGUF shards
are downloaded through `https://hf-mirror.com`, with resumable ranges and the
repository's SHA256 values checked before serving. The initial environment-only
proxy bypass did not bypass the host's transparent TUN. Subsequent requests use
real DNS and sockets bound to the physical NIC; the routing mistake and corrected
connection evidence are retained in the external report.
The matching stock BF16 MTP tensors are copied from the existing local
checkpoint unchanged and verified against the previously qualified compact
checkpoint hash. The runtime prepares its existing Q8_0 MTP expert route.
The repository's standalone Q8_0 GGUF draft file is also retained locally.

## Runtime contract

Use [the launcher](../../examples/serve_flashnext_gguf_256k.sh) from its owning
source checkout and an environment with the matching native extensions.

- Four V100-SXM2-16GB GPUs, physical GPUs 4–7, TP4, one active sequence.
- Python 3.12.13, Torch 2.10.0+cu128, Triton 3.6.0, CUDA toolkit 12.8.
- Capacity is **262144 total input and output tokens**, with chunk size 8192.
- Prefix caching, Qwen3 reasoning and Qwen3-Coder tool calling enabled.
- Temperature 0.6, top-p 0.95, top-k 20; thinking enabled.
- MTP4, greedy draft, FULL target graphs at M5, existing draft single graph.
- Original experts retain 80% GPU / 20% host placement. PLE remains on disk.
- Authoritative target KV uses host E4M3. Encoded target hot capacity is 32768;
  the independent draft cache retains FP16 and 16384 tokens.
- The accepted grouped-prefill and dense restoration policy, cold-expert DMA,
  balanced chunks and host-KV zeroing skip are enabled through existing paths.
- Rejected HC/PLE, indexer, tile and larger-chunk screens are not added.
- Native components are built by normal CMake targets into the source package.
  No private native sidecar, source overlay or preload is required.
- Compiler caches and the service's memory guard are owned by this deployment.

These settings preserve existing runtime support and fallback boundaries.
They do not qualify a different TP size, concurrent requests, all quantization
tiers, or every length within the configured capacity.

## Reproduce

Set an API key and the local paths before launching:

```bash
export MODEL_FILE=/path/to/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf
export MTP_DIR=/path/to/mtp-stock-bf16
export CACHE_DIR=/path/to/dedicated-runtime-cache
export PYTHON_BIN=/path/to/venv/bin/python
export CUDA_HOME=/path/to/cuda-12.8
export VLLM_API_KEY='your-key'
export GPU_IDS=4,5,6,7 CPU_SET=24-47,72-95
bash examples/serve_flashnext_gguf_256k.sh
```

Acquire the selected GPUs' locks and start the memory guard as part of the
service manager. The local deployment retains the existing user service and
public tunnel, with a dedicated internal API port. Public forwarding is enabled
after authenticated generation, tools, prefix reuse and capacity checks pass.
Credentials stay outside the source repository.

## Evidence

The prefill dependency's original-model TP4 screen measured **5010 tokens/s**
at 34688 input tokens, median TTFT 6.907 s. The decode dependency's original
32K screen measured conservative mean verifier cost 15.494 ms. These are
dependency results on the 9683 host, not performance measurements of this
abliterated model on the local machine.

The combined build passes 81 focused GPU cases covering encoded/protected KV,
expert-cache swaps, grouped-prefill boundaries and segment-prefill restoration.
All twelve normal native components are built and their library search paths
are audited. Route checks, unprofiled prefill/decode, tools, prefix reuse
and long-context results are recorded after validation. Raw download journals,
checksums, builds and model responses are retained in the task's external
`flashnext-abliterated-local-20261014` report directory.
