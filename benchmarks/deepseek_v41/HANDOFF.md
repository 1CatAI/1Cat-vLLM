# DeepSeek-V4.1-Flash-1017 on 4× V100-32G: handoff (phases 1–3)

**Hardware:** 54633, 4× V100-SXM2 32G with NVLink. All four GPUs sit behind one x16 Gen3 uplink (12.2 GB/s aggregate). Host RAM is 62 GB; the compute container's memcg is 54 GiB.

**Model:**

- ISTA-DASLab `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS` (138 GB)
- Engram IQ4_NL (110.6 GB)
- DSpark5 draft from the base repository

## Launch

```bash
# once: build the v41k CUDA extension (inside the ds41 container)
.venv/bin/python benchmarks/deepseek_v41/v41k/build.py   # -> benchmarks/deepseek_v41/v41k/build/v41k.so

# API with 512K context, packed FP4 KV, session prefix cache, tools, thinking, DSpark5
docker exec -d ds41-compute-20261024 bash benchmarks/deepseek_v41/tools/run_serve3.sh serve3
# public exposure on 54633: forward 127.0.0.1:8000 (1cat-tunnel preset DFLASH2-API) -> container :18000
python3 benchmarks/deepseek_v41/tools/tcp_forward.py 8000 <container-ip> 18000
```

### Configs

`serve-ds41-phase3.yaml`:

- chunk 8192, max-model-len 524288
- KV 256 MiB (window layers only)
- offload 0.33, prefill group 4096
- prefix caching on, tool parser `deepseek_v41`, reasoning parser `deepseek_v4`
- default `{"thinking": true}`
- `api-key`

Requests can pass `chat_template_kwargs: {"thinking": false}` or `reasoning_effort`.

`serve-ds41-phase2.yaml` is the 16K, non-packed variant (`run_serve2.sh`).

`run_serve3.sh` sets these env vars:

- `VLLM_V41_DOWN_ROWS=1`
- `VLLM_V41_SESSION_PREFIX=1`
- `VLLM_V41_PACKED_KV=1`
- `VLLM_V41_INDEX_TILE_MB=32`

## Results

| | Phase 1 (rel1) | Now |
|---|---:|---:|
| TTFT, short prompt (≤100 tok) | 5.6 s | 0.8–1.1 s |
| TTFT 2K / 8K / 15K | 5.7 / 27.7 / 44.0 s | 4.2 / 5.8 / 11.1 s |
| TTFT 32K / 128K / 500K (packed) | n/a (16K max) | 25 / 110 / 651 s |
| Follow-up turn TTFT (session prefix hit, 4K–500K) | full prefill | 0.9–1.7 s |
| Fresh-text decode, 24 prompts (`real_decode --corpus fresh`) | 57.8 ms/round, 50.8 tok/s | 53.5 ms (55.0 packed+512K), 54.5 tok/s |
| Warm decode round | 24.9 ms | about 25 ms |
| Steady VRAM allocation | 28.6 GiB | 26.9 GiB |
| KV per token | 54 KB | 12.4 KB paged (packed: about 1 KB GPU + host) |

Other results:

- Quality: e2e 5/5 including the 16K context check.
- At ≤16K, packed KV gives tokens identical to the paged FP16 cache.
- Needle retrieval passes at 4K, 32K, 128K and 500K.
- API soak: 80/80 OK.

## Changes (switch, default)

### G1: prefill activation memory

- MoE down output goes to a per-tile scratch plus an FP32 token accumulator.
- `expand_hc` runs in place.
- `hc_transition` is windowed (`VLLM_V41_HC_WINDOW`).
- DSpark aux is stored as pre-scaled FP16.
- Large all-reduces run in place (`VLLM_INPLACE_AR_MIN`).
- `empty_cache` runs before prefill passes of ≥2048 tokens (`VLLM_V41_EMPTY_CACHE_TOKENS`).
- `all_gather` goes through pynccl (`VLLM_PYNCCL_ALLGATHER`).
- Chunks up to 16384.

### G2: cold-expert staging ring

- `VLLM_V41_STAGE_RING=1`, `VLLM_V41_STAGE_SLOTS=128`.
- Stages cold experts only, issued ahead across layers.
- The host never blocks on a pinned row.

### G3: down projection split by rows (`VLLM_V41_DOWN_ROWS`)

- Each rank owns 1280 output rows with full K, so there are no duplicated TP boundary blocks.
- Saves 2.25 GiB per GPU and about 10% of PCIe bytes.
- Decode: `gate_up7` → IPC `peer_gather` (about 8 µs) → `down_rows`.
- Prefill: P2P h-row banks plus a per-tile NCCL fence.

### G4: KV

- **Window rings** (`VLLM_V41_WINDOW_RING`): a 256-row private ring per target layer ≥ 1.
- **Packed KV** (`VLLM_V41_PACKED_KV`; `kv_pack.py`, `kv_store.py`), lossless:
    - Compressed latents are FP4 + E4M3 scales (288 B/row). They live once in a shared host memfd (rank 0 writes) plus a 32K-position GPU hot ring.
    - Index keys are FP4 + power-of-two scales (68 B/row), stored on the GPU.
    - Decode: selected-row buffer, CUDA `index_scores_decode`, and segment-parallel exact top-k.
    - Prefill: staging plus an unpack → cuBLAS indexer.
- **RoPE:** positions past the table rotate on the fly.
- **Limits:** C2, Engram and the guard are lifted to 512K. The DSpark draft length cap is lifted (`speculative.py`).

### G5: prefill compute

- Engram reader threads: `VLLM_V41_ENGRAM_READERS` = 64.
- Adaptive indexer tile: `VLLM_V41_INDEX_TILE_MB`.
- Persistent-grid activation: `VLLM_V41_ACT_PROGRAMS`.

### G6: short prompts (`VLLM_V41_ROUTED_PREFILL_MAX` = 1536)

- Only routed experts are transcoded or staged, with one sync per layer.

### G7: same-session prefix cache (`VLLM_V41_SESSION_PREFIX`)

- The scheduler accepts a hit only when the new request continues the last executed sequence, the hit position is even, and processed − 128 ≤ hit ≤ processed.
- The worker's `restore_prefix` rewinds the Engram history and the C2 positions.

## Tests and tools

- **`benchmarks/deepseek_v41/tests/`**: GPU micro-tests for each change (bit-exact or equivalent).
    - `test_kv_pack.py`, `test_packed_attn.py`, `test_index_packed.py`, `test_topk_par.py`
    - `test_window_ring.py`, `test_g3_down.py`, `test_peer_gather.py`, `test_g1mem.py`, `test_rope_tail.py`
    - Run: `PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python <test>`.
- **`benchmarks/deepseek_v41/tools/`**:
    - `v41_e2e.py` / `run_e2e2.sh`: full e2e.
    - `real_decode.py` / `run_real.sh`: fresh-text decode.
    - `prefill_profile.py` / `run_prefill_profile.sh`: prefill segment timers, enabled with `VLLM_V41_PREFILL_TIMING=1 VLLM_V41_ATTN_TIMING=1`.
    - `session_test.py`, `longctx_test.py` (needle, TTFT, follow-up decode).
    - `api_multiturn.py`, `api_think.py`, `api_soak.py`.
    - `run_decode_spy.sh`: py-spy.

## Dead ends (kept, default off)

- Fused flash-style sparse attention (`VLLM_V41_FUSED_SPARSE`).
- Token-blocked HC (`VLLM_V41_HC_TOKEN_BLOCKED`).
- Exact per-tile GEMM sizes (`VLLM_V41_EXACT_GEMM`).
- TurboMind tuning pass (`VLLM_V41_TUNE_PREFILL_PASSES`).
- Prefill group 2048.

Triton `tl.dot` on sm70 falls back to FMA, so heavy dots need CUDA or cuBLAS.

## Open items

1. **Packed KV at 500K:** minimum free VRAM drops to 25 MiB during 500K prefill.
   - Cause: fp16 unpack of all index keys, plus staging.
   - Fix: score keys in tiles; keep staging persistent and copy only the increment.
2. **Deep-context prefill:** non-MoE time grows from 4.4 s to 10+ s per 8K chunk, from the indexer O(T·N) work and staging.
3. **Thinking-mode prefix cache:** with thinking on, `drop_thinking` removes earlier reasoning from the history, so follow-up hits land at the previous prompt end. That is outside the ring reach, so a full prefill runs.
   - Fix: snapshot the window rings at each prompt end (about 10 MB).
4. **Fresh-text decode:** 27 cold-expert misses per round dominate. Lossless levers are a better hit rate, prefetch accuracy and per-miss cost.
5. **Sync scheduling only (`v41_e2e`):** about 4 ms per round of host work is not overlapped. Serve uses async.
