# Higgs-Audio-v3 TTS on Tesla V100 (SM70)

Runs [`bosonai/higgs-audio-v3-tts-4b`](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)
text-to-speech on a single Tesla V100 with the `FLASH_ATTN_V100` backend and the
Stage-0 CUDA graph (low-latency) profile, reaching **real-time** generation
(RTF ~1.0 — about 2.4x faster than the eager profile).

## Requirements

- **1Cat-vLLM** with the SM70 decode CUDA graph kernel **>= `e64d39aa7`**
  ("Stabilize SM70 Qwen MTP paths"). Earlier kernels cap the scalar-paged decode
  workspace at the capture-time `seq_len`, so the talker CUDA graph replays a
  short/stale KV span and produces incorrect audio.
- **vllm-omni** with the talker CUDA-graph-capture fix
  ([vllm-project/vllm-omni#4563](https://github.com/vllm-project/vllm-omni/pull/4563)).
  Without it, Stage-0 capture aborts with
  `operation not permitted when stream is capturing`.

## Run

```bash
python examples/generate/multimodal/higgs_audio_v3/tts.py \
    --text "Hello! This is Higgs Audio version three, generating speech on a Tesla V100." \
    --deploy-config examples/generate/multimodal/higgs_audio_v3/higgs_v100_low_latency.yaml \
    --out higgs_out.wav
```

For the **eager baseline** (correct audio, no CUDA graph — works without the two
fixes above), set Stage-0 `enforce_eager: true` in the deploy config.

## Notes

- Stage 0 (talker) uses `FLASH_ATTN_V100` + `FULL_DECODE_ONLY` CUDA graph in
  `float16`; Stage 1 (code2wav) stays `enforce_eager: true` in `float32`.
- Verified on a V100: the generated audio transcribes back to the input prompt.
