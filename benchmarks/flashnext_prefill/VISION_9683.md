# Flash-Next 9683 image input

This change depends on the memory/runtime branch in PR #1210, measured at
`81edd142bd1d48e8fa428d24f17b034348d20ade` and documented at
`1e4f098828aec70214288791b1fb4d603e4fe60d`. The declared task integration line
is that owned branch; the final project integration line remains `onecat/main`.

The native GGUF adapter continues to own the language checkpoint. A conditional
model declares its language module scope so expert-arena storage, prepared linear
layouts and weight loading use the same underlying language model. The wrapper
delegates dense/HC preparation to that owner. The existing Qwen3-VL tower loads
unquantized weights through its normal loader, without changing the expert
quantization, GPU fraction, host KV, MTP or language attention families.

`GGUFModelLoader` accepts one optional loader argument:

```json
{"multimodal_safetensors":"/path/to/visual.safetensors"}
```

The local companion file must contain exactly the active towers' parameters,
after the model's checkpoint name mapping. Missing/extra parameters, inactive
towers, non-tower overrides and unknown loader options reject explicitly. This
qualifies the native Qwen4Exp wrapper with its matching vision checkpoint; it
does not qualify arbitrary mixed checkpoints or every multimodal architecture.
Legacy GGUF loading without this option retains its existing path.

Set `VISION_DIR` when using `examples/serve_flashnext_gguf_256k.sh`. It must hold
the matching full HF config, processor/tokenizer assets and `visual.safetensors`.
The default remains text-only when this variable is absent. Image mode retains
262144 total context, prefix caching, thinking/tools, TP4, MTP4, FULL decode
graphs, 80% GPU experts, HC sequence parallelism and the 512 MiB allocator reserve.

The launcher uses the largest signed 32-bit count for images so an independent
per-request image count cannot constrain any request within the total context.
It disables video input, which is outside this deployment's requested scope.
`--skip-mm-profiling` avoids synthesizing that count during startup; it is not
evidence of runtime memory safety. Encoder weights are sharded across TP4;
processor caches are bounded to 0.25 GiB per process to preserve host headroom.
Image tokens, aggregate payload size, encoder activations and available memory
still constrain requests. The existing proxy's body limit also remains in force.

The optional `--mm-encoder-batch-size` bounds items per runtime encoder forward
in both worker runners, before input transfer, without dropping or reordering
request items. Its default 0 preserves unrestricted batching. Positive values
participate in the multimodal configuration hash; language-only mode ignores
the inactive setting. Startup dummy profiling remains conservative and is not
reduced by this runtime bound. The deployment uses one image per forward and
`max_pixels=4194304`, automatically resizing larger images while preserving all
request images. These are memory policies, not an increased image-count limit.

The official `Qwen/Qwen3.8-Flash-Next` visual tensors were fetched directly on
9683 using only HF-mirror byte ranges. The companion contains 333 original BF16
tensors, 897862112 payload bytes, without quantization or numerical conversion.
Its SHA256 is
`d03e3c31d4975a86a0b4579658f73a0f99b23cd720510ced524428a362abe69f`.
The original shard header, range metadata and manifest are retained with it.

CPU ownership/loader/configuration gates: 76 passed. Two stale fixture groups
also fail on the unmodified parent; fixtures were updated with the existing DCP
field and existing PLE packed-result preparation stub. Real tower parameter
layout was constructed on meta for TP4: 220.671 MiB per rank, all 333 names match
and every real weight remains finite when converted to the deployment FP16 dtype.
Meta construction is not native execution or an end-to-end vision quality gate.
TP4 native tower execution passed 1/4/17/64/256 images at 256 pixels and a
1536-pixel single image, with bit-exact outputs against the normal unquantized
vision loader on every rank. The selected native backend is TORCH_SDPA. A single
256-image forward needs about 1 GiB of temporary activations, motivating the
runtime microbatch bound above. Bounded-forward and public quality checks remain
pending.
