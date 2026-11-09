# SM70 verifier convolution preprocessing

The fused convolution, gating and output initialization operator accepts five
MTP4 rows and the existing eight-row verifier shape. It preserves the existing
convolution implementation and FP32 gating arithmetic. Padded output stores
are masked by the actual row count.

The convolution's logical history window is `M + 2` for width four, even if
physical cache storage is larger. Physical cache strides remain unchanged.
An initial implementation used the physical length for the logical window;
the five-row, ten-entry cache test caught a state mismatch. The corrected
operator matches the existing convolution state and output bitwise.

## Validation

The ordinary source-containing wheel passes 15 GPU tests: five/eight rows,
contiguous/strided input, accepted-history offsets, shorter/empty sequences,
invalid slots and changed-input/metadata CUDA Graph replay. QKV, convolution
state, FP32 gating and initialized output match bitwise.

Python 3.12.3, Torch 2.10.0+cu128, CUDA 12.8.93, Triton 3.6.0,
V100-SXM2-32GB, FP16 QKV/cache and FP32 gating. The synthetic chain uses 36
distinct layers, QKV row stride 4096 and five samples of 100 warmed graph
replays. It compares output zeroing plus the existing convolution and gating
with the fused operator.

| Rows | Existing chain (us) | Fused chain (us) | Saved (us) |
| --- | ---: | ---: | ---: |
| 5 | 218.358 | 107.827 | 110.531 |
| 8 | 211.333 | 93.993 | 117.340 |

These are operator-chain measurements, not complete model rounds. Model
admission and recurrent execution are unchanged. The Flash-Next MTP path
already has gating inside its recurrent kernel; it must not duplicate that
work when this preprocessing operator is connected.

Measured Python source: `c23a39e0db`. Whole wheel: `1.5.2.dev679+gc23a39e0d`.
Native `_C` SHA256:
`339dbb29f79ed9be61b930f93a38118026be7cb9d129429403a52ff978f4a539`.
Wheel SHA256:
`91b3ec43981117be50a81fa798af12f6ba45cf66d952300789532b643dd47e72`.
No private kernel libraries or preload are required.
