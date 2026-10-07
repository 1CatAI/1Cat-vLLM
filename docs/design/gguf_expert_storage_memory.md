# GGUF expert storage and GPU memory

The default SM70 expert route keeps canonical banks for TurboMind and retains
original IQ blocks for calibrated small-batch operators. Both representations
occupy GPU memory. A compressed GGUF checkpoint's size divided by TP size does
not include this duplication, replicated dense parameters, MTP, graph storage
or temporary workspaces.

Flash-Next IQ3_S has 17 IQ3_XXS, 20 IQ2_S and 10 IQ3_S expert gate/up layers.
With E512, local N160/K2560 and two matrices per layer, their original-block
banks occupy 6.752 GiB per TP4 rank, in addition to canonical storage. This
calculation includes the original-row alignment bytes and excludes allocator
overhead. PLE's large embedding table uses host storage separately.

`kernel_config.sm70_gguf.expert_storage` selects the expert representation:

- `canonical` is the existing default and retains calibrated TurboMind routes.
- `original` prepares original-block expert banks and uses the packaged native
  fallback. It avoids constructing canonical expert banks and their retained
  original copies. Dense projections keep their existing policy.

Keep `sm70_gguf.enabled` enabled: the original path still requires packaged
operators. Operator capabilities select its supported fallback at preparation
time. This is a memory tradeoff; it does not imply equal inference speed.

Q2_0 experts need special handling when TP4 splits K640 into K160. The existing
adapter converts those blocks losslessly to Q4_1 before slicing. The original
FP16 coefficient and integer values are preserved. No additional weight
quantization is introduced.

CPU loader tests cover both policies and compare the Q2_0/Q4_1 reconstruction
element by element. Actual 4x16GB loading, output quality, available KV memory
and performance are pending GPU validation. A no-MTP capacity test does not
qualify MTP4 or the default fast path on that hardware.

On four 16 GiB V100-SXM2 GPUs, the first original-storage MTP4 run exhausted
memory during target expert preparation, before loading the draft. A safety-tail
allocation copied an entire expert bank (114--126 MiB) when only about 110 MiB
remained free. Native expert banks now allocate their zero safety tail before
checkpoint rows are copied, so preparation reuses the same allocation. The tail
belongs to the final allocation, rather than every row. Tensor bytes and logical
shapes are unchanged. This removes the observed transient copy; it does not yet
establish that the complete MTP4 model fits on those GPUs.
