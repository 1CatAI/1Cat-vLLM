# DFlash split windows with E4M3 hybrid caches

An SM70 hybrid target using E4M3 KV chooses 1648-token pages. Attention cache
specifications refresh the shared block size after loading, so FP16 draft
attention can also receive 1648-token pages. The split-window implementation
already indexes the live page size and K/V strides, but the public adapter only
admits 832/1024/2048. This sends the draft back to general paged attention when
the target changes cache dtype.

Admit 1648 pages for the existing single-request Q8/H8/KV2/D128 FP16 split
window. The per-engine draft-window policy disables the new hybrid page routes.
The native concurrent implementation retains its 1024/2048 page ABI. Unsupported
query counts, dtypes, layouts or unavailable split imports retain the fallback.
There is no new kernel, weight conversion, approximation or environment variable.
Probability, PV and partial reduction remain FP32.

Focused tests extend the FP64 window oracle, page indirection, zero lengths,
live graph lengths, public dispatch without the native symbol and concurrent
fallback. Installed-wheel and model comparisons are pending.
