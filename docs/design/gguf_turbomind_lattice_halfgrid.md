# IQ3 shared FP16 codebook experiment

IQ3_XXS and IQ3_S currently reconstruct four FP16 pairs from biased codebook
bytes for each eight-weight fragment. This experiment expands the small
codebook into FP16 during CTA initialization, then reads two aligned 64-bit
rows and restores signs/scales in registers. Canonical weight storage,
activation precision and FP32 accumulation stay the same.

The official IQ3_XXS table contains 1024 values in [4,62], and IQ3_S contains
2048 values in [1,15]. Every value is exactly representable in FP16, with
zero conversion error. Shared storage increases from 1024 to 2048 bytes and
from 2048 to 4096 bytes respectively. IQ1 and IQ2 retain their byte tables.

Existing lattice tests cover all seven codecs, exact canonical reconstruction,
dense/batch/prefill/grouped operators, graph replay and dispatch boundaries.
GPU validation and actual Flash-Next TP4 timing remain pending. The experiment
will be retained only if those measurements justify its shared-memory cost.
