# SM70 verifier input projections

On TP4 V100, a single-request DFlash2 verifier can combine the channel FP8
QKVZ projection with the FP16 b/a projections and write contiguous QKV, Z,
b and a outputs directly. The M8 QKVZ accumulation schedule stays unchanged.
Weights and scales retain their loaded precision. Other geometries, LoRA,
and other devices use the existing projection path.

The b/a reduction order requires numerical validation against dense FP32 or
FP64 evaluation before promotion. This change is under evaluation.
