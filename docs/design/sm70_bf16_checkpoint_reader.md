# BF16 checkpoint weights on Volta

FP16 activations do not require casting checkpoint weights to FP16. The
Flash-Next MTP checkpoint contains 31 BF16 tensors, totaling 5,214,301,696
bytes. A streaming check found 583,979 values that change when converted to
FP16. The IQ3_S target contains 1,527,889,920 bytes of BF16 tensors; 106,591
values change, including 64,633 HC values. The largest target conversion error
is 2.9802322387695312e-8. These are finite small values, rather than overflow.

A baseline small-M linear reader keeps the original two-byte BF16 storage,
expands its bits directly to FP32 registers, performs FP32 FMA and reduction,
and rounds only at the existing FP16 activation boundary. A warp owns one
output column and reuses each weight across M1..5 rows. It handles column and
K tails without padding the checkpoint. This is a correctness/reference
implementation; no speedup or model-level default is established yet.

The underflow regression uses BF16 weights below the smallest FP16 subnormal
and finite FP16 activations whose products are visible in the output. A
checkpoint cast would produce zero; the direct reader must match the FP64
reference. Further checks cover CUDA graph replay and a dense FP64 dot-product
reference. Operator tests use constructed inputs for numerical coverage and
make no performance claim. Timed measurements require real prompt activations,
cold L2 and the actual TP4 shapes.

The seven numerical cases pass on V100 in the independent research extension.
They cover M1..5 underflow preservation, changing graph inputs and two FP64
dot-product checks. A normal installed build remains required before model
integration or performance qualification.

MoE grouped reading, checkpoint allocation, HC and norm integration, real
activation measurements, and model distribution/acceptance checks are
pending. Model integration must preserve the original BF16 parameters; it
must not use automatic Half checkpoint conversion as a baseline.
