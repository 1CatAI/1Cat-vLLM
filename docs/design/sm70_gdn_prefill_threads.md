# SM70 original GDN prefill launch tuning

`kernel_config.gdn.original_prefill_threads` accepts 128 or 256, with 128 as the
default. It is resolved before worker initialization, serialized with the
existing GDN policy and included in active original-FlashQLA computation hashes.
Other prefill backends and the independent FlashQLA API keep their defaults.

The 256-thread accelerated subset is one indexed request or one direct-state
sequence on SM70, FP16 Q/K/V, FP32 state/gate/beta, local H4/HV12 and DK=DV128,
with 4096–8192 query rows. A direct state has leading dimension one and two
cumulative sequence offsets.
These are rows in one prefill chunk, not the total context length. Unsupported
tensors use the existing 128-thread core. State slots, causal processing,
prefix checkpoints, direct output aliases and decode remain in their existing
families. The ordinary serve/offline configuration uses the same admission.

For an explicit candidate run, add this field to the existing kernel config:

```json
{"gdn": {"original_prefill_threads": 256}}
```

The bundled original TileLang factory accepts a thread-count argument.
Serving calls that public API directly; no private factory, native override,
import-order dependency or process-global math switch is used.

## Measured operator scope

The screen uses four concurrent V100-SXM2-16GB GPUs at 1312 MHz, CUDA 12.8,
Torch 2.10.0+cu128, Python 3.12.13 and normalized synthetic FP16 operands.
Each call restores FP32 state; state reset is included in timing. Indexed
state timings are below; direct state repeated controls at 7344/8192 rows are
1.382/1.537 ms, versus 1.030/1.143 ms with 256 threads.

| Query rows | Repeated128-thread control (ms) |256-thread core (ms) |
| ---: | ---: | ---: |
|7344 |1.578 |1.079 |
|8192 |1.749 |1.127 |
|6108 |1.317 |0.857 |
|6528 |1.392 |0.908 |

Rows4095/4096/4097/8193 also pass paired numerical checks. All measured outputs,
final state and untouched cache slots are exact. The default remains128;
these operator times do not establish complete-model TTFT or general model
quality/performance coverage.

Eight ordinary API GPU regressions cover indexed and direct first/continued
state, absent/zero/random initial state, ragged rows4097/6108 and private
untouched slots. Policy tests
cover worker snapshots, active/inactive hashes, threshold neighbors and
unqualified tensor fallback. The real Flash-Next TP4 serving A/B and task
checks are recorded in the prefill campaign report before promotion. The local
IQ3_S TP4/MTP4 run at chunk8192 and 256K capacity executes 256 threads and
measures 5.831594 s / 5941.04 actual prompt tok/s with the HC candidate and
the original uniform expert tuner;
five prompt hashes/counts match control and the 32K task checks pass. That
combined result includes earlier HC improvements. GDN drops about 63 ms in
the route trace; the experimental profile tuner was rejected and removed.
The default remains 128. No 128K/near-256K paired task qualification or
8000 tok/s result is claimed.

```bash
python -m pytest -q tests/config/test_gdn_original_threads.py
python -m pytest -q tests/kernels/mamba/test_flashqla_original_threads.py
```
