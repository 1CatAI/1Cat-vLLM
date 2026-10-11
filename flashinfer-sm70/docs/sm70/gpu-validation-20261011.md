# V100 hardware validation, 2026-10-11

The user supplied machine 54633 and authorized stopping its Studio inference
service. At validation start that service was already absent and all four V100s
were idle. This supersedes the earlier GPU deferral. Delivery is consolidated
in Draft PR #1177 against `onecat/main`; #1175 is closed with its commits retained.

## Independent installed artifact

Library source: `564e536ea5`, integration/benchmark source: `b41c0c7aa884`.
The only test change after that source is the startup-namespace correction below.
Main integration base remains `be3722bc095b9792bf9833068595f0b1ef70ad37`.

- Python 3.12.14; Torch `2.10.0+cu128`; TVM-FFI `0.1.10`.
- CUDA toolkit `12.8.93`, driver `580.173.02`, Tesla V100-SXM2-32GB, SM70.
- A fresh operator environment has no vLLM. It installs the ordinary source/JIT
  wheel, then compiles shipped sources into its own cache with CUDA 12.8.
- Wheel: `flashinfer_python-0.7.2+1cat.sm70.2-py3-none-any.whl`.
  SHA256: `b18477c8b332de3fcb07a8fbae05340cdbbc67411fd01ff63afd97bac0fe279a`.
- No `PYTHONPATH`, `LD_PRELOAD`, private `LD_LIBRARY_PATH`, borrowed engine DSO,
  source overlay, or engine-private extension is used.
- GPU 2 was held with the shared ownership lock during tests. Other tasks' builds and GPU
  processes are preserved. The P400 display device is excluded.

The loaded CUDA 12.8 artifact identities are:

| Component | Source SHA256 | Native library SHA256 |
| --- | --- | --- |
| basics | `b5cf0cf71251350587e2842cef29912fb2cf14bf9a9d89589808921c3e4823a8` | `d784b3afebd73318e37faefa226dbb8dd7210f1d645170acd16c8aa80ceb7cb6` |
| qsa | `9b83a952aa60a5e39d54d393d097d8b646aac788db0cb85dc4e220b6682838a8` | `f036fcfd77b30540070038d1079fe398f3dfadf2a5437bcd5e5f5c1df592c5ff` |

## Correctness result

**40 passed in 10.86 seconds**: 9 packaging/ownership tests, 16 GPU primitive
cases and 15 GPU QSA cases. Dependency consistency passes for all 38 installed
packages. Commands, from a directory containing the standalone tests:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=7.0
.venv-operators/bin/python -I -c \
  'from flashinfer.sm70 import prepare; prepare("basics"); prepare("qsa")'
.venv-operators/bin/python -I -m pytest -x -v \
  -c tests/sm70/pytest.ini --confcutdir=tests/sm70 tests/sm70 \
  --junitxml=artifacts/operator-gpu-tests.xml
uv pip check --python .venv-operators/bin/python
```

RMSNorm/Gemma widths 72/128/2560/8192, residual rounding, activations and partial
strided/in-place RoPE pass. QSA covers C1/C4, M1/M5/M20/M32, strided layouts,
interleaved requests, page size 204, invalid physical pages, ties, numerical
extremes, int32/int64 positions, empty/padded rows, changing replay metadata,
independent captures and cache rebinding. The FP64 reference retains the existing
`rtol=atol=2e-6`; selected IDs and visibility are exact. No tolerances changed.

An expected warning occurs when testing rejection of plan allocation inside an
otherwise empty CUDA graph. It is not a failed capture of an execution plan.

The first run stopped at the fresh-process lazy-import assertion: NVIDIA's
`cuda-bindings` startup `.pth` creates the `cuda` namespace before FlashInfer is
imported. A before/after module inventory proves FlashInfer adds no backend
modules. The test now checks newly imported modules, including backend submodules,
rather than attributing interpreter startup work to this library. The failed log
and JUnit record are retained alongside the passing run; no production change
was needed for this failure.

## Normal engine build and driver recovery

The normal engine wheel from source `b41c0c7aa884` built and installed successfully:
`1cat_vllm-1.5.2+fi.sm70.b41c0c7-cp312-cp312-linux_x86_64.whl`, SHA256
`d9a89efa3309fbf710763c2860c0b128713af8c413bfb5438e1541524a7dd898`.
Its 19 native libraries include the engine, QSA indexer, Flash-V100 and FlashQLA.
Native `csrc`, the CMake file and the CMake directory are identical to the main
baseline; no previous task's compiled engine extension is reused. The wheel
and build/test logs are saved on the validation host and locally.

Dependency validation of this first engine artifact found three missing runtime
packages (cuDNN frontend, CUDA 12.8.93 compiler and CUDA 12.8.90 CCCL) and an
unnecessary QuACK requirement. QuACK would reintroduce CuTe DSL transitively;
the SM70 metadata now omits it, with the non-SM70 requirement preserved and both
metadata tests passing. The rebuilt `1.5.2+fi.sm70.f3ad7b684480` wheel has SHA256
`bd5e5a8e6b9dedd84942391528c5c0c984aad778ecf94668ac4928f21694ecf3`.
Its declared dependencies are installed and all 190 engine packages pass
dependency validation. Torch remains `2.10.0+cu128`; no CuTe or QuACK is installed.

During engine compilation the host's system package update replaced NVIDIA
driver libraries with `580.178.04`, while the running kernel still had
`580.173.02`. NVML reports a driver/library mismatch; the new matching kernel
module is present on disk. The first capacity-screen attempt stopped at this
hardware check before executing any GPU benchmark. No timing result exists.
The user approved reboot after compilation and artifact preservation. Both are
complete; the desktop reboot request initially timed out and the system interface
required interactive administrator authentication. A subsequent new boot was
verified with all four V100s on `580.178.04`. The original 40-test record remains
under `pre-driver-update`; all 40 tests passed again in 7.66 seconds after recovery.
Later Studio started another TP4 service; the identified Studio engine service
was stopped under the user's existing authorization before further GPU tests.

## Grouped cancellation follow-up

The `.2` engine screen passed scores and exact selected/expanded indices at 8K
and 32K for M1/M4 decode and M5/M20 verification. At 128K, M1/M4/M5 passed, but
M20 stopped at the unchanged score tolerance before timing that point. A focused
diagnosis over all 12 synthetic layers isolated three near-zero cancellation
cases. Each implementation individually satisfies the FP64 oracle tolerance,
but opposite rounding errors exceed the paired `rtol=atol=2e-6` bound. These
failures are retained; the comparison threshold is not widened.

Three small fixed FP16 fixtures reproduce all three failures with the installed
`.2` library. The `.3` candidate uses two 64-element MMA accumulators in the
grouped route and combines them in FP32; the C1/M<=8 route keeps its original
arithmetic. The fixtures now pass both the FP64 oracle and stored engine control.
The full standalone suite passes **43 tests in 6.53 seconds**, including 34 GPU
cases, with all 38 independent-environment packages consistent and no vLLM.
The source/JIT wheel SHA256 is
`e1c2ff8a757aba97e6fcd779edd4b4e214299c24f26dc3842c80d2e6514fde34`.
Loaded `.3` QSA source SHA256 is
`7bbd0326ea1debb2807bb4c43eab2ad37ac9d53de7cb5055f43c83e56c366770`;
its operator-environment DSO SHA256 is
`2db345401c5d82d7c88659f767660b263ade885bf22431080742e279714848e9`.
The subsequent normal engine artifact `1.5.2+fi.sm70.acc83350d43c` installs this
`.3` provider. Its wheel SHA256 is
`9cc53c6567e1d938d25b894fd26f28d41b14821296c596bbd29050a01f0ca0a1`;
all 190 declared packages pass dependency validation. The installed engine's
QSA DSO SHA256 is
`481b56e295bd16c2fa38310040561108cec5131654ae34ff91e081f9875a1685`.
The library source hash above is unchanged; its native artifact differs from
the operator environment because the independent build/cache paths differ.

## Version .3 paired engine screen

All 24 score/selection points pass at 8K/32K/128K, for M1/M4 ordinary decode and
M5/M20 MTP4 verification. Scores retain `rtol=atol=2e-6`; visible lengths and
selected/expanded IDs match exactly. Repeated control outputs are bit-exact.
The matrix uses the same four command invocations and RNG progression as `.2`:

```bash
python -I benchmarks/kernels/benchmark_sm70_qsa_shared_key.py --flashinfer \
  --rows 5 20 --context 8192 --out qsa-v3-screen-8192.json
python -I benchmarks/kernels/benchmark_sm70_qsa_shared_key.py --flashinfer \
  --rows 1 4 --context 8192 --out qsa-v3-matrix-decode-8192.json
python -I benchmarks/kernels/benchmark_sm70_qsa_shared_key.py --flashinfer \
  --rows 1 4 5 20 --context 32768 --out qsa-v3-matrix-32768.json
python -I benchmarks/kernels/benchmark_sm70_qsa_shared_key.py --flashinfer \
  --rows 1 4 5 20 --context 131072 --out qsa-v3-matrix-131072.json
```

Each point executes 12 independent synthetic TP-local layers, FP16 Q/K and FP32
scores, H4/D128, page size 204, randomized page mapping, seed 902, CUDA graphs,
on GPU 2 after the driver recovery. Both arms use the same ordinary installed
engine. Control selects the existing best scorer, including native C1 sharing;
candidate forces the independently installed library. Five ABBA blocks retain
ten samples per arm, each of 30 graph replays. The table measures the entire
**score + exact Top-K + expansion portion**, not a sum of kernel timings.
Times are milliseconds across all 12 layers. Negative changes mean less time.

| Context | Rows | Control ms | Candidate ms | Median change | Paired block changes | Absolute A/A change |
| --- | --- | --- | --- | --- | --- | --- |
| 8192 | 1 | 0.225161 | 0.221884 | -1.46% | -1.95 to -1.33% | 0.10% |
| 8192 | 4 | 0.477474 | 0.462199 | -3.20% | -4.17 to -2.95% | 0.18% |
| 8192 | 5 | 0.261598 | 0.264397 | +1.07% | +0.91 to +1.31% | 0.17% |
| 8192 | 20 | 0.957338 | 0.562790 | -41.21% | -42.02 to -41.09% | 0.02% |
| 32768 | 1 | 0.652151 | 0.532497 | -18.35% | -18.42 to -18.27% | 0.03% |
| 32768 | 4 | 0.996267 | 0.764228 | -23.29% | -24.06 to -23.12% | 0.06% |
| 32768 | 5 | 0.607147 | 0.609075 | +0.32% | +0.24 to +0.48% | 0.16% |
| 32768 | 20 | 2.703787 | 0.965956 | -64.27% | -64.39 to -64.23% | 0.01% |
| 131072 | 1 | 1.702963 | 1.177822 | -30.84% | -30.85 to -30.79% | 0.03% |
| 131072 | 4 | 2.782498 | 1.959578 | -29.57% | -29.71 to -29.52% | 0.00% |
| 131072 | 5 | 1.350007 | 1.356356 | +0.47% | +0.41 to +0.56% | 0.12% |
| 131072 | 20 | 9.520759 | 2.507964 | -73.66% | -73.69 to -73.58% | 0.01% |

The previously failing 128K/M20 point now passes before timing. C1/M5 retains
its original arithmetic, with a largest partial-chain median regression of
1.07%. The source-complete normal engine, native identity, raw samples, commands,
GPU records and standalone accuracy results are retained locally and on the
host in `qsa-v3-screen-evidence.tar.gz` and `qsa-accuracy-standalone.tar.gz`.
This is not an exact-main whole-model comparison: a separate normal main
artifact from `be3722bc095b` is being built for that next gate.

## Performance and remaining gates

No complete-chain or model speed claim follows from these checks. The `.2`
screen does show a partial-chain improvement for C4/M20: 12 synthetic local
layers of scoring, exact selection and expansion take 0.958805 → 0.562620 ms at
8K and 2.702814 → 0.960785 ms at 32K, reductions of 41.32% and 64.45%. Five paired
ABBA blocks each retain two samples per arm; their reductions are respectively
40.98–41.79% and 64.43–64.54%, with A/A differences below 0.05% at these points.
C1/M5 remains near parity (1.57%, 0.34% and 0.48% slower at 8K/32K/128K).
These numbers belong to `.2`, exclude projection, cache update and attention,
and cannot qualify the `.3` accuracy change or any complete QSA/model path.

A separate `.2` C4/M20 capacity probe (32 versus 20 reserved rows) keeps scores
bit-exact but saves only 1.94% of scoring time, with A/A difference 0.10%.
No extra kernel variant is added for this small isolated result.

Complete QSA, text C1/C4, MTP rollback/acceptance,
D72 vision, HC/MoE/GDN/PLE/output chains and shared Qwen3.6 regressions remain
open. Default selection stays off. Complete-chain and integrated acceptance
remain at least 5% improvement beyond noise, with at most 2% other-point
regression and unchanged numerical/quality criteria.

Raw artifacts are retained in the task's `gpu-54633` bundle: installed package
and loaded-native identities, test logs/JUnit, initial failure, dependency
inventories, GPU topology, dynamic linkage and disassembly. They are not build
or runtime dependencies.
