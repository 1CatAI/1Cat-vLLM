# Two-reader shared-activation gated pair prototype

This research prototype separates the measured M8/N32 activation and MMA
schedule from its weight readers. IQ4_XS/IQ3_S and IQ3_S/IQ4_XS are both
instantiated. Their eleven candidate layers remain on the canonical path:
this source has not been run as a GEMM or selected by a model.

## Reader and skeleton interfaces

`gguf_native_pair_readers_sm70.cuh` defines `NativePairReader<21>` and
`NativePairReader<23>` with the same contract:

| Interface | Responsibility |
| --- | --- |
| `kBlockBytes` | Original source bytes per K256 block: 110 or 136 |
| `kBookId`, `kBookBytes` | Codebook identity and shared-memory requirement; IQ3_S has a 16 KiB book, IQ4_XS uses the existing register LUT |
| Constructor `(source, tile, blocks_k, first_part, col)` | Resolve record and metadata pointers once, before the K loop |
| `initialize(book)` | Initialize a required book with all CTA threads participating |
| `load()` | Read the current K128 record, cache original metadata once per K256, advance pointers and preserve its half-block identity |
| `fragment<Segment,Fragment>(record,book)` | Return one logical eight-weight Half operand through the existing public decoder |

Records belong to each format and remain distinct. IQ3_S retains its
52-byte signed-index record and original d/scale fields. IQ4_XS retains four
aligned 16-byte nibble packets and original d/high/low scale fields. IQ4_XS
scale multiplication remains FP32, followed by final Half operand rounding.
IQ3_S uses the already verified public signed-book operand decoder unchanged.
The fixed 13-bit field extractor is now a shared method of that decoder;
there is no second lookup or weight-decoding formula.

`gguf_pair_shared_a_sm70.cuh` provides
`native_pair_shared_a_kernel<GateReader,UpReader>`. It preserves:

- N32, M8, eight K warps per projection and the `[8][8][136 half]` A tile.
- Unique coalesced activation loads shared by gate and up, with the same two
  barriers around each K128 use.
- FP32 `mma884` accumulation, FP32 partials and the original eight-part
  reduction order.
- Half projection rounding, the same gated SiLU epilogue, and one output
  store without concatenation or a separate reduction launch.

Only the active projection's typed reader is constructed in a reader-state
union. Its branch has no CTA barrier; A staging barriers remain outside the
projection branch. Identical codebooks share one initialization and pointer.
Different books receive disjoint offsets. After every warp completes MMA,
a barrier protects the book/partials union lifetime change. This interface
allows another format to supply a reader without replacing the CTA schedule.

The wrapper `gguf_mixed_pair_research_sm70.cu` is registered only with
`GGUF_MIXED_PAIR_RESEARCH`. It checks both source-byte counts independently
and accepts only M8/N32/K1024 and the two specific type orientations. It is
not included in the normal build or the model/kernel route selector.

## Current verification boundary

CUDA 12.8 SM70 compilation with Torch 2.10 headers succeeds for both
orientations. Each uses 64 registers, zero stack/spill bytes and 33,792 bytes
of shared memory under `__launch_bounds__(512,2)`. Preferred carveout is 100.
The full static K128 loop includes both mutually exclusive reader branches:
1,879 instructions for each orientation. These combined
counts must not be divided into an executed-warp instruction count or used
as a performance estimate.

Moving the fixed IQ3_S index extractor into the public header produces a
complete SASS instruction listing identical to the prior standalone N32
prototype: 1,936 instructions across its two functions, with 625/659 in
the respective K128 loops and unchanged 51/48 register counts. This is a
CPU/SASS check; the measured candidate used by the normal build is unchanged.

The CPU cursor oracle reads actual layer-39 and layer-42 gate/up records,
using two N32 macros per projection and all eight K-warp starting positions.
It checks 10,240 K128 records against independent original GGUF metadata,
index/sign fields and nibble planes. All checks pass, including odd initial
half-blocks and cached metadata advancement. Reproduce with:

```bash
python benchmarks/kernels/benchmark_gguf_pair_reader_cursor.py MODEL.gguf \
  --output pair-cursor-oracle.json
```

The individual IQ4_XS device operand gate has passed both full rank-zero
real tensors and the exhaustive scale/LUT stress matrix. That evidence does
not validate this new reader cursor, mixed MMA wiring, or its gated output
on the GPU. Those require a later operator comparison and same-session
microbenchmark. No new GPU GEMM, speed or model result is claimed here.
