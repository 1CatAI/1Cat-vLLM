# Joint GGUF attention input projections

The TP4 27B full-attention input has M8/K5120 and output widths 3072/256/256
for Q, K and V. The existing coalesced canonical execution still produces
forty-one matrix launches across sixteen layers because formats differ.

The prototype specializes the measured qkvz launch and shared-A body for three
sources and N3584. Each source retains its own reader and output tile range;
FP32 partials are reduced by the last CTA in the same launch. Q2_K and Q4_K
read existing canonical code/stat views, preserving the full stats stride.
Other supported sources reuse original-record readers. No additional
dequantization, activation copy, concatenation or reduction launch is added.

The benchmark screens one real TP4 shard per source-type combination against
the existing coalesced canonical call. Official GGUF reconstruction checks
each projection separately with three seeded inputs, followed by one thousand
bitwise-stable graph replays and cold-L2 ABBA timing. Native FP32 coefficient
products and accumulation are retained. The existing qkvz instantiation also
needs a regression check after the shared launch becomes a template.

This prototype has no model dispatch admission. Timing and clean package
results are pending; unmeasured shapes and M retain canonical execution.
