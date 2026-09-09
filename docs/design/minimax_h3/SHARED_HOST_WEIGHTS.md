# Shared immutable host VAE weights

`--share-host-vae-weights --disable-host-weight-pinning` enables an explicit
TP2/TP4 memory policy. The Python equivalent is
`H3Config(share_host_vae_weights=True, host_weight_pin_memory=False)` through
`H3Engine`. TP1 uses ordinary storage. The option is off by default.

The native video and audio VAEs keep replicated FP32 host masters even though
their computation is distributed. In the measured original-weight TP4 setup,
each rank retains 10,415,484,160 video-VAE bytes and 605,306,340 audio-VAE bytes.
Sharing one physical replica instead of four can remove 33,062,371,500 bytes
of duplicated host storage. This does not change GPU weights, arithmetic,
reference encoding or the VAE compute topology.

Each engine creates an owned temporary directory under `/dev/shm`. Rank zero
writes aligned storage groups and checksums; every worker validates its own
weights/layouts against the snapshot and maps the bytes privately. Private
mappings share clean physical pages and isolate accidental CPU writes.
Shapes, strides, storage offsets, mixed-dtype aliases and empty tensors are
preserved. GPU load/offload still copies the exact original storage bytes.
The parent removes its directory only after its workers stop, including failed
startup. Unrelated shared-memory files are not modified.

The policy requires pageable masters: ordinary PyTorch pinning would copy the
mapping into a separate pinned allocation per rank, defeating the memory
saving. CUDA host registration and automatic policy selection are not added.
The measured VAE snapshot needs approximately 11.02 GB of tmpfs capacity plus
small alignment/metadata overhead. Space is reserved before mapping writes.

## Evidence and current limits

The original-weight Cache-DiT lifecycle trace separates allocation and copies.
First DiT device allocation takes 0.18–0.19 s, while copies take 27.7–43.2 s
with 142,577–268,365 major faults/rank. Later copies take 6.5–8.2 s. The memory
policy targets duplicated host storage and cold paging; it is not evidence of
higher denoising TFLOP/s.

`shared-host-cpu-v1.log`: 19 host-storage and native service checks pass,
including nonzero offsets, transposed and strided views, mixed-dtype aliases,
private writes, mismatched replicas, corrupt snapshots and owned cleanup.
`shared-host-gpu.log`: exact storage roundtrips pass across three GPU
load/offload cycles. Full TP4 generation, measured physical host-memory savings
and final media comparison remain pending. Do not qualify this option for
AUTO or claim an end-to-end speed improvement yet.

Evidence root: `/data/minimax-h3/sm70-general-20260909/`. Runtime:
Python 3.12.13, Torch 2.10.0+cu128, CUDA 12.8.93, V100 SXM2 32GB.
