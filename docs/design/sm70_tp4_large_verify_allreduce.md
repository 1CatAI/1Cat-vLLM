# SM70 TP4 all-reduce for larger verification batches

Captured FP16 verification batches with a hidden size of 5120 now reuse the
existing SM70 TP4 push all-reduce through 64 rows (640 KiB). Admission still
requires four fully interconnected SM70 devices and an initialized IPC push
buffer. Eager calls, other dtypes, and unsupported message sizes retain their
existing implementations. No new runtime option is required.

Messages above 32 rows use 256 threads per CTA, with the existing 80-CTA grid
and two-epoch protocol. Smaller-message launch geometry and accumulation order
are unchanged. The maximum persistent IPC allocation grows by approximately
2.5 MiB per rank; account for this allocation when sizing a nearly full device.

## Validation

`tests/distributed/test_sm70_large_push_all_reduce.py` compares captured chains
against the original dispatch, including partial verification batches,
alternating message lengths, repeated graph replay, rank skew, cancellation,
unsupported dtypes, and output guards.

A preliminary cold-L2 graph microbenchmark on four fully interconnected
V100-SXM2-16GB devices at a 300 W power limit measured a 64-row collective at
58.98 us with the previous two-stage path and 20.16 us with push. The benchmark
used 16 collectives per replay, 60 replays, 128 MiB L2 eviction, and an untimed
GPU collective before the timing events to align the ranks. Outputs were
bitwise identical. These are operator measurements, not model speed claims;
complete verification-round measurements are required before promotion.
