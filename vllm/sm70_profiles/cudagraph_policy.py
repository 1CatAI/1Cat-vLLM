# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Context-bucket policy for the SM70 graph dispatcher."""


def context_buckets_for_descriptor(
    plan, uniform_decode_query_len, cudagraph_mode, batch_descriptor
) -> tuple[int, ...]:
    if (
        batch_descriptor.num_reqs is not None
        and not batch_descriptor.uniform
        and uniform_decode_query_len > 1
        and plan.mtp_explicit
        and not cudagraph_mode.separate_routine()
    ):
        # FULL mode keys decode batches as non-uniform descriptors. Buckets
        # are only dispatched with an attention context, which the runner
        # supplies for uniform decode batches alone.
        if batch_descriptor.num_tokens % uniform_decode_query_len == 0:
            return plan.mtp_buckets
        return ()

    if not batch_descriptor.uniform or batch_descriptor.num_reqs is None:
        return ()

    if uniform_decode_query_len > 1:
        if plan.mtp_explicit:
            # Explicit MTP buckets apply to every uniform verification batch;
            # the replayed bucket covers the longest request in the batch.
            if (
                batch_descriptor.num_tokens
                == batch_descriptor.num_reqs * uniform_decode_query_len
            ):
                return plan.mtp_buckets
            return ()
        if batch_descriptor.num_tokens == uniform_decode_query_len:
            return plan.compressed_buckets
        return ()

    buckets: set[int] = set()
    if batch_descriptor.num_tokens == 1:
        buckets.update(plan.compressed_buckets)
        buckets.update(plan.fp8_buckets)

    return tuple(sorted(buckets))
