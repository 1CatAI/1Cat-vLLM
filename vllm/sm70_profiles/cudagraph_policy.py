# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Context-bucket policy for the SM70 graph dispatcher."""

import os


def context_buckets_for_descriptor(self, batch_descriptor) -> tuple[int, ...]:
    if (
        batch_descriptor.num_reqs is not None
        and not batch_descriptor.uniform
        and self.uniform_decode_query_len > 1
        and "VLLM_SM70_MTP_CONTEXT_BUCKETS" in os.environ
        and not self.cudagraph_mode.separate_routine()
    ):
        # FULL mode keys decode batches as non-uniform descriptors. Buckets
        # are only dispatched with an attention context, which the runner
        # supplies for uniform decode batches alone.
        if batch_descriptor.num_tokens % self.uniform_decode_query_len == 0:
            return self.sm70_mtp_context_buckets
        return ()

    if not batch_descriptor.uniform or batch_descriptor.num_reqs is None:
        return ()

    if self.uniform_decode_query_len > 1:
        if "VLLM_SM70_MTP_CONTEXT_BUCKETS" in os.environ:
            # Explicit MTP buckets apply to every uniform verification batch;
            # the replayed bucket covers the longest request in the batch.
            if (
                batch_descriptor.num_tokens
                == batch_descriptor.num_reqs * self.uniform_decode_query_len
            ):
                return self.sm70_mtp_context_buckets
            return ()
        if batch_descriptor.num_tokens == self.uniform_decode_query_len:
            return self.sm70_dsv4_decode_context_buckets
        return ()

    buckets: set[int] = set()
    if batch_descriptor.num_tokens == 1:
        buckets.update(self.sm70_dsv4_decode_context_buckets)
        buckets.update(self.sm70_fp8_kv_decode_context_buckets)

    return tuple(sorted(buckets))
