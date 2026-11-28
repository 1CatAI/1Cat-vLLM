# SPDX-License-Identifier: Apache-2.0
import unittest
import torch

from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode, SlidingWindowSpec


class TestHybridKvPageScale(unittest.TestCase):
    def test_scales_int8_per_token_head_before_padding(self):
        # Gemma 4 31B TP4 int8_per_token_head: SWA 33,280 B/16 tok vs Global 16,512 B/16 tok
        specs = {
            "swa": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=4,
                head_size=256,
                dtype=torch.int8,
                kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
                sliding_window=1024,
            ),
            "global": FullAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.int8,
                kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
            ),
        }
        unified = unify_kv_cache_spec_page_size(specs)
        self.assertEqual(unified["swa"].block_size, 16)
        self.assertEqual(unified["swa"].page_size_bytes, 33280)
        self.assertEqual(unified["global"].block_size, 32)
        self.assertEqual(unified["global"].page_size_bytes, 33280)
        self.assertEqual(unified["global"].page_size_padded, 33280)

    def test_exact_divisible_unchanged(self):
        specs = {
            "swa": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=4,
                head_size=256,
                dtype=torch.float16,
                sliding_window=1024,
            ),
            "global": FullAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.float16,
            ),
        }
        unified = unify_kv_cache_spec_page_size(specs)
        self.assertEqual(unified["global"].block_size, 32)
        self.assertIsNone(unified["global"].page_size_padded)


if __name__ == "__main__":
    unittest.main()
