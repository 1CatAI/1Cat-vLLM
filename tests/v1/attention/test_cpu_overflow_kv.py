# SPDX-License-Identifier: Apache-2.0
import unittest
import torch

from vllm.v1.attention.ops.cpu_overflow_kv import (
    compute_cpu_overflow_split,
    overflow_attention_req,
)


class TestCpuOverflowKV(unittest.TestCase):
    def test_compute_split(self):
        bpb = 32768
        self.assertEqual(compute_cpu_overflow_split(10000, 10000 * bpb, 1000 * bpb), (9000, 1000))
        self.assertEqual(compute_cpu_overflow_split(2, 2 * bpb, 1 << 30), (2, 0))
        self.assertEqual(compute_cpu_overflow_split(10, 10 * bpb, 100 * bpb), (2, 8))

    def test_mixed_gpu_cpu_attention_parity(self):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)
        n_gpu, n_cpu, bs, g, hq, d = 3, 2, 16, 2, 4, 32
        seq_len = 64  # 4 blocks: blocks [1, 2] on GPU, blocks [3, 4] on CPU overflow
        kv_full = torch.randn((n_gpu + n_cpu, 2, bs, g, d), dtype=torch.float16, device=dev)
        gpu_kv = kv_full[:n_gpu].clone()
        cpu_kv = kv_full[n_gpu:].cpu()

        q = torch.randn((8, hq, d), dtype=torch.float16, device=dev)
        bt = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=dev)
        out_mixed = torch.empty_like(q)
        overflow_attention_req(q, gpu_kv, cpu_kv, n_gpu, out_mixed, seq_len, 1.0, 0, bt, kb=32)

        # Reference: all blocks on GPU (n_gpu = n_gpu + n_cpu)
        out_ref = torch.empty_like(q)
        overflow_attention_req(
            q, kv_full, kv_full[:0].cpu(), n_gpu + n_cpu, out_ref, seq_len, 1.0, 0, bt, kb=32
        )
        self.assertTrue(torch.allclose(out_mixed, out_ref, atol=1e-3, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()
