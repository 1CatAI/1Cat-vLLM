# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU metadata contracts for bounded FP32 index scores; no torch import."""

import ast
import math
import sys
import unittest
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import patch


class TensorMetadata:
    def __init__(self, shape, dtype="torch.float16", strides=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = "cuda:metadata-only"
        self._strides = strides or tuple(
            math.prod(self.shape[i + 1 :]) for i in range(len(shape))
        )

    def stride(self, dim):
        return self._strides[dim]

    def reshape(self, *shape):
        return TensorMetadata(shape, self.dtype)

    def is_contiguous(self):
        return True

    def contiguous(self):
        return self

    def t(self):
        return TensorMetadata(self.shape[::-1], self.dtype, self._strides[::-1])

    def view(self, dtype):
        return TensorMetadata(self.shape, dtype, self._strides)

    def __getitem__(self, keys):
        keys = keys if isinstance(keys, tuple) else (keys,)
        keys += (slice(None),) * (len(self.shape) - len(keys))
        shape = tuple(
            len(range(*key.indices(size))) for key, size in zip(keys, self.shape)
        )
        return TensorMetadata(shape, self.dtype, self._strides)


class KernelMetadata:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            # Record scalar metadata only: retaining args would keep C alive.
            self.calls.append(
                {
                    "grid": grid,
                    "first_shape": args[0].shape,
                    "first_dtype": args[0].dtype,
                    "kwargs": kwargs,
                }
            )

        return launch


class PrefillFP32Contract(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "vllm/models/deepseek_v4/sm70/indexer.py"
        tree = ast.parse(path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_prefill_logits_cublas"
        )
        helper = ModuleType("vllm.v1.worker.gpu.causal_indexer_inputs")
        cast(Any, helper).current_indexer_causal_context = lambda: None
        self.import_patch = patch.dict(sys.modules, {helper.__name__: helper})
        self.import_patch.start()
        self.addCleanup(self.import_patch.stop)
        self.tile_mb = 192
        self.calls: list[dict[str, Any]] = []
        self.score_refs: list[weakref.ReferenceType[TensorMetadata]] = []
        self.epilogue = KernelMetadata()
        self.dequant = KernelMetadata()

        def mm(a, b, **kwargs):
            live = sum(reference() is not None for reference in self.score_refs)
            self.calls.append(
                {
                    "shape": (a.shape[0], b.shape[1]),
                    "kwargs": kwargs,
                    "previous_live_scores": live,
                }
            )
            tensor = TensorMetadata(
                (a.shape[0], b.shape[1]), kwargs.get("out_dtype", a.dtype)
            )
            self.score_refs.append(weakref.ref(tensor))
            return tensor

        torch = SimpleNamespace(
            float16="torch.float16",
            float32="torch.float32",
            uint8="torch.uint8",
            empty=lambda shape, dtype, device: TensorMetadata(shape, dtype),
            mm=mm,
        )
        scope: dict[str, Any] = {
            "torch": torch,
            "sparse_policy": lambda: SimpleNamespace(value=lambda name: self.tile_mb),
            "_dequant_index_k_unscaled_kernel": self.dequant,
            "_relu_weight_headsum_kernel": self.epilogue,
            "_RELU_BLOCK_N": 64,
            "_EPILOGUE_BLOCK_K": 256,
            "_EPILOGUE_BLOCK_H": 8,
            "triton": SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
        }
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
            scope,
        )
        self.run_prefill = scope["_prefill_logits_cublas"]

    def invoke(self, queries=640, heads=32, keys=8192):
        return self.run_prefill(
            TensorMetadata((queries, heads, 128)),
            TensorMetadata((keys, 128), "torch.float8_e4m3fn"),
            TensorMetadata((keys,), "torch.float32"),
            TensorMetadata((queries, heads), "torch.float32"),
        )

    def test_real_shape_uses_four_byte_tiles_without_a_tag(self):
        result = self.invoke()
        self.assertEqual(result.shape, (640, 8192))
        self.assertEqual(result.dtype, "torch.float32")
        self.assertEqual(
            [row["shape"] for row in self.calls], [(20480, 2432)] * 3 + [(20480, 896)]
        )
        self.assertTrue(
            all(row["kwargs"] == {"out_dtype": "torch.float32"} for row in self.calls)
        )
        self.assertTrue(
            all(row["first_dtype"] == "torch.float32" for row in self.epilogue.calls)
        )

    def test_previous_score_is_released_before_next_allocation(self):
        self.invoke()
        self.assertEqual(
            [row["previous_live_scores"] for row in self.calls], [0, 0, 0, 0]
        )
        self.assertTrue(all(reference() is None for reference in self.score_refs))

    def test_budget_depends_on_actual_head_count(self):
        for heads in (1, 7, 8, 9, 16, 32, 64):
            with self.subTest(heads=heads):
                self.calls.clear()
                self.invoke(heads=heads)
                self.assertEqual(sum(row["shape"][1] for row in self.calls), 8192)
                self.assertTrue(
                    all(
                        math.prod(row["shape"]) * 4 <= 192 * 1024**2
                        for row in self.calls
                    )
                )

    def test_sub64_budget_does_not_round_up_past_the_cap(self):
        self.tile_mb = 1
        self.invoke(queries=512, heads=32, keys=17)
        self.assertEqual(
            [row["shape"] for row in self.calls], [(16384, 16), (16384, 1)]
        )
        self.assertTrue(
            all(math.prod(row["shape"]) * 4 <= 1024**2 for row in self.calls)
        )

    def test_too_small_for_one_column_is_rejected_before_kernels(self):
        self.tile_mb = 1
        with self.assertRaisesRegex(ValueError, "one FP32 score column"):
            self.invoke(queries=8193, heads=32, keys=1)
        self.assertFalse(self.calls or self.dequant.calls or self.epilogue.calls)

    def test_nonpositive_budget_is_rejected(self):
        for budget in (0, -1):
            with self.subTest(budget=budget):
                self.tile_mb = budget
                with self.assertRaisesRegex(ValueError, "one FP32 score column"):
                    self.invoke()
        self.assertFalse(self.calls or self.dequant.calls)

    def test_empty_query_or_key_axes_do_not_launch_kernels(self):
        for queries, keys in ((0, 8192), (640, 0)):
            with self.subTest(queries=queries, keys=keys):
                result = self.invoke(queries=queries, keys=keys)
                self.assertEqual(result.shape, (queries, keys))
                self.assertEqual(result.dtype, "torch.float32")
        self.assertFalse(self.calls or self.dequant.calls or self.epilogue.calls)

    def test_zero_heads_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one head"):
            self.invoke(heads=0)
        self.assertFalse(self.calls or self.dequant.calls)


if __name__ == "__main__":
    unittest.main()
