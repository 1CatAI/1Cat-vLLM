# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression of the actual Qwen4ExpMTP.load_weights implementation.

Large tensor loaders and independent FP8/cache finalizers are stubbed. The
production remapper, load_weights method, and validation functions execute
directly from the prepared source, without importing torch or using GPUs.
"""

import ast
import sys
import unittest
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar, cast
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[3] / "vllm/models/qwen4_exp/nvidia/mtp.py"
EXPERTS = {
    "model.layers.0.mlp.experts.w13_weight",
    "model.layers.0.mlp.experts.w2_weight",
}


class LoaderStub:
    def __init__(self, model, **kwargs):
        self.model = model

    def load_weights(self, weights):
        # The full tensor loader is mocked; names still pass through the real
        # Qwen4ExpMTP remapping and skip_checkpoint_weight implementation.
        self.model.loaded_in_test = {name for name, tensor in weights}
        return self.model.loaded_in_test


def load_actual_methods() -> type[Any]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    requested = {
        "_remap_mtp_weight_name",
        "_validate_mtp_expert_weights_loaded",
        "_validate_mtp_embedding_weights_loaded",
    }
    functions: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in requested
    ]
    model_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpMTP"
    )
    methods: list[ast.stmt] = [
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("skip_checkpoint_weight", "load_weights")
    ]
    assert len(functions) == 3 and len(methods) == 2
    selected_class = ast.ClassDef(
        name="ActualMTPMethods", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    selected = ast.fix_missing_locations(
        ast.Module(body=functions + [selected_class], type_ignores=[])
    )
    namespace: dict[str, Any] = {
        "__name__": "guard_contract_test",
        "__package__": "guard_contract_test",
        "nn": SimpleNamespace(Module=object),
        "torch": SimpleNamespace(Tensor=object),
        "Iterable": Iterable,
        "AutoWeightsLoader": LoaderStub,
        "_QWEN4_EXP_IGNORED_MISSING_SUFFIXES": [],
    }
    exec(compile(selected, str(SOURCE), "exec"), namespace)
    return cast(type[Any], namespace["ActualMTPMethods"])


class TestMTPEmbeddingGuard(unittest.TestCase):
    model_type: ClassVar[type[Any]]

    @classmethod
    def setUpClass(cls):
        cls.model_type = load_actual_methods()

    def setUp(self):
        fp8 = ModuleType("guard_contract_test.mtp_fp8_checkpoint")
        fp8.__dict__["prepare_mtp_fp8_checkpoint"] = (
            lambda weights, *args, **kwargs: weights
        )
        finalizer = ModuleType("guard_contract_test.model")
        finalizer.__dict__["_finalize_qsa_e4m3_scale_load"] = (
            lambda *args, **kwargs: None
        )
        self.module_patch = patch.dict(
            sys.modules, {fp8.__name__: fp8, finalizer.__name__: finalizer}
        )
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def make_model(self, share_target_embedding):
        model = self.model_type()
        model.share_target_io_weights = share_target_embedding
        model.model = SimpleNamespace(
            fp8_mtp_checkpoint_prefixes=set(),
            fp8_mtp_tp_size=1,
            config=SimpleNamespace(num_experts=1),
            _kv_cache_dtype="auto",
        )
        names = EXPERTS | (
            set() if share_target_embedding else {"model.embed_tokens.weight"}
        )
        model.named_parameters = lambda: iter(
            (name, object()) for name in sorted(names)
        )
        return model

    @staticmethod
    def checkpoint(include_embedding=False, omit_expert=False):
        names = set(EXPERTS)
        if omit_expert:
            names.remove("model.layers.0.mlp.experts.w2_weight")
        rows = [(name.replace("model.", "mtp.", 1), object()) for name in sorted(names)]
        if include_embedding:
            rows.append(("model.embed_tokens.weight", object()))
        return rows

    def test_pp_private_embedding_missing_fails_before_generic_tracker(self):
        model = self.make_model(False)
        with self.assertRaisesRegex(
            ValueError, "token embedding was not loaded.*model.embed_tokens.weight"
        ):
            model.load_weights(self.checkpoint())
        self.assertNotIn("model.embed_tokens.weight", model.loaded_in_test)

    def test_pp1_target_embedding_sharing_allows_mtp_only_checkpoint(self):
        model = self.make_model(True)
        self.assertEqual(model.load_weights(self.checkpoint()), EXPERTS)

    def test_pp_private_embedding_complete_checkpoint_loads(self):
        model = self.make_model(False)
        self.assertEqual(
            model.load_weights(self.checkpoint(include_embedding=True)),
            EXPERTS | {"model.embed_tokens.weight"},
        )

    def test_existing_expert_guard_remains_active(self):
        model = self.make_model(False)
        with self.assertRaisesRegex(
            ValueError, "routed-expert checkpoint weights were not loaded"
        ):
            model.load_weights(
                self.checkpoint(include_embedding=True, omit_expert=True)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
