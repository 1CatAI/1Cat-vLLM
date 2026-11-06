# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The V4.1 DSpark drafter gets the target embedding on the last PP stage (CPU).

The V4.1 draft has no embedding table: ``V41DraftModel.embed_tokens`` is an
Identity placeholder that the proposer replaces with the target's module. The
proposer's legacy sharing contract (``share_embeddings`` with
``LEGACY_DRAFT_WEIGHTS``) returns early whenever ``pp_size != 1``, which would
leave the Identity in place on PP2's stage 2 and feed token ids to the draft.
``DSparkProposer._maybe_share_embeddings`` then calls the model's
``bind_pipeline_embedding``, which binds the target's stage copy (V4.1 builds
``embed_tokens`` on every stage).

The test takes the real code by source: ``shared_weights.py`` (without its
imports), ``SpecDecodeBaseProposer._maybe_share_embeddings``,
``DSparkProposer._maybe_share_embeddings`` and the V4.1 model's
``bind_pipeline_embedding`` / ``embed_input_ids``. The proposer and model
modules cannot be imported here (they need the compiled extension).

    python tests/models/test_deepseek_v41_pp_drafter.py   # or pytest
"""

import ast
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "vllm" / "model_executor" / "models" / "shared_weights.py"
BASE = ROOT / "vllm" / "v1" / "spec_decode" / "llm_base_proposer.py"
PROPOSER = ROOT / "vllm" / "v1" / "spec_decode" / "dspark.py"
DRAFT = ROOT / "vllm" / "models" / "deepseek_v41" / "dspark.py"
SHARED_NAME = "vllm.model_executor.models.shared_weights"


class PPMissingLayer(nn.Identity):
    """Stand-in for vllm.model_executor.models.utils.PPMissingLayer."""


def _method(path, cls, name):
    tree = ast.parse(path.read_text())
    body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(
        n for n in body.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def _shared_module():
    tree = ast.parse(SHARED.read_text())
    nodes = [n for n in tree.body if not isinstance(n, ast.Import | ast.ImportFrom)]
    module = types.ModuleType(SHARED_NAME)
    module.__dict__.update(
        dataclass=dataclass,
        torch=torch,
        nn=nn,
        PPMissingLayer=PPMissingLayer,
        init_logger=logging.getLogger,
    )
    body = ast.Module(body=nodes, type_ignores=[])
    exec(compile(body, str(SHARED), "exec"), module.__dict__)
    return module


SHARED_MODULE = _shared_module()


def _class(name, bases, methods, namespace, path):
    """Compile methods taken from source inside a class body (zero-arg super)."""
    cls = ast.ClassDef(
        name=name,
        bases=[ast.Name(id=b, ctx=ast.Load()) for b in bases],
        keywords=[],
        body=methods,
        decorator_list=[],
        type_params=[],
    )
    module = ast.Module(body=[cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def _proposer_classes(pp_size):
    namespace = {
        "get_pp_group": lambda: SimpleNamespace(world_size=pp_size),
        "logger": logging.getLogger("test_pp_drafter"),
        "override": lambda fn: fn,
        "torch": torch,
        "nn": nn,
    }
    base_method = _method(BASE, "SpecDecodeBaseProposer", "_maybe_share_embeddings")
    base = _class("Base", [], [base_method], namespace, BASE)
    dspark_method = _method(PROPOSER, "DSparkProposer", "_maybe_share_embeddings")
    dspark = _class("DSparkProposer", ["Base"], [dspark_method], namespace, PROPOSER)
    return base, dspark


def _draft_class(with_hook=True):
    namespace = {
        "nn": nn,
        "PPMissingLayer": PPMissingLayer,
        "bind_shared": SHARED_MODULE.bind_shared,
    }
    names = ["embed_input_ids"] + (["bind_pipeline_embedding"] if with_hook else [])
    methods = [_method(DRAFT, "DSparkDeepseekV41ForCausalLM", n) for n in names]
    return _class("Draft", [], methods, namespace, DRAFT)


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Identity()  # V41DraftModel's placeholder


def _draft(with_hook=True):
    draft = _draft_class(with_hook)()
    draft.has_own_embed_tokens = False
    draft.model = _Inner()
    return draft


def _target(embed=None):
    if embed is None:
        embed = nn.Embedding(32, 8)
    return SimpleNamespace(model=SimpleNamespace(embed_tokens=embed))


def _share(pp_size, draft, target, hook=True):
    base, dspark = _proposer_classes(pp_size)
    proposer = (dspark if hook else base)()
    proposer.model = draft
    saved = sys.modules.get(SHARED_NAME)
    sys.modules[SHARED_NAME] = SHARED_MODULE  # the base method imports it
    try:
        proposer._maybe_share_embeddings(target)
    finally:
        if saved is None:
            del sys.modules[SHARED_NAME]
        else:
            sys.modules[SHARED_NAME] = saved


def test_pp2_binds_target_embedding():
    ids = torch.tensor([3, 1, 4, 1, 5, 9])
    # The defect without the hook: the legacy contract leaves the Identity.
    draft, target = _draft(), _target()
    _share(2, draft, target, hook=False)
    assert isinstance(draft.model.embed_tokens, nn.Identity)
    assert torch.equal(draft.embed_input_ids(ids), ids)
    # DSparkProposer: the model binds the target's stage copy.
    draft, target = _draft(), _target()
    _share(2, draft, target)
    assert draft.model.embed_tokens is target.model.embed_tokens
    assert torch.equal(draft.embed_input_ids(ids), target.model.embed_tokens(ids))


def test_tp4_sharing_unchanged():
    draft, target = _draft(), _target()
    _share(1, draft, target, hook=False)
    assert draft.model.embed_tokens is target.model.embed_tokens
    assert draft.bind_pipeline_embedding(target) is False  # nothing to do
    draft, target = _draft(), _target()
    _share(1, draft, target)
    assert draft.model.embed_tokens is target.model.embed_tokens


def test_missing_target_embedding_raises():
    draft = _draft()
    try:
        _share(2, draft, _target(PPMissingLayer()))
    except ValueError as exc:
        assert "target embedding" in str(exc)
    else:
        raise AssertionError("a PPMissingLayer target embedding was accepted")


def test_other_dspark_models_untouched():
    # A DSpark model without the hook (e.g. V4) keeps the legacy behaviour.
    draft = _draft(with_hook=False)
    _share(2, draft, _target())
    assert isinstance(draft.model.embed_tokens, nn.Identity)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"{len(tests)} passed")
