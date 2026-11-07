# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 on Model Runner V2: selection, state logic and verify math.

CPU only. The kernel tests (greedy_verify, keyed sampling) need CUDA or the
Triton interpreter (``TRITON_INTERPRET=1``) and are skipped otherwise.

The PP2 test drives the real ``DeepseekV41ModelState`` and the runner's stage
hooks around DeepseekV41Model's commit methods (taken from model.py by source,
tests/models/test_deepseek_v41_pp_commit.py) and the device protocol that
traps a missing, double or miscounted commit. The slot ring mirrors
``PPHandler`` (pp_utils.py): one pop per execute_model call (empty batches
too), a receive after each forward that produces a sample, and rows freed
since the receive excluded.
"""

import ast
import os
import random
from collections import Counter, deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.models.test_deepseek_v41_pp_commit import K, _forward, _stage_model
from vllm.config.compilation import CUDAGraphMode
from vllm.models.deepseek_v41.pp_commit import StageCommit
from vllm.sampling_params import SamplingParams
from vllm.v1.spec_decode import coupled_draw
from vllm.v1.worker.gpu import model_states
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.model_states import deepseek_v41 as v41_state
from vllm.v1.worker.gpu.model_states import default as default_state
from vllm.v1.worker.gpu.pp_utils import compute_need_sampled_mask
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode.dspark import v41 as v41_spec

V41 = "deepseek_v41_text"
KERNELS = torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"
needs_kernels = pytest.mark.skipif(
    not KERNELS, reason="Triton kernels need CUDA or TRITON_INTERPRET=1"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ selection


def _vllm_config(model_type, *, architectures=("X",), hybrid=False, spec=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=list(architectures),
            hf_text_config=SimpleNamespace(model_type=model_type),
            is_hybrid=hybrid,
        ),
        speculative_config=spec,
    )


@pytest.mark.parametrize(("model_type", "want"), [(V41, "v41"), ("qwen3", "default")])
def test_model_state_selection(monkeypatch, model_type, want):
    monkeypatch.setattr(
        v41_state, "DeepseekV41ModelState", lambda *a: "v41", raising=True
    )
    monkeypatch.setattr(
        default_state, "DefaultModelState", lambda *a: "default", raising=True
    )
    config = _vllm_config(model_type)
    assert model_states.init_model_state(config, object(), None, "cpu") == want


@pytest.mark.parametrize(("draft_type", "want"), [(V41, "v41"), ("qwen3", "dspark")])
def test_speculator_selection(monkeypatch, draft_type, want):
    from vllm.v1.worker.gpu.spec_decode.dspark import speculator as dspark

    monkeypatch.setattr(v41_spec, "V41DSparkSpeculator", lambda *a: "v41")
    monkeypatch.setattr(dspark, "DSparkSpeculator", lambda *a: "dspark")
    spec = SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=draft_type)
        ),
    )
    assert init_speculator(_vllm_config(V41, spec=spec), "cpu") == want


@pytest.mark.parametrize(("env", "want"), [(None, False), ("0", False), ("1", True)])
def test_v41_runner_switch(monkeypatch, env, want):
    # Unset or 0: the V1 runner (V4.1 is not a default MRV2 architecture).
    from vllm.config.vllm import VllmConfig

    if env is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", env)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["DeepseekV41ForCausalLM"],
            runner_type="generate",
            is_moe=True,
            is_quantized=True,
        ),
        speculative_config=SimpleNamespace(use_dflash=lambda: False),
    )
    config._is_default_v2_model_runner_model = (
        lambda: VllmConfig._is_default_v2_model_runner_model(config)
    )
    assert VllmConfig.use_v2_model_runner.fget(config) is want


# ------------------------------------------------------------- runner helpers


def _runner(model_type=V41, max_num_reqs=1, mode=CUDAGraphMode.FULL, sp=False):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type)
        ),
        max_num_reqs=max_num_reqs,
        compilation_config=SimpleNamespace(
            cudagraph_mode=mode, pass_config=SimpleNamespace(enable_sp=sp)
        ),
        uniform_decode_query_len=1 + K,
    )


@pytest.mark.parametrize(
    ("kwargs", "want"),
    [
        ({}, 1),
        ({"model_type": "qwen3"}, 1 + K),
        ({"max_num_reqs": 2}, 1 + K),
        ({"mode": CUDAGraphMode.PIECEWISE}, 1 + K),
        ({"mode": CUDAGraphMode.FULL_AND_PIECEWISE}, 1 + K),
        ({"sp": True}, 1 + K),
    ],
)
def test_target_graph_query_len(kwargs, want):
    # V1's graph_query_len: exact M1..6 graphs for one V4.1 request.
    assert GPUModelRunner._target_graph_query_len(_runner(**kwargs)) == want


def _hook_runner(model, *, pp=True, first=True, last=False, model_type=V41):
    runner = _runner(model_type)
    runner.use_pp, runner.is_first_pp_rank, runner.is_last_pp_rank = pp, first, last
    runner.model = model
    return runner


def test_pp_stage_hooks_resolution():
    model = SimpleNamespace(pp_before_forward=lambda *a: 1, pp_after_forward=None)
    model.pp_after_forward = lambda *a: 2
    resolve = GPUModelRunner._resolve_pp_stage_hooks
    hooks = resolve(_hook_runner(model))
    assert hooks == (model.pp_before_forward, model.pp_after_forward)
    assert resolve(_hook_runner(model, pp=False, last=True)) is None
    assert resolve(_hook_runner(model, first=False, last=True)) is None
    assert resolve(_hook_runner(model, model_type="qwen3")) is None
    with pytest.raises(ValueError, match="pp_before_forward"):
        resolve(_hook_runner(SimpleNamespace()))


# ---------------------------------------------------------------- model state


class _LanguageModel:
    """DeepseekV41ForCausalLM's hooks over a stage model from
    test_deepseek_v41_pp_commit (model.py's commit methods)."""

    def __init__(self, model):
        self.model = model
        self.restores = []

    def commit_speculative_state(self, counts):
        self.model.commit(counts)

    def restore_prefix(self, cached, prompt_ids):
        # DeepseekV41Model.restore_prefix: reset the protocol, rewind state.
        self.restores.append((cached, list(prompt_ids)))
        if self.model.stage_commit is not None:
            self.model.stage_commit.reset()
        for state in self.model.states:
            state.position, state.pending = cached, False

    def pp_before_forward(self, req_ids, num_computed):
        self.model.pp_before_forward(req_ids, num_computed)

    def pp_after_forward(self, req_ids, num_computed, num_prompt):
        self.model.pp_after_forward(req_ids, num_computed, num_prompt)


def _model_state(monkeypatch, language_model, *, max_num_reqs=1, overlap=0, seed=0):
    monkeypatch.setattr(default_state, "get_rope_state", lambda *a, **k: None)
    model_config = SimpleNamespace(
        max_model_len=4096,
        dtype=torch.float16,
        seed=seed,
        get_inputs_embeds_size=lambda: 8,
    )
    config = SimpleNamespace(
        model_config=model_config,
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_reqs, max_num_batched_tokens=8192
        ),
        kernel_config=SimpleNamespace(tp_prefill_overlap_min_tokens=overlap),
    )
    return v41_state.DeepseekV41ModelState(
        config, language_model, None, torch.device("cpu")
    )


def _new_req(req_id, computed=0, prompt=(5, 6, 7, 8), **params):
    return SimpleNamespace(
        req_id=req_id,
        num_computed_tokens=computed,
        prefill_token_ids=list(prompt),
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(**params) if params is not None else None,
        mm_features=None,
    )


def test_model_state_fails_closed(monkeypatch):
    with pytest.raises(ValueError, match="tp_prefill_overlap"):
        _model_state(monkeypatch, _LanguageModel(_stage_model(0, 40)), overlap=64)
    with pytest.raises(ValueError, match="commit_speculative_state"):
        _model_state(monkeypatch, SimpleNamespace(model=object()))
    state = _model_state(monkeypatch, _LanguageModel(_stage_model(0, 40)))
    with pytest.raises(ValueError, match="one request"):
        state.postprocess_state(torch.zeros(2, dtype=torch.int32), torch.ones(2))


def test_model_state_add_request(monkeypatch):
    lm = _LanguageModel(_stage_model(0, 20))
    state = _model_state(monkeypatch, lm, max_num_reqs=2, seed=7)
    commit = lm.model.stage_commit
    commit.after_forward(["old"], [10], [10])  # an armed decode forward
    state.add_request(0, _new_req("a", temperature=0.0))
    assert not commit.armed and lm.restores == []  # position 0: reset
    state.add_request(1, _new_req("b", computed=4, temperature=0.8, seed=3))
    assert lm.restores == [(4, [5, 6, 7, 8])]
    # The V1 runner's keys (_fill_coupled_keys) and greedy temperature 0.
    assert state.keys_np[0] == coupled_draw.request_key(None, "a", 7)
    assert state.keys_np[1] == coupled_draw.request_key(3, "b", 7)
    assert state.temperature_np.tolist() == pytest.approx([0.0, 0.8])
    state.add_request(0, _new_req("c", temperature=0.5))
    state.apply_staged_writes()
    assert state.keys_gpu.tolist() == state.keys_np.tolist()
    assert not state._keys_dirty


def _module_names(tree):
    """Names bound at module level (imports, assignments, defs)."""
    bound = set()
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, ast.Assign):
            bound |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
        elif isinstance(node, ast.FunctionDef | ast.ClassDef):
            bound.add(node.name)
    return bound


def _model_restore_prefix():
    """DeepseekV41Model.restore_prefix and _ratio2_states from model.py.

    Every global those methods read must be bound in model.py itself (dd522db
    logged through an undefined ``logger``; 7e9a156 defines it)."""
    import builtins
    from pathlib import Path

    import vllm.envs as envs

    path = Path(v41_state.__file__).parents[4] / "models/deepseek_v41/model.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV41Model"
    )
    names = {"restore_prefix", "_ratio2_states"}
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    bound = _module_names(tree) | set(dir(builtins))
    for fn in nodes:
        local = {a.arg for a in fn.args.args}
        local |= {
            n.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        loads = {
            n.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        assert loads - local - bound == set(), (fn.name, loads - local - bound)
    namespace = {"torch": torch, "envs": envs, "logger": v41_state.logger}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["restore_prefix"], namespace["_ratio2_states"]


class _HotStore:
    def __init__(self):
        self.refills = []

    def refill_hot(self, cached):
        self.refills.append(cached)
        return 0


class _RestoreStage:
    """A DeepseekV41Model stage with model.py's restore_prefix (1Cat #1341
    hot-ring refill on every stage, Engram/C2 rewind on the first)."""

    def __init__(self, start, end):
        from types import MethodType

        restore_prefix, ratio2_states = _model_restore_prefix()
        self.restore_prefix = MethodType(restore_prefix, self)
        self._ratio2_states = MethodType(ratio2_states, self)
        self.start_layer, self.end_layer = start, end
        first = start == 0
        self.stage_commit = StageCommit() if first and end < 40 else None
        self.hash_state = (
            SimpleNamespace(
                history=torch.zeros(4, dtype=torch.long),
                token_map=torch.arange(64, dtype=torch.long) + 100,
                metadata=torch.ones(4, dtype=torch.long),
            )
            if first
            else None
        )
        state = SimpleNamespace(
            partial=torch.ones(3),
            position=torch.zeros(1, dtype=torch.long),
            metadata=torch.ones(4, dtype=torch.long),
        )
        compressor = SimpleNamespace(ratio=2, state=state)
        self.layers = {
            i: SimpleNamespace(attn=SimpleNamespace(compressor=compressor))
            for i in range(start, end)
        }
        self.c2 = state
        self._v41_shared = SimpleNamespace(store=_HotStore())


class _RestoreLM:
    """DeepseekV41ForCausalLM's hooks (restore_prefix -> model.restore_prefix)."""

    def __init__(self, model):
        self.model = model

    def commit_speculative_state(self, counts):
        raise AssertionError("not called by add_request")

    def restore_prefix(self, cached, prompt_ids):
        self.model.restore_prefix(cached, prompt_ids)


def test_prefix_hit_restores_every_pp_stage(monkeypatch):
    # V1 (gpu_model_runner.py _update_states) calls model.restore_prefix on
    # every PP rank when num_computed_tokens > 0; MRV2 runs add_request on every
    # rank too, so each stage rewinds its own private state.
    prompt = [7, 8, 9, 10, 11, 12]
    stages = [_RestoreStage(0, 20), _RestoreStage(20, 40)]
    for stage in stages:
        state = _model_state(monkeypatch, _RestoreLM(stage))
        if stage.stage_commit is not None:
            stage.stage_commit.after_forward(["old"], [10], [10])
            assert stage.stage_commit.armed
        state.add_request(0, _new_req("hit", computed=6, prompt=prompt))
        if stage.start_layer == 0:
            assert not stage.stage_commit.armed  # the protocol is reset
            assert stage.hash_state.history.tolist() == [110, 111, 112, 6]
            assert stage.hash_state.metadata[2] == 0
            assert stage.c2.position.item() == 6
            assert stage.c2.partial.abs().sum() == 0
        else:
            assert stage.c2.position.item() == 0  # later stages: no rewind
        # Position 0: no restore, no refill.
        state.add_request(0, _new_req("fresh", computed=0, prompt=prompt))


def test_model_state_commits(monkeypatch):
    model = _stage_model(0, 40)  # one stage: commit after sampling
    state = _model_state(monkeypatch, _LanguageModel(model))
    idx = torch.zeros(1, dtype=torch.int32)
    _forward(model, 0, 6, True)
    state.postprocess_state(idx, 0)  # int: no sampled counts, no commit
    assert model.hash_state.pending
    state.postprocess_state(idx, torch.tensor([0], dtype=torch.int32))
    assert model.hash_state.commits == [(0, 6)]  # prefill: pending_length
    _forward(model, 6, 1 + K, False)
    state.postprocess_state(idx, torch.tensor([3], dtype=torch.int32))
    assert model.hash_state.commits == [(0, 6), (6, 3)]


def _attn_batch(num_reqs, padded, prefilling):
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_reqs_after_padding=padded,
        num_tokens=num_reqs * 6,
        num_tokens_after_padding=padded * 6,
        is_prefilling_np=np.array(prefilling, dtype=np.bool_),
        query_start_loc_np=np.arange(padded + 1, dtype=np.int32) * 6,
        query_start_loc=torch.arange(padded + 1, dtype=torch.int32) * 6,
        num_scheduled_tokens=np.full(num_reqs, 6, dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.full((padded,), 40, dtype=torch.int32),
        seq_lens=torch.full((padded,), 40, dtype=torch.int32),
        dcp_local_seq_lens=None,
        positions=torch.arange(padded * 6),
        prefix_anchor_lens=None,
        capture_context_len=None,
        is_dummy_batch=False,
    )


def _same(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.equal(a, b)
    return a is b or a == b


def test_prepare_attn_matches_default_plus_prefill_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(v41_state, "build_attn_metadata", lambda **k: calls.append(k))
    monkeypatch.setattr(
        default_state, "build_attn_metadata", lambda **k: calls.append(k)
    )
    state = _model_state(
        monkeypatch, _LanguageModel(_stage_model(0, 40)), max_num_reqs=4
    )
    args = ((), torch.zeros(1), [], None)
    for mode, capture in [
        (CUDAGraphMode.FULL, False),
        (CUDAGraphMode.NONE, False),
        (CUDAGraphMode.FULL, True),
    ]:
        calls.clear()
        batch = _attn_batch(2, 4, [True, False])
        state.prepare_attn(batch, mode, *args, for_capture=capture)
        default_state.DefaultModelState.prepare_attn(
            state, batch, mode, *args, for_capture=capture
        )
        got, want = calls
        extra = got.pop("model_specific_attn_metadata")
        assert got.pop("is_dummy_batch") is capture
        assert got.keys() == want.keys()
        for key in want:
            assert _same(got[key], want[key]), key
        num_reqs = 4 if mode == CUDAGraphMode.FULL else 2
        flag = extra.get_extra_common_attn_kwargs(0, num_reqs)["is_prefilling"]
        assert flag.tolist() == [True, False, False, False][:num_reqs]


# ------------------------------------------------- PP2 commits under the ring


def _request(rng, rid):
    prompt = rng.randint(1, 200)
    cached = rng.randrange(2, prompt, 2) if prompt > 8 and rng.random() < 0.3 else 0
    return SimpleNamespace(
        rid=rid,
        prompt=prompt,
        cached=cached,
        max_tokens=rng.randint(1, 40),
        chunk=rng.choice([16, 64, 8192]),
        eos_rate=rng.choice([0.0, 0.05, 0.2]),
    )


def _steps(rng, req):
    """(start, n, prefill, num_sampled) per forward. MRV2 commits num_sampled
    unclipped (the final step may pass max_tokens: the request ends there)."""
    steps, position = [], req.cached
    while position < req.prompt:
        n = min(req.chunk, req.prompt - position)
        steps.append((position, n, True, int(position + n == req.prompt)))
        position += n
    outputs = 1
    while outputs < req.max_tokens and rng.random() >= req.eos_rate:
        count = 1 + rng.randint(0, K)
        steps.append((position, 1 + K, False, count))
        position, outputs = position + count, outputs + count
    return steps


class _Stage:
    """One MRV2 PP stage of DeepSeek-V4.1 for one request at a time."""

    def __init__(self, monkeypatch, end_layer, ring_depth=2):
        self.model = _stage_model(0, end_layer)
        self.lm = _LanguageModel(self.model)
        self.state = _model_state(monkeypatch, self.lm)
        last = end_layer == 40
        runner = _hook_runner(self.lm, pp=not last, last=last)
        self.hooks = GPUModelRunner._resolve_pp_stage_hooks(runner)
        assert (self.hooks is None) == last
        # PPHandler ring: pp world size 2 (1 in the lookahead design below).
        self.queue = deque([None] * ring_depth)
        self.gen = 0  # req_idx_gen_np[0]
        self.deferred = 0
        self.bounds = []  # hash-state commit index at each add_request
        commit = self.model.stage_commit
        if commit is not None:
            before = commit.before_forward

            def counted(req_ids, num_computed):
                count = before(req_ids, num_computed)
                self.deferred += count is not None
                return count

            commit.before_forward = counted

    def segments(self):
        """Commits per request, in order (a final decode committed by the next
        call belongs to its own request: the ring pops before add_request)."""
        commits = self.model.hash_state.commits
        ends = self.bounds[1:] + [len(commits)]
        return [commits[a:b] for a, b in zip(self.bounds, ends)]

    def _postprocess(self, num_sampled):
        idx = torch.zeros(1, dtype=torch.int32)
        self.state.postprocess_state(idx, num_sampled)

    def execute(self, step=None, req=None, add=None, finished=False):
        """execute_model (+ sample_tokens when it ran the model)."""
        if self.hooks is not None:
            slot = self.queue.popleft()  # update_pp_decode_requests
            self.queue.append(None)
            if slot is not None and slot[0] == self.gen:
                self._postprocess(torch.tensor([slot[1]], dtype=torch.int32))
        if finished:
            self.gen += 1  # free_states -> on_req_idx_freed
        if add is not None:
            self.bounds.append(len(self.model.hash_state.commits))
            self.state.add_request(0, _new_req(add.rid, add.cached, range(add.prompt)))
        if step is None:
            return
        start, n, prefill, num_sampled = step
        if self.hooks is not None:
            self.hooks[0]([req.rid], [start])
        _forward(self.model, start, n, prefill)
        if self.hooks is None:  # last rank: postprocess_sampled
            self._postprocess(torch.tensor([num_sampled], dtype=torch.int32))
            return
        self.hooks[1]([req.rid], [start], [req.prompt])
        batch = SimpleNamespace(
            num_computed_tokens_np=np.array([start], dtype=np.int32),
            prefill_len_np=np.array([req.prompt], dtype=np.int32),
            max_seq_len_np=np.array([req.prompt + req.max_tokens], dtype=np.int32),
            num_scheduled_tokens=np.array([n], dtype=np.int32),
        )
        if compute_need_sampled_mask(batch) is not None:  # PPHandler.receive
            self.queue[-1] = (self.gen, num_sampled)
        else:
            self._postprocess(0)


def _run(monkeypatch, end_layer, reqs, streams, rng, warmup):
    """Engine cadence: under PP2 one request's real steps alternate with
    empty ones (its next step waits for the in-flight one); a finish arrives
    with the next request's first step or on its own empty step."""
    stage = _Stage(monkeypatch, end_layer)
    pp2 = end_layer < 40
    if warmup:  # warmup.py: fake prefill, then a decode with drafts, no gaps
        fake = SimpleNamespace(rid="warmup", prompt=7, cached=0, max_tokens=16)
        stage.execute((0, 7, True, 1), fake, add=fake)
        stage.execute((7, 1 + K, False, 1), fake)
        stage.execute(finished=True)
    finished = False
    for req, steps in zip(reqs, streams):
        if finished and rng.random() < 0.5:
            stage.execute(finished=True)
            finished = False
        early = False
        for index, step in enumerate(steps):
            stage.execute(step, req, add=req if index == 0 else None, finished=finished)
            finished = False
            if pp2:
                # A finish known before the last receive is popped (async
                # scheduling): the slot is excluded and the forward dropped.
                early = index == len(steps) - 1 and rng.random() < 0.3
                stage.execute(finished=early)
        finished = not early
    return stage


@pytest.mark.parametrize("warmup", [False, True])
def test_pp2_stage1_commits_match_single_stage(monkeypatch, warmup):
    for seed in range(150):
        rng = random.Random(seed)
        reqs = [_request(rng, f"r{seed}-{i}") for i in range(4)]
        streams = [_steps(rng, req) for req in reqs]
        single = _run(monkeypatch, 40, reqs, streams, random.Random(1), warmup)
        pp2 = _run(monkeypatch, 20, reqs, streams, random.Random(seed), warmup)
        want, got = single.segments(), pp2.segments()
        assert len(want) == len(got) == len(reqs) + warmup
        if warmup:
            # The fake decode is committed with the prefill's count (1), a
            # valid count; the next request resets the protocol.
            assert want[0] == got[0] == [(0, 7), (7, 1)]
            want, got = want[1:], got[1:]
        for steps, w, g in zip(streams, want, got):
            # Every forward commits once with the single-stage count, except
            # a final decode that PP2 may drop (the next request starts at 0
            # or after restore_prefix, which reset the protocol).
            assert w == [
                (start, count) if not prefill else (start, n)
                for start, n, prefill, count in steps
            ], seed
            assert g == w or (not steps[-1][2] and g == w[:-1]), seed
        # The sampled counts always arrive first: no deferred-delta commit.
        assert pp2.deferred == 0, seed
        for states in (single.model.states, pp2.model.states):
            assert all(s.commits == states[0].commits for s in states)


# ------------------------------------------- async scheduling at request end


class _Engine:
    """EngineCore.step_with_batch_queue (core.py) with the PP2 batch queue
    (MultiprocExecutor.max_concurrent_batches = pp_size = 2) over a one-request
    scheduler (max_num_seqs=1), driving the stage-1 model above. The scheduler
    learns that a request ended only when it applies that step's output (stop
    token, max_tokens), or on an abort (stop strings, client disconnect) that
    can land while a step is in flight. A waiting request is admitted in the
    scheduler output that carries the previous one's finish.

    Each schedule() is one execute_model on every rank, so stage 1's PPHandler
    consumes the slot of call c in call c + 2. Both MRV2 modes put a request's
    next decode at least pp_size calls after its previous step, so that slot
    is consumed (or excluded, once the request is freed) before the forward.

    Modes:
    * "sync": Scheduler. A decode needs the previous step applied (its sampled
      token and drafts); prefill chunks pipeline.
    * "async": AsyncScheduler under MRV2 (async_scheduler.py): a final prefill
      chunk or a decode sets next_decode_eligible_step = current_step + pp_size;
      prefill chunks pipeline.
    * "v1_async": the same without that rule (the V1 runner's async path): the
      next decode is scheduled with placeholders before the previous output
      is applied, so it can follow the step that ended the request. That is
      V1's crash at request end: the stateful clip leaves the whole row -1 and
      _pp_check_token_ids raises (seen in a PP2 async serve log).
    * "lookahead": a one-step lookahead design (not built): "v1_async"
      scheduling, where the step
      after a request's last one is one phantom decode at the position the
      worker computes, and its output is ignored (the request has ended)."""

    def __init__(self, stage, plans, rng, mode, abort_rate):
        self.stage, self.rng, self.mode = stage, rng, mode
        self.abort_rate = abort_rate
        self.waiting = deque(plans)
        self.running = None
        self.current_step = 0
        self.finished_pending = False
        self.executed = {}  # rid -> steps run on the workers
        self.beyond_end = 0  # steps scheduled after the request's last one
        self.unapplied_at_decode = 0  # decodes scheduled ahead of an output

    def has_requests(self):
        return self.running is not None or bool(self.waiting) or self.finished_pending

    def _end(self, run):
        run.done = True
        self.running = None
        self.finished_pending = True

    def schedule(self):
        self.current_step += 1
        out = SimpleNamespace(
            step=None, req=None, add=None, run=None, finished=self.finished_pending
        )
        self.finished_pending = False
        if self.running is None and self.waiting:
            req, steps = self.waiting.popleft()
            self.running = SimpleNamespace(
                req=req,
                steps=steps,
                next=0,
                applied=0,
                eligible=0,
                done=False,
                phantom=False,
            )
            self.executed[req.rid] = []
            out.add = req
        run = self.running
        if run is None:
            return out
        if run.next >= len(run.steps):
            # The last step ran but its output is not applied yet.
            if self.mode == "lookahead" and not run.phantom:
                last = run.steps[-1]
                start = last[0] + (last[1] if last[2] else last[3])
                run.phantom = True
                self.beyond_end += 1
                out.step = (start, 1 + K, False, 1 + self.rng.randint(0, K))
                out.req, out.run = run.req, run
                return out
            if self.mode in ("sync", "lookahead") or self.current_step < run.eligible:
                return out
            self.beyond_end += 1
            return out
        step = run.steps[run.next]
        in_flight = run.applied < run.next
        if self.mode == "sync":
            if in_flight and not (step[2] and run.steps[run.next - 1][2]):
                return out
        elif self.mode == "async" and self.current_step < run.eligible:
            return out
        if not step[2]:
            self.unapplied_at_decode += in_flight
        if self.mode == "async" and (not step[2] or step[3] == 1):
            run.eligible = self.current_step + 2  # pp_size
        run.next += 1
        out.step, out.req, out.run = step, run.req, run
        return out

    def apply(self, out):
        """update_from_output: a finished or aborted request ignores it."""
        run = out.run
        if run is None or run.done:
            return
        run.applied += 1
        if run.applied == len(run.steps):
            self._end(run)

    def run(self):
        queue = deque()
        for _ in range(20000):
            if not (self.has_requests() or queue):
                return
            run = self.running
            if (
                run is not None
                and run.applied < run.next
                and not run.steps[run.next - 1][2]
                and self.rng.random() < self.abort_rate
            ):
                self._end(run)  # abort with a decode in flight
            if self.has_requests():
                out = self.schedule()
                if self.mode != "v1_async":
                    self.stage.execute(
                        out.step, out.req, add=out.add, finished=out.finished
                    )
                if out.step is not None:
                    self.executed[out.req.rid].append(out.step)
                queue.appendleft(out)
                # Return without blocking when a model ran, the queue has room
                # and the oldest batch is not done. With a queue of 2, room
                # means this batch is the only one, and FutureWrapper.done()
                # turns True only inside a result() call, so it is not done.
                if out.step is not None and len(queue) < 2:
                    continue
            self.apply(queue.pop())
        raise AssertionError("engine did not drain")


def _recording_stage(monkeypatch, ring_depth=2):
    stage = _Stage(monkeypatch, 20, ring_depth)
    stage.consumed = []  # num_sampled of every ring slot stage 1 applied
    postprocess = stage._postprocess

    def recorded(num_sampled):
        if isinstance(num_sampled, torch.Tensor):
            stage.consumed.append(int(num_sampled[0]))
        postprocess(num_sampled)

    stage._postprocess = recorded
    return stage


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_pp2_request_end_under_mrv2_scheduling(monkeypatch, mode):
    """The last request ends (stop token, max_tokens, abort) while stage 1
    already ran its final forward: no step follows the request's end, stage 1
    never consumes a row without a sampled token (V1's all -1 row), commits
    match the steps that ran (a final decode may be dropped), and the next
    request starts clean. Includes runs where no request follows."""
    cases = Counter()
    for seed in range(300):
        rng = random.Random(seed)
        reqs = [_request(rng, f"r{seed}-{i}") for i in range(rng.randint(1, 4))]
        plans = [(req, _steps(rng, req)) for req in reqs]
        stage = _recording_stage(monkeypatch)
        engine = _Engine(stage, plans, rng, mode, abort_rate=rng.choice([0, 0.2]))
        engine.run()
        assert engine.beyond_end == 0, seed
        assert engine.unapplied_at_decode == 0, seed
        assert all(n >= 1 for n in stage.consumed), seed
        assert stage.deferred == 0, seed
        segments = stage.segments()
        assert len(segments) == len(reqs), seed
        for index, (req, got) in enumerate(zip(reqs, segments)):
            ran = engine.executed[req.rid]
            want = [
                (start, n) if prefill else (start, count)
                for start, n, prefill, count in ran
            ]
            assert got == want or (not ran[-1][2] and got == want[:-1]), seed
            last = "last" if index == len(reqs) - 1 else "followed"
            end = "decode committed" if got == want else "decode dropped"
            cases[f"{last}, final {end if not ran[-1][2] else 'prefill'}"] += 1
            cases["aborted"] += len(ran) < len(plans[index][1])
            cases["prefix hit after a request"] += index > 0 and req.cached > 0
    # Every request-end case occurs: the last request's final decode still in
    # stage 1's ring when it finishes (committed by the empty step that
    # carries the finish) or is aborted (dropped), with and without a successor.
    for case in ("last", "followed"):
        assert cases[f"{case}, final decode committed"] > 0, cases
        assert cases[f"{case}, final decode dropped"] > 0, cases
    assert cases["aborted"] > 0 and cases["prefix hit after a request"] > 0, cases


def test_pp2_request_end_v1_async_reaches_the_crash_path(monkeypatch):
    """Without MRV2's eligibility rule the same engine schedules a decode after
    the step that ended the request: the simulation reaches V1's crash path."""
    hits = 0
    for seed in range(100):
        rng = random.Random(seed)
        reqs = [_request(rng, f"r{seed}-{i}") for i in range(3)]
        plans = [(req, _steps(rng, req)) for req in reqs]
        engine = _Engine(None, plans, rng, "v1_async", abort_rate=0)
        engine.run()
        hits += engine.beyond_end > 0
    assert hits > 0


def _lookahead_run(monkeypatch, seed, ring_depth):
    rng = random.Random(seed)
    reqs = [_request(rng, f"r{seed}-{i}") for i in range(rng.randint(1, 4))]
    plans = [(req, _steps(rng, req)) for req in reqs]
    stage = _recording_stage(monkeypatch, ring_depth)
    engine = _Engine(stage, plans, rng, "lookahead", abort_rate=rng.choice([0, 0.2]))
    engine.run()
    return reqs, plans, stage, engine


def test_pp2_lookahead_design_commits(monkeypatch):
    """Design check for the one-step lookahead (LOOKAHEAD-DESIGN.md; not
    built). With a stage-1 ring of depth 1 the commit protocol holds: decodes
    run ahead of their predecessor's output, and the phantom decode after a
    request's last step is committed past the end (the next add_request
    resets it, or restore_prefix rewinds it from token ids) or dropped. Not
    modelled here: compute_need_sampled_mask on the optimistic CPU
    num_computed (the sim uses exact positions) and the window-ring reach."""
    cases = Counter()
    for seed in range(300):
        reqs, plans, stage, engine = _lookahead_run(monkeypatch, seed, 1)
        cases["phantom decodes"] += engine.beyond_end
        cases["decodes ahead of an output"] += engine.unapplied_at_decode
        assert all(n >= 1 for n in stage.consumed), seed
        assert stage.deferred == 0, seed
        segments = stage.segments()
        assert len(segments) == len(reqs), seed
        for (req, steps), got in zip(plans, segments):
            ran = engine.executed[req.rid]
            real, phantom = ran[: len(steps)], ran[len(steps) :]
            want = [(a, n) if p else (a, c) for a, n, p, c in real]
            if got == want:
                cases["committed"] += 1
            elif phantom and got == want + [(a, c) for a, _, _, c in phantom]:
                cases["phantom committed past the end"] += 1
            else:
                assert not real[-1][2] and got == want[:-1], seed
                cases["final decode dropped"] += 1
    assert all(
        cases[case] > 0
        for case in (
            "phantom decodes",
            "decodes ahead of an output",
            "phantom committed past the end",
        )
    ), cases


def test_pp2_lookahead_needs_a_depth1_ring(monkeypatch):
    """With MRV2's stock ring (depth pp_size = 2) the lookahead runs a forward
    before its predecessor's sampled counts arrive: the device protocol traps
    or StageCommit raises."""
    failures = 0
    for seed in range(100):
        try:
            _lookahead_run(monkeypatch, seed, 2)
        except AssertionError as error:
            assert "device trap" in str(error), error
            failures += 1
        except RuntimeError as error:
            assert "V4.1 PP stage 1" in str(error), error
            failures += 1
    assert failures > 50


# --------------------------------------------------------- keyed verification


@needs_kernels
def test_greedy_verify_matches_v1_token_matching():
    # MRV2 rows [t0, d1..dk] vs V1's draft/target/bonus indices.
    from vllm.v1.sample.rejection_sampler import token_match_sample
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    from vllm.v1.worker.gpu.spec_decode.sm70_greedy_verify import greedy_verify

    rng = random.Random(0)
    for _ in range(40):
        drafts = [[rng.randint(0, 3) for _ in range(rng.choice([0, K]))]]
        drafts += [[rng.randint(0, 3) for _ in range(K)] for _ in range(2)]
        rows = sum(len(d) + 1 for d in drafts)
        samples = torch.tensor([rng.randint(0, 3) for _ in range(rows)])
        input_ids, targets, bonus, cu, start = [], [], [], [0], 0
        for d in drafts:
            input_ids += [9] + d
            targets += list(range(start, start + len(d)))
            bonus.append(start + len(d))
            start += len(d) + 1
            cu.append(start)
        metadata = SpecDecodeMetadata(
            draft_token_ids=torch.tensor(sum(drafts, []), dtype=torch.int32),
            num_draft_tokens=[len(d) for d in drafts],
            cu_num_draft_tokens=torch.tensor(
                np.cumsum([len(d) for d in drafts]), dtype=torch.int32
            ),
            cu_num_sampled_tokens=torch.tensor(cu[1:], dtype=torch.int32),
            target_logits_indices=torch.tensor(targets, dtype=torch.int32),
            bonus_logits_indices=torch.tensor(bonus, dtype=torch.int32),
            logits_indices=torch.arange(rows, dtype=torch.int32),
        )
        v1 = token_match_sample(metadata, samples)
        sampled, num_sampled = greedy_verify(
            samples,
            torch.tensor(input_ids, dtype=torch.int32),
            torch.tensor(cu, dtype=torch.int32),
            K,
        )
        for r in range(len(drafts)):
            want = [t for t in v1[r].tolist() if t != -1]
            assert sampled[r, : num_sampled[r]].tolist() == want


class _Sampler:
    """MRV2 sampler states for temperature-only requests; the processing is
    the temperature division (Sampler.apply_sampling_params' only step)."""

    def __init__(self, temperature, vocab):
        n = len(temperature)
        t = torch.tensor(temperature, dtype=torch.float32)
        self.sampling_states = SimpleNamespace(
            temperature=SimpleNamespace(np=t.numpy(), gpu=t),
            top_k=SimpleNamespace(np=np.full(n, vocab)),
            top_p=SimpleNamespace(np=np.ones(n, dtype=np.float32)),
            min_p=SimpleNamespace(np=np.zeros(n, dtype=np.float32)),
            vocab_size=vocab,
            max_num_logprobs=lambda idx: -1,
        )
        self.logprob_token_ids_state = SimpleNamespace(max_num_token_ids=lambda i: 0)
        self.logit_bias_state = SimpleNamespace(use_logit_bias=np.zeros(n, bool))
        self.penalties_state = SimpleNamespace(use_penalty=np.zeros(n, bool))
        self.bad_words_state = SimpleNamespace(
            num_bad_words=SimpleNamespace(np=np.zeros(n, np.int32))
        )
        self.compute_nans = False

    def apply_sampling_params(self, logits, expanded, idx_np, pos, ids, local_pos):
        t = self.sampling_states.temperature.gpu[expanded]
        t = torch.where(t > 0, t, torch.ones_like(t))
        return logits.float() / t.unsqueeze(-1)


def _keyed_setup(temperature, local):
    vocab, hidden = 300, 16
    weight = torch.randn(hidden, vocab, generator=torch.Generator().manual_seed(1))
    num_reqs = len(temperature)
    rows = num_reqs * (K + 1)
    model = SimpleNamespace(
        compute_logits=lambda h: h @ weight,
        local_logits=lambda h, n: (h @ weight)[:, :n],
    )
    spec = object.__new__(v41_spec.V41DSparkSpeculator)
    spec.draft_logits = None
    spec._paths_logged = set()
    spec._local_check = [0, 0]
    spec._keyed_shard = (
        coupled_draw.KeyedShard(0, vocab, 0, 1, lambda x: x) if local else None
    )
    keys = torch.tensor([11, 22, 33][:num_reqs], dtype=torch.int64)
    spec._model_state = SimpleNamespace(keys_gpu=keys)
    rejection = SimpleNamespace(
        sampler=_Sampler(temperature, vocab),
        num_speculative_steps=K,
        synthetic_conditional_rates=None,
    )
    expanded = torch.arange(num_reqs).repeat_interleave(K + 1)
    batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_draft_tokens=num_reqs * K,
        idx_mapping_np=np.arange(num_reqs, dtype=np.int32),
        expanded_idx_mapping=expanded,
        expanded_local_pos=torch.arange(K + 1).repeat(num_reqs),
        logits_indices=torch.arange(rows),
        positions=torch.arange(rows, dtype=torch.int64) + 40,
        input_ids=torch.zeros(rows, dtype=torch.int32),
        cu_num_logits=torch.arange(num_reqs + 1, dtype=torch.int32) * (K + 1),
    )
    hidden_states = torch.randn(rows, hidden, generator=torch.Generator())
    return spec, model, rejection, batch, hidden_states, keys, expanded


@needs_kernels
@pytest.mark.parametrize("local", [False, True])
def test_keyed_verify_token_matching(monkeypatch, local):
    monkeypatch.setattr(coupled_draw, "ENABLED", True)
    monkeypatch.setenv("VLLM_V41_COUPLED_LOCAL", "1" if local else "0")
    temperature = [0.7, 1.0, 0.0] if not local else [0.7, 1.0]
    spec, model, rejection, batch, h, keys, expanded = _keyed_setup(temperature, local)
    # Reference: the target's keyed samples at output position P = pos + 1.
    t = rejection.sampler.sampling_states.temperature.gpu[expanded]
    processed = rejection.sampler.apply_sampling_params(
        model.compute_logits(h), expanded, None, None, None, None
    )
    target = coupled_draw.keyed_argmax(
        processed, t, keys[expanded], batch.positions + 1
    )
    accepted = [2, K, 0][: len(temperature)]
    for r, a in enumerate(accepted):
        base = r * (K + 1)
        drafts = target[base : base + K].clone()
        if a < K:
            drafts[a] = (drafts[a] + 1) % 300  # first mismatch at draft a
        batch.input_ids[base + 1 : base + K + 1] = drafts.to(torch.int32)
    out = spec.try_sample_target(model, rejection, h, batch, None)
    assert out is not None and out.num_sampled.tolist() == [a + 1 for a in accepted]
    for r, a in enumerate(accepted):
        base = r * (K + 1)
        want = target[base : base + a + 1].tolist()
        assert out.sampled_token_ids[r, : a + 1].tolist() == want


def test_keyed_verify_fallbacks(monkeypatch):
    spec, model, rejection, batch, h, _, _ = _keyed_setup([0.7], local=False)
    fallback = spec._keyed_verify_fallback
    monkeypatch.setattr(coupled_draw, "ENABLED", False)
    assert fallback(rejection, h, batch, None) == "off"
    monkeypatch.setattr(coupled_draw, "ENABLED", True)
    assert fallback(rejection, h, batch, object()) == "grammar"
    assert fallback(rejection, h[:-1], batch, None) == "layout"
    states = rejection.sampler.sampling_states
    states.max_num_logprobs = lambda idx: 0
    assert fallback(rejection, h, batch, None) == "logprobs"
    states.max_num_logprobs = lambda idx: -1
    rejection.sampler.penalties_state.use_penalty[:] = True
    assert fallback(rejection, h, batch, None) == "penalties"
    rejection.sampler.penalties_state.use_penalty[:] = False
    states.temperature.np[:] = 0.0
    assert fallback(rejection, h, batch, None) == "all_greedy"
    assert spec.try_sample_target(model, rejection, h, batch, None) is None
    states.temperature.np[:] = 0.7
    assert fallback(rejection, h, batch, None) is None


def test_verify_truncation_guard(monkeypatch):
    import sys

    import vllm.model_executor.layers.quantization as quantization

    name = "vllm.model_executor.layers.quantization.gguf_shared_runtime"
    gguf_shared_runtime = SimpleNamespace()
    monkeypatch.setitem(sys.modules, name, gguf_shared_runtime)
    monkeypatch.setattr(
        quantization, "gguf_shared_runtime", gguf_shared_runtime, raising=False
    )
    trunc = torch.full((1,), 1 << 20, dtype=torch.int32)
    monkeypatch.setattr(
        gguf_shared_runtime, "current_trunc", lambda d: trunc, raising=False
    )
    monkeypatch.delenv("VLLM_V41_SWAP_FREE", raising=False)
    monkeypatch.setenv("VLLM_V41_TRUNC_CUTOFF", "0")
    v41_spec._raise_if_verify_truncation("cpu")  # planner that never truncates
    for name, value in (("VLLM_V41_SWAP_FREE", "1"), ("VLLM_V41_TRUNC_CUTOFF", "4")):
        monkeypatch.setenv(name, value)
        with pytest.raises(RuntimeError, match="masked only by the V1 runner"):
            v41_spec._raise_if_verify_truncation("cpu")
    # A cold set of zero has no planner: current_trunc is None.
    monkeypatch.setattr(gguf_shared_runtime, "current_trunc", lambda d: None)
    v41_spec._raise_if_verify_truncation("cpu")


# ---------------------------------------------------------------- draft order


class _Drafter:
    def __init__(self, greedy=None, shard=None, fused=None, vocab=50):
        self.calls, self.vocab = [], vocab
        self._greedy, self._shard, self._fused = greedy, shard, fused

    def greedy_draft_tokens(self, hidden, anchor):
        self.calls.append(("greedy", anchor.tolist()))
        return self._greedy

    def coupled_shard_draft_tokens(self, hidden, anchor, temp, keys, positions):
        self.calls.append(("shard", positions.tolist()))
        return self._shard

    def compute_logits(self, hidden):
        self.calls.append(("logits", hidden.shape[0]))
        return torch.zeros(hidden.shape[0], self.vocab)

    def coupled_draft_tokens(self, base, prev, temp, keys, positions, scale):
        self.calls.append(("fused", prev.tolist(), positions.tolist()))
        return self._fused

    def markov_embed(self, prev):
        return prev

    def markov_bias(self, embed):
        return torch.zeros(embed.shape[0], self.vocab)

    def map_draft_to_target(self, ids):
        return ids + 1000


def _draft_spec(drafter, all_greedy):
    spec = object.__new__(v41_spec.V41DSparkSpeculator)
    spec.model = drafter
    spec.draft_block = K
    spec.draft_logits = None
    spec._paths_logged = set()
    spec._round = (all_greedy, torch.zeros(1, dtype=torch.int64))
    spec._run_model = lambda *a: torch.randn(K, 4)
    spec.sample_indices = torch.arange(K)
    spec.input_buffers = SimpleNamespace(input_ids=torch.tensor([42, 0, 0, 0, 0]))
    spec._anchor_idx = torch.arange(1) * K
    spec.temperature = torch.tensor([0.8])
    spec._model_state = SimpleNamespace(keys_gpu=torch.tensor([5]))
    spec.sample_pos = torch.arange(K, dtype=torch.int64) + 101
    spec.sample_idx_mapping = torch.zeros(K, dtype=torch.int32)
    spec.draft_tokens = torch.zeros(1, K, dtype=torch.int64)
    spec._sample_step = lambda logits, idx, pos, step: torch.full((1,), 7 + step)
    return spec


def test_draft_order(monkeypatch):
    run = v41_spec.V41DSparkSpeculator._generate_draft
    metadata = {"layer": object()}
    tokens = torch.arange(K) + 500
    # Greedy B=1: the captured greedy graph.
    spec = _draft_spec(_Drafter(greedy=tokens), all_greedy=True)
    run(spec, 1, K, metadata, None, None)
    assert spec.model.calls == [("greedy", [42])]
    assert spec.draft_tokens.tolist() == [tokens.tolist()]
    # Coupled (knob on, not greedy): the shard graph at P = positions.
    monkeypatch.setattr(coupled_draw, "ENABLED", True)
    spec = _draft_spec(_Drafter(shard=tokens), all_greedy=False)
    run(spec, 1, K, metadata, None, None)
    assert spec.model.calls == [("shard", list(range(101, 101 + K)))]
    # No shard graph: the fused kernels from the anchor, mapped to target ids.
    spec = _draft_spec(_Drafter(fused=tokens.view(1, K)), all_greedy=False)
    run(spec, 1, K, metadata, None, None)
    kinds = [c[0] for c in spec.model.calls]
    assert kinds == ["shard", "logits", "fused"]
    assert spec.model.calls[2][1:] == ([42], [list(range(101, 101 + K))])
    assert spec.draft_tokens.tolist() == [(tokens + 1000).tolist()]
    # Knob off: the MRV2 per-step loop.
    monkeypatch.setattr(coupled_draw, "ENABLED", False)
    spec = _draft_spec(_Drafter(), all_greedy=False)
    run(spec, 1, K, metadata, None, None)
    assert [c[0] for c in spec.model.calls] == ["logits"]
    assert spec.draft_tokens.tolist() == [[7 + s for s in range(K)]]


class _Log:
    def __init__(self):
        self.records = []

    def info(self, message, *args):
        self.records.append(("INFO", message % args))

    def warning(self, message, *args):
        self.records.append(("WARNING", message % args))

    def warnings(self):
        return [m for level, m in self.records if level == "WARNING"]


def test_missed_fast_paths_warn_once(monkeypatch):
    # main 11-03: a coupled path that does not engage logs a WARNING.
    log = _Log()
    monkeypatch.setattr(v41_spec, "logger", log)
    monkeypatch.setattr(coupled_draw, "ENABLED", True)
    # The per-step loop's Triton draw (kernel tests cover it): plain argmax.
    monkeypatch.setattr(
        coupled_draw, "keyed_argmax", lambda logits, *a: logits.argmax(-1)
    )
    monkeypatch.setenv("VLLM_V41_COUPLED_SHARD", "1")
    run = v41_spec.V41DSparkSpeculator._generate_draft
    metadata = {"layer": object()}
    tokens = torch.arange(K) + 500
    # Shard graph runs: info only.
    spec = _draft_spec(_Drafter(shard=tokens), all_greedy=False)
    run(spec, 1, K, metadata, None, None)
    assert log.warnings() == []
    # Shard graph missing with VLLM_V41_COUPLED_SHARD=1: one warning per
    # speculator, the fused kernels still run.
    spec = _draft_spec(_Drafter(fused=tokens.view(1, K)), all_greedy=False)
    run(spec, 1, K, metadata, None, None)
    run(spec, 1, K, metadata, None, None)
    assert len(log.warnings()) == 1
    assert "VLLM_V41_COUPLED_SHARD MRV2 fallback" in log.warnings()[0]
    # Shard knob off: the fused kernels are the configured path, no warning.
    monkeypatch.setenv("VLLM_V41_COUPLED_SHARD", "0")
    spec = _draft_spec(_Drafter(fused=tokens.view(1, K)), all_greedy=False)
    run(spec, 1, K, metadata, None, None)
    assert len(log.warnings()) == 1
    # No fused kernels either: the per-step loop warns.
    spec = _draft_spec(_Drafter(), all_greedy=False)
    run(spec, 1, K, metadata, None, None)
    assert "keyed drafts, per-step loop" in log.warnings()[-1]
    # Greedy B=1 without the greedy graph warns.
    spec = _draft_spec(_Drafter(), all_greedy=True)
    run(spec, 1, K, metadata, None, None)
    assert "greedy B=1 draft graph did not run" in log.warnings()[-1]

    # Verify: a skipped keyed step warns once per reason; greedy steps do not.
    spec, model, rejection, batch, h, _, _ = _keyed_setup([0.7], local=False)
    before = len(log.warnings())
    states = rejection.sampler.sampling_states
    states.max_num_logprobs = lambda idx: 0
    for _ in range(2):
        assert spec.try_sample_target(model, rejection, h, batch, None) is None
    assert len(log.warnings()) == before + 1
    assert "verification off for steps with logprobs" in log.warnings()[-1]
    states.max_num_logprobs = lambda idx: -1
    states.temperature.np[:] = 0.0
    assert spec.try_sample_target(model, rejection, h, batch, None) is None
    assert len(log.warnings()) == before + 1

    # One grep token for every missed path.
    assert all("MRV2 fallback:" in w for w in log.warnings())

    # COUPLED_LOCAL=1 that cannot apply names its reason.
    states.temperature.np[:] = 0.7
    local_shard = spec._local_shard
    monkeypatch.setenv("VLLM_V41_COUPLED_LOCAL", "0")
    assert local_shard(model, rejection.sampler, batch) == (None, None)
    monkeypatch.setenv("VLLM_V41_COUPLED_LOCAL", "1")
    states.top_p.np[:] = 0.9
    assert local_shard(model, rejection.sampler, batch) == (None, "top_p")
    states.top_p.np[:] = 1.0
    spec._keyed_shard = None  # built: the model has no shard layout
    assert local_shard(model, rejection.sampler, batch) == (
        None,
        "no vocabulary shard layout",
    )
    shard = coupled_draw.KeyedShard(0, 300, 0, 1, lambda x: x)
    spec._keyed_shard = shard
    assert local_shard(model, rejection.sampler, batch) == (shard, None)


def test_propose_sets_round_from_host_temperature(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    seen = []
    monkeypatch.setattr(
        DSparkSpeculator, "propose", lambda self, *a, **k: seen.append(self._round)
    )
    spec = object.__new__(v41_spec.V41DSparkSpeculator)
    spec._model_state = SimpleNamespace(temperature_np=np.array([0.0, 0.9]))
    batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping_np=np.array([1], dtype=np.int32),
        idx_mapping=torch.tensor([1]),
    )
    spec.propose(batch)
    batch.idx_mapping_np[0] = 0
    spec.propose(batch)
    spec.propose(batch, dummy_run=True)
    assert [r if r is None else r[0] for r in seen] == [False, True, None]
    assert spec._round is None
