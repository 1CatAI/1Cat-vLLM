# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Project actual sequence candidates into the frozen calculation AST oracle.

Only control-flow representation changes here. Every native calculation and
logging statement comes from current source, never a saved branch body.
"""

import ast
import copy

from vllm.v1.attention.backends.flash_v100 import prefill, prefill_candidates

OPS = {
    "supports_bmhd": "self._flash_prefill_paged_supports_dflash2_bmhd",
    "split_pages": "self._flash_prefill_paged_dflash2_split_pages",
    "tree_prefill": "self._flash_v100_ddtree_small_query_prefill_dense",
    "small_query": "self._flash_v100_small_query_prefill_as_decode",
    "allow_rows": "self._prefill_prefix_decode_rows_allowed",
    "decode_rows": "self._run_prefill_prefix_decode_rows",
    "bridge": "self._run_fp8_prefill_bridge",
    "run_paged": "self._run_prefill_paged_call",
    "should_bridge": "self._should_use_fp8_prefill_bridge",
    "should_bfla": "self._should_use_prefill_bfla",
    "should_contig": "self._should_use_prefill_contig_dense",
    "should_gather": "self._should_use_prefill_gather_dense",
    "should_split": "self._should_use_prefill_splitkv",
    "bhmd": "self.flash_attn_bhmd_func",
    "dense": "self.flash_attn_func",
    "paged": "self.flash_attn_prefill_paged",
    "bfla": "self.flash_attn_prefill_paged_bfla",
    "splitkv": "self.flash_attn_prefill_paged_splitkv",
    "uniform": "_dense_prefill._uniform_cu_seqlens",
    "try_fa2": "_dense_prefill._try_sm70_fa2_d256_prefill",
}


def parse(expression):
    return ast.parse(expression, mode="eval").body


class SequenceLocals(ast.NodeTransformer):
    def __init__(self, logs):
        self.logs = logs

    def visit_Expr(self, node):
        call = node.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
            name = call.func.attr
            if name.startswith("log_"):
                assert ast.unparse(call.func.value) == "self.executor.ops"
                assert [ast.unparse(a) for a in call.args] == [
                    "self.executor.config",
                    *(
                        ["fa2_route"]
                        if name == "log_fa2"
                        else [
                            "request.num_seqs",
                            "request.max_query_len",
                            "request.block_size",
                        ]
                        if name == "log_noncausal"
                        else []
                    ),
                ]
                assert not call.keywords
                return [self.visit(copy.deepcopy(n)) for n in self.logs[name].body]
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        owner = ast.unparse(node.value)
        if owner in ("self.executor.ops", "self.ops"):
            return parse(OPS[node.attr])
        if owner == "request":
            return ast.Name(
                "dflash_dump" if node.attr == "dump_enabled" else node.attr, node.ctx
            )
        if owner in (
            "config",
            "config.policy",
            "self.executor.config",
            "self.executor.config.policy",
        ):
            return ast.Attribute(ast.Name("self", ast.Load()), node.attr, node.ctx)
        return self.generic_visit(node)


def sequence_calculations():
    from pathlib import Path

    tree = ast.parse(Path(prefill_candidates.__file__).read_text())
    classes = {
        n.name: {f.name: f for f in n.body if isinstance(f, ast.FunctionDef)}
        for n in tree.body
        if isinstance(n, ast.ClassDef)
    }
    values = {
        ast.unparse(n.targets[0]): n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
    }
    prepared_order = values["PREPARED_CANDIDATES"]
    fallback_order = values["FALLBACK_CANDIDATES"]
    assert isinstance(prepared_order, ast.Tuple)
    assert isinstance(fallback_order, ast.Tuple)
    prepared = [ast.unparse(n) for n in prepared_order.elts]
    fallback = [ast.unparse(n) for n in fallback_order.elts]
    assert prepared == [
        "BflaPrefill",
        "Fa2Prefill",
        "ContiguousBhmdPrefill",
        "ContiguousDensePrefill",
    ]
    assert fallback == ["Fp8BridgePrefill", "SplitKvPrefill", "PagedPrefill"]
    executor = classes["PrefillExecutor"]
    assert ast.unparse(executor["sequence"].body[0]) == (
        "return _plan.execute(request, self.candidates(request))"
    )
    assert len(executor["sequence"].body) == 1
    assert [ast.unparse(n) for n in executor["candidates"].body] == [
        "for candidate in PREPARED_CANDIDATES:\n    yield candidate(self)",
        "self.prepare_fallback(request)",
        "for fallback in FALLBACK_CANDIDATES:\n    yield fallback(self)",
    ]
    logs = {
        n.name: n
        for n in ast.parse(Path(prefill.__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("log_")
    }
    normalizer = SequenceLocals(logs)

    def normalize(nodes):
        return normalizer.visit(ast.Module(copy.deepcopy(nodes), [])).body

    def admit(name):
        body = classes[name]["admit"].body
        assert len(body) == 1 and isinstance(body[0], ast.Return)
        return copy.deepcopy(body[0].value)

    def run(name):
        body = copy.deepcopy(classes[name]["run"].body)
        if name != "ContiguousBhmdPrefill":
            assert ast.unparse(body.pop(0)) == "out_is_destination = False"
        returned = body.pop()
        assert isinstance(returned, ast.Return)
        assert returned.value is not None
        expected = (
            "PrefillResult(None, False, True)"
            if name == "ContiguousBhmdPrefill"
            else "PrefillResult(out_seq, out_is_destination, False)"
        )
        assert ast.unparse(returned.value) == expected
        return body

    def prepared_parts(name, output):
        body = run(name)
        index = next(
            i
            for i, n in enumerate(body)
            if ast.unparse(n) == "self.executor.prepare_fallback(request)"
        )
        decline = body[index - 1]
        assert ast.unparse(decline) == f"if {output} is None:\n    return None"
        assert (
            sum(
                ast.unparse(n) == "self.executor.prepare_fallback(request)"
                for n in body
            )
            == 1
        )
        return body[: index - 1], body[index + 1 :]

    bfla_prep, bfla_run = prepared_parts("BflaPrefill", "bfla_block_mask")
    fa2_prep, fa2_run = prepared_parts("Fa2Prefill", "fa2_paged_out")
    bhmd_prep, bhmd_run = prepared_parts(
        "ContiguousBhmdPrefill", "contig_dense_kv_bhmd"
    )
    dense_prep, dense_run = prepared_parts("ContiguousDensePrefill", "contig_dense_kv")
    assert ast.unparse(admit("ContiguousBhmdPrefill")) == "True"
    assert ast.unparse(admit("ContiguousDensePrefill")) == "request.contig_allowed"
    assert len(bhmd_prep) == 3 and len(dense_prep) == 1
    assert ast.unparse(bhmd_prep[0].targets[0]) == "request.contig_allowed"
    assert ast.unparse(bhmd_prep[1]) == "contig_dense_kv_bhmd = None"
    bhmd_gate = bhmd_prep[2]
    assert isinstance(bhmd_gate, ast.If) and not bhmd_gate.orelse
    assert isinstance(bhmd_gate.test, ast.BoolOp)
    assert ast.unparse(bhmd_gate.test.values[0]) == "request.contig_allowed"
    bhmd_gate.test.values = bhmd_gate.test.values[1:]
    result = ast.parse("bfla_block_mask = None").body
    result += [ast.Assign([ast.Name("use_bfla", ast.Store())], admit("BflaPrefill"))]
    result += [ast.If(parse("use_bfla"), bfla_prep, [])]
    result += ast.parse("fa2_paged_out = None\nfa2_route = None").body
    fa2_condition = ast.BoolOp(
        ast.And(), [parse("bfla_block_mask is None"), *admit("Fa2Prefill").values]
    )
    result += [ast.If(fa2_condition, fa2_prep, [])]
    result += ast.parse("contig_dense_kv = None\ncontig_dense_kv_bhmd = None").body
    condition = ast.BoolOp(
        ast.And(),
        [
            parse("bfla_block_mask is None"),
            parse("fa2_paged_out is None"),
            bhmd_prep[0].value,
        ],
    )
    result += [
        ast.If(
            condition,
            [bhmd_gate, ast.If(parse("contig_dense_kv_bhmd is None"), dense_prep, [])],
            [],
        )
    ]
    guards = copy.deepcopy(executor["prepare_fallback"].body)
    guard_targets = []
    for guard in guards:
        assert isinstance(guard, ast.Assign)
        guard_targets.append(ast.unparse(guard.targets[0]))
    assert len(guards) == 2 and guard_targets == [
        "request.use_splitkv",
        "request.use_fp8_bridge",
    ]
    result += guards
    assert ast.unparse(admit("Fp8BridgePrefill")) == "request.use_fp8_bridge"
    assert ast.unparse(admit("SplitKvPrefill")) == "request.use_splitkv"
    assert ast.unparse(admit("PagedPrefill")) == "True"
    chain = run("PagedPrefill")
    for test, body in reversed(
        [
            (parse("bfla_block_mask is not None"), bfla_run),
            (parse("fa2_paged_out is not None"), fa2_run),
            (parse("contig_dense_kv_bhmd is not None"), bhmd_run + [ast.Continue()]),
            (parse("contig_dense_kv is not None"), dense_run),
            (admit("Fp8BridgePrefill"), run("Fp8BridgePrefill")),
            (admit("SplitKvPrefill"), run("SplitKvPrefill")),
        ]
    ):
        chain = [ast.If(test, body, chain)]
    return normalize(result + chain)


def batch_calculations():
    from pathlib import Path

    tree = ast.parse(Path(prefill_candidates.__file__).read_text())
    classes = {
        n.name: {f.name: f for f in n.body if isinstance(f, ast.FunctionDef)}
        for n in tree.body
        if isinstance(n, ast.ClassDef)
    }
    order = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "BATCH_CANDIDATES"
    )
    assert isinstance(order, ast.Tuple)
    names = [ast.unparse(n) for n in order.elts]
    assert names == [
        "NoncausalBatch",
        "TreeBatch",
        "SmallQueryBatch",
        "DecodeRowsBatch",
    ]
    assert [ast.unparse(n) for n in classes["PrefillExecutor"]["batch"].body] == [
        "result = _plan.try_execute(request, (candidate(self) "
        "for candidate in BATCH_CANDIDATES))",
        "return PrefillBatchResult(False, None, set()) if result is None else result",
    ]
    result = []
    for index, name in enumerate(names):
        admit = classes[name]["admit"].body
        assert len(admit) == 1 and isinstance(admit[0], ast.Return)
        condition = copy.deepcopy(admit[0].value)
        assert condition is not None
        body = copy.deepcopy(classes[name]["run"].body)
        returned = body.pop()
        assert isinstance(returned, ast.Return)
        value = returned.value
        assert (
            isinstance(value, ast.Call)
            and ast.unparse(value.func) == "PrefillBatchResult"
        )
        assert not value.keywords and len(value.args) == 3
        if index < 3:
            assert ast.unparse(value.args[0]) == "True"
            assert ast.unparse(value.args[2]) == "set()"
            body.append(ast.Return(value.args[1]))
        else:
            assert [ast.unparse(a) for a in value.args] == [
                "False",
                "None",
                "decode_rows",
            ]
            result += ast.parse("decode_rows: set[int] = set()").body
        result.append(ast.If(condition, body, []))
    logs = {
        n.name: n
        for n in ast.parse(Path(prefill.__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("log_")
    }
    return SequenceLocals(logs).visit(ast.Module(result, [])).body


class DebugLocals(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if ast.unparse(node.value) == "event":
            callback = {
                "dense": "self.flash_attn_func",
                "torch_reference": "_masks._torch_attention_reference",
                "layer_info": "self._layer_debug_info",
                "scale": "self.scale",
                "kv_cache_dtype": "self.kv_cache_dtype",
            }.get(node.attr)
            if callback:
                return parse(callback)
            return ast.Name(
                "dflash_dump" if node.attr == "dump_enabled" else node.attr, node.ctx
            )
        return self.generic_visit(node)


def prefill_debug_calculations(call):
    from pathlib import Path

    from vllm.v1.attention.backends.flash_v100 import debug_compare

    helper = next(
        n
        for n in ast.parse(Path(prefill.__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "observe_prefill_reference"
    )
    parameters = [a.arg for a in helper.args.args]
    assert not call.keywords
    assert [ast.unparse(a) for a in call.args] == [
        "dflash_dump" if a == "dump_enabled" else a for a in parameters
    ]
    assert len(helper.body) == 1 and isinstance(helper.body[0], ast.Expr)
    emit = helper.body[0].value
    assert (
        isinstance(emit, ast.Call)
        and ast.unparse(emit.func) == "_events.prefill_debug.emit"
    )
    assert len(emit.args) == 1 and not emit.keywords
    event = emit.args[0]
    assert (
        isinstance(event, ast.Call)
        and ast.unparse(event.func) == "_events.PrefillDebugEvent"
    )
    assert not event.keywords
    assert [ast.unparse(a) for a in event.args] == parameters[1:] + [
        "self.kv_cache_dtype",
        "self.scale",
        "self.flash_attn_func",
        "_masks._torch_attention_reference",
        "self._layer_debug_info",
    ]
    tree = ast.parse(Path(debug_compare.__file__).read_text())
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    subscribers = []
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value, ast.Call
        ):
            continue
        registration = statement.value
        if ast.unparse(registration.func) == "_events.prefill_debug.subscribe":
            assert len(registration.args) == 1 and not registration.keywords
            subscribers.append(ast.unparse(registration.args[0]))
    assert subscribers == ["PrefixReferenceObserver()", "PrefixReportObserver()"]
    reference = next(
        n
        for n in classes["PrefixReferenceObserver"].body
        if isinstance(n, ast.FunctionDef) and n.name == "__call__"
    )
    report = next(
        n
        for n in classes["PrefixReportObserver"].body
        if isinstance(n, ast.FunctionDef) and n.name == "__call__"
    )
    assert ast.unparse(reference.body[-1]) == (
        "event.reference = _events.PrefillReference("
        "k_cont, v_cont, ref_out, diff, nan_count)"
    )
    assert [ast.unparse(n) for n in report.body[:3]] == [
        "reference = event.reference",
        "assert reference is not None",
        "k_cont, v_cont, ref_out, diff, nan_count = reference",
    ]
    body = copy.deepcopy(reference.body[:-1] + report.body[3:])
    return DebugLocals().visit(ast.Module(body, [])).body
