# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-preservation gates for the first SM70 KV codec extraction.

These run without importing vLLM, Torch, Triton, or optional extensions.
GPU and model gates are recorded separately; AST equality cannot replace them.
"""

import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import regex as re

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/flash_v100_kv_refactor.json").read_text()
)
BACKEND = ROOT / "vllm/v1/attention/backends/flash_attn_v100.py"
METADATA = BACKEND.parent / "flash_v100/metadata.py"
CODEC = BACKEND.parent / "flash_v100/codec.py"
MASKING = BACKEND.parent / "flash_v100/masking.py"
REFERENCE = BACKEND.parent / "flash_v100/reference.py"
CACHE_VIEW = BACKEND.parent / "flash_v100/cache_view.py"
TRITON_WRITER = ROOT / "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py"
FUSED_WRITER = ROOT / "vllm/model_executor/layers/attention/sm70_qwen38_qk_rope.py"


@pytest.mark.parametrize(
    ("group", "path"),
    [
        ("backend_ast", BACKEND),
        ("metadata_ast", METADATA),
        ("codec_ast", CODEC),
        ("masking_ast", MASKING),
        ("reference_ast", REFERENCE),
        ("cache_view_ast", CACHE_VIEW),
        ("decode_policy_ast", BACKEND.parent / "flash_v100/decode_policy.py"),
        ("triton_writer_host_ast", TRITON_WRITER),
        ("qwen_rope_host_ast", FUSED_WRITER),
        ("qwen_rope_encoder_ast", TRITON_WRITER.parent / "kv_codec.py"),
    ],
)
def test_original_numerics_and_dispatch_are_unchanged(group, path):
    nodes = {
        node.name: node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    for name, expected in FIXTURE[group].items():
        assert name in nodes, f"Removed original symbol {name}"
        actual = hashlib.sha256(
            ast.dump(nodes[name], include_attributes=False).encode()
        ).hexdigest()
        assert actual == expected, f"Original numerical/dispatch body changed: {name}"


def test_metadata_reexports_keep_existing_imports_working():
    imports = {
        name.asname or name.name
        for node in ast.parse(BACKEND.read_text()).body
        if isinstance(node, ast.ImportFrom)
        and node.module == "vllm.v1.attention.backends.flash_v100.metadata"
        for name in node.names
    }
    assert set(FIXTURE["metadata_ast"]) <= imports


def test_codec_reexports_keep_existing_imports_working():
    imports = {
        name.asname or name.name
        for node in ast.parse(BACKEND.read_text()).body
        if isinstance(node, ast.ImportFrom)
        and node.module == "vllm.v1.attention.backends.flash_v100.codec"
        for name in node.names
    }
    assert set(FIXTURE["codec_ast"]) <= imports


@pytest.mark.parametrize(
    "module", ["masking", "reference", "cache_view", "decode_policy"]
)
def test_mask_and_reference_reexports_keep_existing_imports_working(module):
    imports = {
        name.asname or name.name
        for node in ast.parse(BACKEND.read_text()).body
        if isinstance(node, ast.ImportFrom)
        and node.module == f"vllm.v1.attention.backends.flash_v100.{module}"
        for name in node.names
    }
    assert set(FIXTURE[f"{module}_ast"]) <= imports


def test_decode_policy_constants_match_immutable_source():
    baseline = subprocess.check_output(
        [
            "git",
            "show",
            f"{FIXTURE['base_sha']}:vllm/v1/attention/backends/flash_attn_v100.py",
        ],
        cwd=ROOT,
        text=True,
    )
    names = {"_DEFAULT_DECODE_PARTITION_SIZE", "_VALID_DECODE_PARTITION_SIZES"}
    values = []
    for source in (
        baseline,
        (BACKEND.parent / "flash_v100/decode_policy.py").read_text(),
    ):
        values.append(
            {
                target.id: ast.literal_eval(node.value)
                for node in ast.parse(source).body
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Name) and target.id in names
            }
        )
    assert set(values[0]) == names
    assert values[0] == values[1]


def test_shared_codec_include_is_source_complete():
    canonical = ROOT / "flash-attention-v100/kernel/kv_codec.cuh"
    for path in (
        ROOT / "flash-attention-v100/kernel/fp8_kv_utils.cuh",
        ROOT / "csrc/attention/sm70_grouped_long/kernel/fp8_kv_utils.cuh",
    ):
        directive = next(
            line
            for line in path.read_text().splitlines()
            if line.startswith("#include")
        )
        target = directive.split('"')[1]
        assert (path.parent / target).resolve() == canonical
        assert "__device__" not in path.read_text(), (
            "Duplicated converter implementation"
        )
    assert (
        "recursive-include flash-attention-v100/kernel *.cu *.cuh *.h"
        in (ROOT / "MANIFEST.in").read_text()
    )
    assert (
        "recursive-include kernel *.cu *.cuh *.h"
        in (ROOT / "flash-attention-v100/MANIFEST.in").read_text()
    )


def test_cuda_dtype_aliases_preserve_each_entry_point(tmp_path):
    header = (ROOT / "flash-attention-v100/kernel/kv_codec.cuh").read_text()
    start = header.index("inline int kv_cache_dtype_code_from_string(")
    end = header.index("\n}\n", start) + 2
    parser = header[start:end]
    constants = re.findall(r"constexpr int KV_CACHE_DTYPE_\w+ = \d+;", header)
    assert len(constants) == 3
    original = FIXTURE["original_cuda_dtype_parsers"]
    sources = []
    policies = []
    for index, (name, body) in enumerate(original.items()):
        source = (ROOT / name).read_text()
        pattern = (
            r"int kv_cache_dtype_code_from_string\(const std::string& kv_cache_dtype\) "
            r"\{.*?\n\}"
        )
        match = re.search(pattern, source, re.DOTALL)
        assert match is not None
        policy = "false" if "fused_mha_forward_paged" in name else "true"
        expected = (
            "int kv_cache_dtype_code_from_string(const std::string& kv_cache_dtype) {\n"
            "  return flash_v100::kv_cache_dtype_code_from_string(kv_cache_dtype, "
            f"{policy});\n}}"
        )
        assert re.sub(r"\s+", "", match.group()) == re.sub(r"\s+", "", expected)
        restored = source[: match.start()] + body + source[match.end() :]
        assert (
            hashlib.sha256(re.sub(r"\s+", "", restored).encode()).hexdigest()
            == FIXTURE["cuda_dtype_parser_parent_sources_sha256"][name]
        )
        sources.append(f"namespace baseline{index} {{\n{body}\n}}")
        policies.append(
            f"if (baseline{index}::kv_cache_dtype_code_from_string(name) != "
            f"flash_v100::kv_cache_dtype_code_from_string(name, {policy})) return 1;"
        )
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for the CUDA host-parser gate")
    assert compiler is not None
    path = tmp_path / "aliases.cpp"
    path.write_text(
        "#include <string>\n#include <iostream>\nnamespace flash_v100 {\n"
        + "\n".join(constants)
        + "\n"
        + parser
        + "\n}\n"
        + "\n".join(sources)
        + "\nint main() { std::string name; while (std::getline(std::cin, name)) {\n"
        + "\n".join(policies)
        + "\n} }\n"
    )
    executable = tmp_path / "aliases"
    subprocess.run(
        [compiler, "-std=c++17", str(path), "-o", str(executable)], check=True
    )
    names = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "",
        "float32",
        "fp8_ds_mla",
        "fp8_per_token_head",
        "int8",
        "int8_per_token_head",
        "nvfp4",
        "turboquant_k8v4",
        "FP8",
        " auto",
        "auto\0",
        "未知",
    ]
    subprocess.run(
        [str(executable)], input="\n".join(names) + "\n", text=True, check=True
    )


def test_packed_converters_have_one_preserved_implementation():
    source = (ROOT / "flash-attention-v100/kernel/kv_codec.cuh").read_text()
    start = source.index(
        "__device__ __forceinline__ uint32_t\nfp8_e5m2_pair_to_half2_bits"
    )
    end = source.index("\ntemplate <int KV_DTYPE", start)
    preserved = re.sub(r"\s+", "", source[start:end])
    assert (
        hashlib.sha256(preserved.encode()).hexdigest()
        == FIXTURE["packed_converters_sha256"]
    )
    for path in (
        ROOT / "flash-attention-v100/kernel/flash_decode_paged.cu",
        ROOT / "csrc/attention/sm70_grouped_long/kernel/grouped-attention.cu",
    ):
        attention = path.read_text()
        assert "fp8_e4m3fn_pair_to_half2_bits(" not in attention
        assert "KVReader<KV_DTYPE>::template load_half8" in attention
        assert "KVReader<KV_DTYPE>::template half8_from_packed" in attention


def test_native_writer_helpers_and_cache_kernel_bodies_are_preserved():
    header = (ROOT / "csrc/kv_cache_codec.cuh").read_text()
    start = header.index("// Used to copy/convert one element")
    end = header.rindex("\n}  // namespace vllm")
    helpers = header[start:end].replace("KVWriter", "CopyWithScaleOp")
    compact = lambda text: re.sub(r"\s+", "", text)
    assert (
        hashlib.sha256(compact(helpers).encode()).hexdigest()
        == FIXTURE["native_writer_helpers_sha256"]
    )
    source = (ROOT / "csrc/libtorch_stable/cache_kernels.cu").read_text()
    original_include = (
        "#ifdef USE_ROCM\n"
        '  #include "../quantization/w8a8/fp8/amd/quant_utils.cuh"\n'
        "#else\n"
        '  #include "../quantization/w8a8/fp8/nvidia/quant_utils.cuh"\n'
        "#endif"
    )
    source = source.replace('#include "../kv_cache_codec.cuh"', original_include)
    source = source.replace("KVWriter", "CopyWithScaleOp")
    marker = (
        "template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>\n"
        "__global__ void reshape_and_cache_kernel"
    )
    assert source.count(marker) == 1
    restored = source.replace(marker, helpers + "\n" + marker)
    assert (
        hashlib.sha256(compact(restored).encode()).hexdigest()
        == FIXTURE["native_cache_kernels_sha256"]
    )
    assert "recursive-include csrc *" in (ROOT / "MANIFEST.in").read_text()


class _EraseCodecExpressions(ast.NodeTransformer):
    """Guard address/scale stores and launch ABI while extracting encoding math."""

    targets = {"key_tile", "value_tile", "k_scale", "v_scale", "k_q", "v_q"}

    def visit_Assign(self, node):
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in self.targets
        ):
            node.value = ast.Name(id="_codec_expression", ctx=ast.Load())
        return self.generic_visit(node)

    def visit_If(self, node):
        if isinstance(node.test, ast.Name) and node.test.id == "FP8_KV_CACHE":
            targets = {
                target.id
                for item in ast.walk(node)
                if isinstance(item, ast.Assign)
                for target in item.targets
                if isinstance(target, ast.Name)
            }
            if targets in ({"key_tile"}, {"value_tile"}):
                return ast.Assign(
                    targets=[ast.Name(id=next(iter(targets)), ctx=ast.Store())],
                    value=ast.Name(id="_codec_expression", ctx=ast.Load()),
                )
        return self.generic_visit(node)


def test_triton_writer_addressing_and_typed_stores_are_preserved():
    nodes = {
        n.name: n
        for n in ast.parse(TRITON_WRITER.read_text()).body
        if isinstance(n, ast.FunctionDef)
    }
    for name, expected in FIXTURE["triton_writer_address_ast"].items():
        node = _EraseCodecExpressions().visit(nodes[name])
        assert (
            hashlib.sha256(
                ast.dump(node, include_attributes=False).encode()
            ).hexdigest()
            == expected
        )
    codec = ast.parse((TRITON_WRITER.parent / "kv_codec.py").read_text())
    ranges = next(
        n
        for n in codec.body
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == "_PER_TOKEN_HEAD_QUANT_PARAMS"
    )
    assert (
        hashlib.sha256(ast.dump(ranges, include_attributes=False).encode()).hexdigest()
        == FIXTURE["triton_writer_range_ast"]
    )


class _EraseFusedCodecExpressions(ast.NodeTransformer):
    """Only the two legacy cache-scaling expressions move to the codec."""

    def visit_Call(self, node):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "scale_kv_e4m3_per_tensor"
        ):
            values, scale = (ast.unparse(arg) for arg in node.args)
            assert values in {"processed", "value"}
            # K first becomes FP32; V was already FP32 in the original writer.
            if values == "processed":
                values += ".to(tl.float32)"
            return ast.parse(f"tl.div_rn({values}, tl.load({scale}))", mode="eval").body
        return self.generic_visit(node)


def test_fused_writer_math_addresses_stores_and_launch_are_preserved():
    tree = ast.parse(FUSED_WRITER.read_text())
    kernel = next(n for n in tree.body if getattr(n, "name", None) == "_qk_norm_rope")
    restored = _EraseFusedCodecExpressions().visit(kernel)
    assert (
        hashlib.sha256(
            ast.dump(restored, include_attributes=False).encode()
        ).hexdigest()
        == FIXTURE["qwen_rope_kernel_ast"]
    )
    imports = {
        name.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "vllm.v1.attention.ops.kv_codec"
        for name in node.names
    }
    assert "_e4m3_satfinite" in imports


def test_path_matrix_covers_static_and_dynamic_route_sites():
    matrix = json.loads((ROOT / "docs/design/sm70_kv_path_matrix.json").read_text())
    tree = ast.parse(BACKEND.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_record_route"
    ]
    static = {
        node.args[0].value for node in calls if isinstance(node.args[0], ast.Constant)
    }
    dynamic = {
        ast.unparse(node.args[0])
        for node in calls
        if not isinstance(node.args[0], ast.Constant)
    }
    assert static == {row["legacy_route"] for row in matrix["literal_routes"]}
    assert dynamic == set(matrix["dynamic_route_expressions"])
    for row in matrix["literal_routes"] + matrix["additional_paths"]:
        assert set(row["formats"]) == {"fp16", "e4m3", "e5m2", "int8"}
        assert all(
            status
            in {
                "native",
                "bridge",
                "reference",
                "metadata",
                "fallback",
                "unsupported",
                "pending",
                "upstream_only",
                "review",
            }
            for status in row["formats"].values()
        )
        assert row["formats"]["int8"] in {"pending", "upstream_only"}
        assert not any(
            name in row["route"] for name in ("fp8", "e4m3", "e5m2", "fp16", "int8")
        )
    assert all((ROOT / row["source"]).exists() for row in matrix["additional_paths"])
    table = {
        tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        for line in (ROOT / "docs/design/sm70_kv_path_matrix.md")
        .read_text()
        .splitlines()
        if line.startswith("|")
    }
    for row in matrix["literal_routes"]:
        assert (
            row["route"],
            f"`{row['legacy_route']}`",
            *(row["formats"][fmt] for fmt in ("fp16", "e4m3", "e5m2", "int8")),
        ) in table


def test_environment_inventory_covers_backend_reads():
    tree = ast.Module(
        body=[
            node
            for path in [BACKEND, *sorted((BACKEND.parent / "flash_v100").glob("*.py"))]
            for node in ast.parse(path.read_text()).body
        ],
        type_ignores=[],
    )
    names = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "envs"
        and node.attr.startswith("VLLM_")
    }
    names.update(
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"getenv", "get"}
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and node.args[0].value.startswith("VLLM_")
    )
    document = (ROOT / "docs/design/sm70_kv_environment_inventory.md").read_text()
    assert len(names) == 89
    assert all(f"`{name}`" in document for name in names)
