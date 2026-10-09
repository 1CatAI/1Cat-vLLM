# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tools.pre_commit.check_sm70_linear_policy import violations


def test_reject_registered_variable_outside_compatibility_adapter(tmp_path):
    path = tmp_path / "vllm" / "new_kernel.py"
    path.parent.mkdir()
    path.write_text("import vllm.envs as envs\nx = envs.VLLM_SM70_NVFP4_QPN2\n")
    assert len(violations(path)) == 1
    path.write_text('import os\nx = os.getenv("VLLM_SM70_NVFP4_QPN2", "0")\n')
    assert len(violations(path)) == 1
    path.write_text("x = config.kernel_config.sm70_nvfp4.qpn2\n")
    assert not violations(path)


def test_awq_runtime_alias_read_is_rejected(tmp_path):
    path = tmp_path / "vllm" / "new_awq_kernel.py"
    path.parent.mkdir()
    path.write_text("x = envs.VLLM_SM70_AWQ_MLP_ENGINE\n")
    assert len(violations(path)) == 1


def test_fp8_serialized_loader_and_kernel_cannot_reparse_aliases(tmp_path):
    directory = tmp_path / "vllm"
    directory.mkdir()
    path = directory / "fp8.py"
    path.write_text("class Fp8LinearMethod:\n    policy = envs.VLLM_SM70_FP8_QPN8\n")
    assert len(violations(path)) == 1
    path.write_text("class Fp8Config:\n    admission = envs.VLLM_SM70_FP8_TURBOMIND\n")
    assert not violations(path)
    path = directory / "sm70_fp8.py"
    path.write_text('import os\nx = os.getenv("VLLM_SM70_FP8_QPN8")\n')
    assert len(violations(path)) == 1


def test_dflash_verifier_cannot_bypass_its_engine_policy(tmp_path):
    path = tmp_path / "vllm" / "new_verifier.py"
    path.parent.mkdir()
    path.write_text("enabled = envs.VLLM_SM70_DFLASH2_CONTEXT_PIPELINE\n")
    assert len(violations(path)) == 1
    path.write_text('enabled = os.getenv("VLLM_SM70_DFLASH2_FUSED_GDN_NORM", "0")\n')
    assert len(violations(path)) == 1
    path.write_text('enabled = sm70_dflash2_enabled("context_pipeline", policy)\n')
    assert not violations(path)


def test_moe_executor_cannot_reparse_legacy_policy(tmp_path):
    path = tmp_path / "vllm/model_executor/layers/fused_moe/sm70/new_kernel.py"
    path.parent.mkdir(parents=True)
    path.write_text("x = envs.VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_STAGE_FASTPATH\n")
    assert len(violations(path)) == 1
    path.write_text('x = os.getenv("VLLM_SM70_AWQ_MOE_BATCHED_GEMM")\n')
    assert len(violations(path)) == 1
    path.write_text('label = "VLLM_SM70_AWQ_MOE_BATCHED_GEMM"\nx = plan.w13\n')
    assert not violations(path)


def test_runtime_consumers_cannot_bypass_resolved_engine_policy(tmp_path):
    path = tmp_path / "vllm" / "new_runner.py"
    path.parent.mkdir()
    for expression in (
        "envs.VLLM_USE_AOT_COMPILE",
        'os.getenv("VLLM_SM70_GLM_MHC_PRE_THREADS")',
        'os.environ["VLLM_SM70_AWQ_WARMUP_MAX_M"]',
        'getattr(envs, "VLLM_SM70_QWEN38_FP16_GEMV")',
    ):
        path.write_text(f"choice = {expression}\n")
        assert len(violations(path)) == 1
    path.write_text('label = "VLLM_USE_AOT_COMPILE"\nchoice = policy.aot_compile\n')
    assert not violations(path)


def test_diagnostics_cannot_reparse_initialized_filters(tmp_path):
    path = tmp_path / "vllm" / "new_diagnostics.py"
    path.parent.mkdir()
    for expression in (
        'os.getenv("VLLM_SM70_DUMP_GDN_GRAPH_DIR")',
        "envs.VLLM_SPEC_DUMP_ALIGNMENT",
        "envs.VLLM_DFLASH_PROFILE",
        'os.environ["VLLM_SM70_DUMP_AWQ_MOE_LABELS"]',
    ):
        path.write_text(f"value = {expression}\n")
        assert violations(path)
    path.write_text('value = diagnostics.channels["gdn_graph"].policy.directory\n')
    assert not violations(path)
