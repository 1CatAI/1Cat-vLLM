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
