# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SM70 lifecycle policy and legacy feature warmup bindings."""

from vllm.model_executor.warmup.plan import WarmupTask, warmup_boolean
from vllm.platforms import current_platform


def auxiliary_warmup_enabled(config) -> bool:
    return bool(
        config.kernel_config.sm70_runtime.auxiliary_warmup
        and current_platform.is_device_capability(70)
    )


def proposer_warmup_tasks(proposer) -> list[WarmupTask]:
    tasks = []
    for name in (
        "warmup_sm70_mtp_hotpath_kernels",
        "warmup_sm70_mtp_moe_kernels",
        "warmup_sm70_dflash_hotpath_kernels",
    ):
        run = getattr(proposer, name, None)
        if run is not None:
            tasks.append(WarmupTask(name, run))
    return tasks


def model_convolution_warmup_task(model) -> WarmupTask:
    def run() -> bool:
        modules = getattr(model, "modules", None)
        if modules is not None:
            for module in modules():
                warm = getattr(module, "_warmup_sm70_causal_conv1d_real_state", None)
                if warm is not None and warm():
                    return True
        return False

    return warmup_boolean("gdn_causal_conv1d", run)


def warmup_bound_convolution(forward_context, *, enabled: bool) -> bool:
    if not enabled:
        return False
    for layer in forward_context.values():
        warmup = getattr(layer, "_warmup_sm70_causal_conv1d_real_state", None)
        if warmup is not None and warmup():
            return True
    return False


def warmup_v2_convolution(static_forward_context, logger, *, enabled: bool) -> None:
    # Preserve MRV2's ordering before the rest of the auxiliary tasks.
    if warmup_bound_convolution(static_forward_context, enabled=enabled):
        logger.info_once("SM70 MRV2 GDN causal-conv warmup finished.")


def speculator_warmup_tasks(speculator, dummy_run) -> list[WarmupTask]:
    warmup = getattr(speculator, "warmup_sm70_mtp_moe_kernels", None)
    if warmup is None:
        return []
    return [WarmupTask("mtp_moe", lambda: warmup(dummy_run))]
