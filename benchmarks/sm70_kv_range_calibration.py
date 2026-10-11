# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect independent KV ranges on an eager, unquantized-cache reference.

This benchmark extension never changes runtime scales or captured graphs.
Exported ranges belong in a checkpoint loaded by a fresh FP8 engine before
capture. Calibration inputs must be frozen separately from quality inputs.
"""

from benchmarks.sm70_dflash2_prefill_route_audit import PrefillRouteAuditExtension


class KVRangeCalibrationExtension(PrefillRouteAuditExtension):
    def begin_kv_range_calibration(self):
        import torch

        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.attention import Attention

        config = self.vllm_config
        if not config.model_config.enforce_eager:
            raise ValueError("Range calibration requires an eager reference engine")
        if config.cache_config.cache_dtype not in ("auto", "float16", "bfloat16"):
            raise ValueError("Range calibration requires an unquantized reference KV")
        if getattr(self, "_kv_range_handles", None):
            raise RuntimeError("A range collection is already active")

        states, handles = {}, []

        def observe(layer, args, kwargs):
            context = get_forward_context()
            if context.attn_metadata is None:
                return
            if not isinstance(context.slot_mapping, dict):
                raise ValueError("Range collection requires a single microbatch")
            slots = context.slot_mapping.get(layer.layer_name)
            if slots is None:
                raise ValueError("Range collection needs the actual writer slot mask")
            valid = slots >= 0
            state = states[layer.layer_name]
            state["rows"].add_(valid.sum())
            for index, field in enumerate(("query", "key", "value")):
                tensor = args[index] if index < len(args) else kwargs.get(field)
                if tensor is None or tensor.shape[0] < slots.shape[0]:
                    raise ValueError("Range collection needs complete Q/K/V rows")
                rows = tensor[: slots.shape[0]].detach().float().flatten(1)
                finite = torch.isfinite(rows)
                state["nonfinite"][index].logical_or_(
                    ((~finite) & valid[:, None]).any()
                )
                absolute = torch.where(finite & valid[:, None], rows.abs(), 0.0)
                if absolute.numel():
                    torch.maximum(
                        state["absmax"][index],
                        absolute.amax(),
                        out=state["absmax"][index],
                    )

        try:
            for name, layer in self.get_model().named_modules():
                if not isinstance(layer, Attention):
                    continue
                states[layer.layer_name] = {
                    "module": name,
                    "absmax": torch.zeros(3, device=self.device),
                    "nonfinite": torch.zeros(3, device=self.device, dtype=torch.bool),
                    "rows": torch.zeros((), device=self.device, dtype=torch.int64),
                    "ranges": [
                        float(getattr(layer, f"{field}_range"))
                        for field in ("q", "k", "v")
                    ],
                }
                handles.append(
                    layer.register_forward_pre_hook(observe, with_kwargs=True)
                )
        except Exception:
            for handle in handles:
                handle.remove()
            raise
        if not handles:
            raise ValueError("Reference model has no ordinary Attention operators")
        self._kv_range_states, self._kv_range_handles = states, handles
        return {"rank": torch.distributed.get_rank(), "layers": len(states)}

    def end_kv_range_calibration(self):
        import torch

        handles = getattr(self, "_kv_range_handles", ())
        if not handles:
            raise RuntimeError("No range collection is active")
        for handle in handles:
            handle.remove()
        self._kv_range_handles = []
        rows = []
        for name, state in sorted(self._kv_range_states.items()):
            rows.append(
                {
                    "layer": name,
                    "module": state["module"],
                    "absmax": state["absmax"].cpu().tolist(),
                    "nonfinite": state["nonfinite"].cpu().tolist(),
                    "observed_rows": int(state["rows"].cpu()),
                    "ranges": state["ranges"],
                }
            )
        del self._kv_range_states
        return {
            "rank": torch.distributed.get_rank(),
            "layers": rows,
            "scope": "FP16-reference ranges; no runtime scale mutation",
        }
