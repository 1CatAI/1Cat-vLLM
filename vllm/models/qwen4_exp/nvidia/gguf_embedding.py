# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed GGUF PLE rows with the existing device/host/disk placement policy."""

from typing import cast

import torch
from torch.nn import Parameter

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.ple_offload_layer import is_offload_process
from vllm.model_executor.layers.quantization.gguf import (
    GGUFEmbeddingMethod,
    dequantize_gguf_rows,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.utils import set_weight_attrs
from vllm.transformers_utils.gguf_rows import PackedGGUFRowReader
from vllm.transformers_utils.gguf_tensor_reader import quant_size
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .ple_layer import Qwen4ExpPinnedHostEmbedding, _advise_random_file_access


@triton.jit
def _pinned_iq4nl_rows(
    pointer,
    ids,
    codebook,
    output,
    K: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    index = tl.load(ids + row).to(tl.int64)
    col = tl.arange(0, BLOCK)
    valid = col < K
    table = pointer.to(tl.int64).to(tl.pointer_type(tl.uint8))
    base = table + index * WIDTH + (col // 32) * 18
    lo = tl.load(base, valid, other=0).to(tl.uint16)
    hi = tl.load(base + 1, valid, other=0).to(tl.uint16)
    scale = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    packed = tl.load(base + 2 + col % 16, valid, other=0)
    code = (packed >> (4 * ((col % 32) // 16))) & 15
    value = tl.load(codebook + code.to(tl.int32)).to(tl.float32)
    tl.store(output + row * K + col, scale * value, valid)


def pinned_iq4nl_rows(pointer, ids, codebook, output, k):
    _pinned_iq4nl_rows[(ids.numel(),)](
        pointer,
        ids,
        codebook,
        output,
        K=k,
        WIDTH=k // 32 * 18,
        BLOCK=triton.next_power_of_2(k),
        num_warps=4,
    )


def packed_pinned_lookup(ids, output, layer_name):
    table = get_forward_context().no_compile_layers[layer_name]
    index = ids.device.index
    if index is None:
        index = torch.accelerator.current_device_index()
    pinned_iq4nl_rows(
        table._accelerator_weight_ptrs[index],
        ids,
        table._pinned_codebook,
        output,
        table.embedding_dim,
    )


def packed_pinned_lookup_fake(ids, output, layer_name):
    return


direct_register_custom_op(
    op_name="qwen4_exp_ple_pinned_iq4nl_lookup",
    op_func=packed_pinned_lookup,
    mutates_args=["output"],
    fake_impl=packed_pinned_lookup_fake,
)


@triton.jit
def _gather_packed_rows(pointer, ids, output, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    index = tl.load(ids + row)
    offsets = tl.arange(0, BLOCK)
    pointer = pointer.to(tl.int64).to(tl.pointer_type(tl.uint8))
    values = tl.load(pointer + index * WIDTH + offsets, offsets < WIDTH, other=0)
    tl.store(output + row * WIDTH + offsets, values, offsets < WIDTH)


def packed_ple_gather(
    ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    use_host: bool,
    width: int,
) -> None:
    if not ids.numel():
        return
    table = get_forward_context().no_compile_layers[layer_name]
    index = ids.device.index
    if index is None:
        index = torch.accelerator.current_device_index()
    pointer = (
        table._accelerator_weight_ptrs[index] if use_host else table._device_table_ptr
    )
    _gather_packed_rows[(ids.numel(),)](
        pointer,
        ids,
        output,
        WIDTH=width,
        BLOCK=triton.next_power_of_2(width),
        num_warps=4,
    )


def packed_ple_gather_fake(
    ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    use_host: bool,
    width: int,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen4_exp_ple_packed_gather",
    op_func=packed_ple_gather,
    mutates_args=["output"],
    fake_impl=packed_ple_gather_fake,
)


class Qwen4ExpPLEGGUFEmbeddingMethod(GGUFEmbeddingMethod):
    def embedding(self, layer, input_):
        return layer.embedding_lookup(input_)

    def process_weights_after_loading(self, layer):
        layer.prepare_accelerator_weight()


class Qwen4ExpPackedGGUFEmbedding(Qwen4ExpPinnedHostEmbedding):
    qweight: Parameter
    qweight_type: Parameter

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        params_dtype,
        padding_size,
        prefix,
        quant_method,
    ):
        with torch.device("meta"):
            VocabParallelEmbedding.__init__(
                self,
                num_embeddings,
                embedding_dim,
                params_dtype=params_dtype,
                padding_size=padding_size,
                prefix=prefix,
                quant_method=quant_method,
            )
        original = self.qweight
        placeholder = Parameter(
            torch.empty((0, 0), dtype=torch.uint8, device="cpu"), False
        )
        set_weight_attrs(placeholder, dict(vars(original)))
        placeholder._vllm_keep_on_cpu = True
        self.qweight = placeholder
        original_type = self.qweight_type
        self.qweight_type = Parameter(
            torch.zeros(1, dtype=torch.uint8, device="cpu"), False
        )
        set_weight_attrs(self.qweight_type, dict(vars(original_type)))
        self.qweight_type._vllm_keep_on_cpu = True
        self._meta_weight_shape = original.tensor_shape
        self._meta_weight_dtype = torch.uint8
        self._output_dtype = quant_method.params_dtype
        self._storage_dim = 0
        self._source_type = None
        self._cpu_reader = None
        self._cpu_owned = is_offload_process()
        self.layer_name = prefix
        context = get_current_vllm_config().compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self
        self._accelerator_weight_views = {}
        self._accelerator_weight_ptrs = {}
        self._device_rows = self._host_rows = self._disk_rows = 0
        self._device_table_ptr = 0
        self.ple_device_table = self.ple_host_storage = None
        self._checkpoint_shard_loaded = False

    def weight_loader(self, param, loaded_weight):
        if param is self.qweight_type:
            if loaded_weight.numel() != 1:
                raise ValueError("PLE GGUF table requires one source type")
            self._source_type = int(loaded_weight.item())
            self.qweight_type.weight_type = self._source_type
            self.qweight_type.data.fill_(self._source_type)
            return
        if self._source_type is None:
            raise ValueError("Missing PLE GGUF type before packed rows")
        block, size = quant_size(self._source_type)
        if self.embedding_dim % block:
            raise ValueError("PLE GGUF width cuts a source block")
        self._storage_dim = self.embedding_dim // block * size
        data = (
            loaded_weight.detach().view(torch.uint8).reshape(loaded_weight.shape[0], -1)
        )
        if data.device.type != "cpu" or data.shape[1] != self._storage_dim:
            raise ValueError("PLE GGUF requires complete packed CPU rows")
        if data.shape[0] < self.org_vocab_size:
            raise ValueError("PLE GGUF table is shorter than the logical vocabulary")
        _advise_random_file_access(data)
        if self._cpu_owned:
            # The parameter and reader retain the mmap owner, with no table copy.
            self.qweight = Parameter(data, False)
            self.qweight._vllm_keep_on_cpu = True
            self._cpu_reader = PackedGGUFRowReader(
                data.numpy(), self._source_type, self.embedding_dim, self.org_vocab_size
            )
        else:
            self.materialize_tables()
            assert self.ple_device_table is not None
            assert self.ple_host_storage is not None
            start = self.shard_indices.org_vocab_start_index
            local = data[start : start + self.num_embeddings_per_partition]
            self.ple_device_table.copy_(local[: self._device_rows])
            self.ple_host_storage.copy_(
                local[self._device_rows : self._device_rows + self._host_rows]
            )
        self._checkpoint_shard_loaded = True

    def materialize_tables(self):
        if self._cpu_owned or self.ple_device_table is not None:
            return
        if self._source_type is None:
            raise ValueError("PLE GGUF placement requires loaded source metadata")
        device = torch.device("cuda", torch.accelerator.current_device_index())
        placement = self._plan_placement(device)
        self.ple_device_table = torch.empty(
            (placement.vram_rows, self._storage_dim), dtype=torch.uint8, device=device
        )
        self.ple_host_storage = torch.empty(
            (placement.host_rows, self._storage_dim),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=placement.host_rows > 0,
        )
        self._device_rows, self._host_rows, self._disk_rows = (
            placement.vram_rows,
            placement.host_rows,
            placement.disk_rows,
        )
        self._device_table_ptr = self.ple_device_table.data_ptr()

    def prepare_accelerator_weight(self):
        if not self._checkpoint_shard_loaded:
            raise ValueError("PLE GGUF packed table was not loaded")
        if not self._cpu_owned:
            self.get_accelerator_weight(
                torch.device("cuda", torch.accelerator.current_device_index())
            )
            from vllm.model_executor.kernels.ple.gguf_pinned import pinned_decode_active

            if pinned_decode_active():
                import gguf

                if self._source_type != 20 or self._device_rows or self._disk_rows:
                    raise RuntimeError("Admitted PLE table is not wholly pinned IQ4_NL")
                self.register_buffer(
                    "_pinned_codebook",
                    torch.tensor(
                        gguf.quants.IQ4_NL.kvalues,
                        dtype=torch.float32,
                        device=torch.device(
                            "cuda", torch.accelerator.current_device_index()
                        ),
                    ),
                    persistent=False,
                )

    def embedding_lookup(self, input_, remote_rows=None):
        if input_.device.type == "cpu":
            if self._cpu_reader is None:
                raise ValueError("PLE GGUF CPU reader was not loaded")
            dtype = {torch.float16: "float16", torch.float32: "float32"}[
                self._output_dtype
            ]
            return torch.from_numpy(self._cpu_reader.lookup(input_.numpy(), dtype))
        ids = input_.reshape(-1)
        if hasattr(self, "_pinned_codebook") and remote_rows is None:
            output = torch.empty(
                (ids.numel(), self.embedding_dim),
                dtype=self._output_dtype,
                device=input_.device,
            )
            torch.ops.vllm.qwen4_exp_ple_pinned_iq4nl_lookup(
                ids, output, self.layer_name
            )
            return output.reshape(*input_.shape, self.embedding_dim)
        remote = ids >= self._device_rows + self._host_rows
        if remote_rows is None and self._disk_rows:
            raise ValueError("PLE GGUF disk rows require offloader output")
        if self._device_rows == 0 and self._host_rows == 0:
            if (
                remote_rows is None
                or remote_rows.dtype != self._output_dtype
                or remote_rows.numel() != ids.numel() * self.embedding_dim
            ):
                raise ValueError("PLE GGUF remote rows have invalid dtype or width")
            # There is no resident row zero to gather in a disk-only placement.
            # Retain independent output storage from the offloader's buffer.
            return remote_rows.reshape(*input_.shape, self.embedding_dim).clone()
        if remote_rows is not None:
            ids = torch.where(remote, 0, ids)
        packet = torch.empty(
            (ids.numel(), self._storage_dim), dtype=torch.uint8, device=input_.device
        )
        gather = torch.ops.vllm.qwen4_exp_ple_packed_gather
        if self._device_rows == 0 or self._host_rows == 0:
            gather(ids, packet, self.layer_name, self._host_rows > 0, self._storage_dim)
        else:
            on_host = ids >= self._device_rows
            host = torch.empty_like(packet)
            gather(
                torch.where(on_host, 0, ids),
                packet,
                self.layer_name,
                False,
                self._storage_dim,
            )
            gather(
                torch.where(on_host, ids - self._device_rows, 0),
                host,
                self.layer_name,
                True,
                self._storage_dim,
            )
            packet.copy_(torch.where(on_host[:, None], host, packet))
        if self._source_type in (0, 1, 30):
            dtype = {0: torch.float32, 1: torch.float16, 30: torch.bfloat16}[
                self._source_type
            ]
            output = packet.view(dtype).to(self._output_dtype)
        else:
            # Packet rows already follow the requested order. Decode them
            # directly instead of indexing them by an identity arange.
            output = dequantize_gguf_rows(
                packet,
                self._source_type,
                self.embedding_dim,
                dtype=self._output_dtype,
                native_enabled=cast(
                    Qwen4ExpPLEGGUFEmbeddingMethod, self.quant_method
                ).native_enabled,
            )
        if remote_rows is not None:
            if (
                remote_rows.dtype != output.dtype
                or remote_rows.numel() != output.numel()
            ):
                raise ValueError("PLE GGUF remote rows have invalid dtype or width")
            output = torch.where(
                remote[:, None], remote_rows.reshape_as(output), output
            )
        return output.reshape(*input_.shape, self.embedding_dim)
