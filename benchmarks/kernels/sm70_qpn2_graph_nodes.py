# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private executable-node substitutions. Raw graph and its edges stay frozen."""

import ctypes as C
from dataclasses import dataclass

rt = C.CDLL("libcuda.so.1")


class Dim3(C.Structure):
    _fields_ = [("x", C.c_uint), ("y", C.c_uint), ("z", C.c_uint)]


class Params(C.Structure):
    _fields_ = [
        ("func", C.c_void_p),
        ("gridDim", Dim3),
        ("blockDim", Dim3),
        ("sharedMemBytes", C.c_uint),
        ("kernelParams", C.POINTER(C.c_void_p)),
        ("extra", C.POINTER(C.c_void_p)),
        ("kern", C.c_void_p),
        ("ctx", C.c_void_p),
    ]


class EdgeData(C.Structure):
    _fields_ = [
        ("from_port", C.c_ubyte),
        ("to_port", C.c_ubyte),
        ("type", C.c_ubyte),
        ("reserved", C.c_ubyte * 5),
    ]


for name, types in {
    "cuGraphGetNodes": [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)],
    "cuGraphGetEdges_v2": [
        C.c_void_p,
        C.POINTER(C.c_void_p),
        C.POINTER(C.c_void_p),
        C.POINTER(EdgeData),
        C.POINTER(C.c_size_t),
    ],
    "cuGraphNodeGetType": [C.c_void_p, C.POINTER(C.c_int)],
    "cuGraphKernelNodeGetParams_v2": [C.c_void_p, C.POINTER(Params)],
    "cuGraphExecKernelNodeSetParams_v2": [C.c_void_p, C.c_void_p, C.POINTER(Params)],
    "cuFuncGetName": [C.POINTER(C.c_char_p), C.c_void_p],
    "cuFuncGetParamInfo": [
        C.c_void_p,
        C.c_size_t,
        C.POINTER(C.c_size_t),
        C.POINTER(C.c_size_t),
    ],
}.items():
    getattr(rt, name).argtypes = types
    getattr(rt, name).restype = C.c_int
rt.cuGetErrorString.argtypes = [C.c_int, C.POINTER(C.c_char_p)]
rt.cuGetErrorString.restype = C.c_int


def check(status):
    if status:
        message = C.c_char_p()
        rt.cuGetErrorString(status, C.byref(message))
        raise RuntimeError((status, message.value))


def param_sizes(func, count):
    assert 0 < count <= 16
    sizes = []
    for index in range(count):
        offset, size = C.c_size_t(), C.c_size_t()
        check(rt.cuFuncGetParamInfo(func, index, C.byref(offset), C.byref(size)))
        assert 0 < size.value <= 16, (index, size.value)
        sizes.append(size.value)
    return sizes


def known_count(name, params):
    # Frozen public signatures, not error-driven queries past the last argument.
    if "nvfp4_qpn2_sm70_kernel" in name or "nvfp4_qpn2_gated_sm70_kernel" in name:
        return 8
    if name == "_sm70_dflash2_gemma_fused_add_rms_kernel":
        return 8  # five data pointers, epsilon, two Triton scratch pointers
    if name == "_sm70_dflash2_fixed_gemma_rms_kernel":
        # Both valid signatures have at least five pointers. The no-residual
        # specialization has three data pointers then two null scratch pointers;
        # the residual specialization has five nonnull data pointers then scratch.
        assert param_sizes(params.func, 5) == [8] * 5
        values = [
            int.from_bytes(C.string_at(params.kernelParams[i], 8), "little")
            for i in range(5)
        ]
        assert all(values[:3])
        if values[3:] == [0, 0]:
            return 5
        assert values[3] and values[4], "Unsupported fixed-norm signature"
        return 7
    return None


def clone_params(params, arguments, *, template=None):
    value = Params.from_buffer_copy(params)
    if template is not None:
        value.func, value.kern, value.ctx = template.func, template.kern, template.ctx
        value.blockDim = template.blockDim
        value.sharedMemBytes = template.sharedMemBytes
    expected = param_sizes(value.func, len(arguments))
    assert list(map(len, arguments)) == expected, (list(map(len, arguments)), expected)
    buffers = [C.create_string_buffer(arg, len(arg)) for arg in arguments]
    pointers = (C.c_void_p * len(buffers))(*(C.addressof(b) for b in buffers))
    value.kernelParams, value.extra = pointers, None
    return value, (buffers, pointers)


@dataclass
class Node:
    handle: int
    name: str
    params: Params
    args: list[bytes]
    storage: object

    def integer(self, index):
        return int.from_bytes(self.args[index], "little")

    @property
    def grid(self):
        return (self.params.gridDim.x, self.params.gridDim.y, self.params.gridDim.z)


def nodes(graph, *, single_parameter_count=None):
    ptr = graph.raw_cuda_graph()
    count = C.c_size_t()
    check(rt.cuGraphGetNodes(ptr, None, C.byref(count)))
    handles = (C.c_void_p * count.value)()
    check(rt.cuGraphGetNodes(ptr, handles, C.byref(count)))
    result = {}
    for handle in handles:
        kind = C.c_int()
        check(rt.cuGraphNodeGetType(handle, C.byref(kind)))
        if kind.value != 0:
            continue
        params = Params()
        check(rt.cuGraphKernelNodeGetParams_v2(handle, C.byref(params)))
        name = C.c_char_p()
        check(rt.cuFuncGetName(C.byref(name), params.func))
        text_name = name.value.decode()
        recognized = (
            "nvfp4_qpn2_sm70_kernel" in text_name
            or "nvfp4_qpn2_gated_sm70_kernel" in text_name
            or text_name
            in (
                "_sm70_dflash2_fixed_gemma_rms_kernel",
                "_sm70_dflash2_gemma_fused_add_rms_kernel",
            )
        )
        if single_parameter_count is None and not recognized:
            continue
        # cuBLAS may use the packed launch-parameter buffer convention. It is
        # outside this rewrite and must be skipped before assuming pointer args.
        assert params.kernelParams and not params.extra
        count_params = (
            single_parameter_count
            if single_parameter_count is not None
            else known_count(name.value.decode(), params)
        )
        if count_params is None:
            continue
        sizes = param_sizes(params.func, count_params)
        arguments = [
            C.string_at(params.kernelParams[i], n) for i, n in enumerate(sizes)
        ]
        owned, storage = clone_params(params, arguments)
        result[handle] = Node(handle, name.value.decode(), owned, arguments, storage)
    edge_count = C.c_size_t()
    check(rt.cuGraphGetEdges_v2(ptr, None, None, None, C.byref(edge_count)))
    if edge_count.value == 0:
        return result, {handle: [] for handle in handles}
    sources = (C.c_void_p * edge_count.value)()
    targets = (C.c_void_p * edge_count.value)()
    data = (EdgeData * edge_count.value)()
    check(rt.cuGraphGetEdges_v2(ptr, sources, targets, data, C.byref(edge_count)))
    assert all((d.from_port, d.to_port, d.type) == (0, 0, 0) for d in data), (
        "Unsupported graph dependency type"
    )
    parents = {handle: [] for handle in handles}
    for a, b in zip(sources, targets):
        parents[b].append(a)
    return result, parents


def set_params(graph, node, params):
    check(
        rt.cuGraphExecKernelNodeSetParams_v2(
            graph.raw_cuda_graph_exec(), node, C.byref(params)
        )
    )


def only_kernel(graph, parameter_count):
    kernels, _ = nodes(graph, single_parameter_count=parameter_count)
    assert len(kernels) == 1, [n.name for n in kernels.values()]
    return next(iter(kernels.values()))
