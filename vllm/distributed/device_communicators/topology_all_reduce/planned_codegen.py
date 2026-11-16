# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Keep generated CUDA strings byte-identical to the qualified implementation.
# ruff: noqa: E501

"""Generate fixed-DAG CUDA peer kernels without fixed topology or message size."""

from .collective_planner import validate


def layout(plan, capacity):
    offsets = []
    sizes = [0] * len(plan["devices"])
    for message in plan["messages"]:
        dest = message["dest"]
        offsets.append(sizes[dest])
        sizes[dest] += (
            2
            * capacity
            * (2 if plan["values"][message["value"]]["dtype"] == "fp32" else 1)
        )
    return offsets, [max(16, n) for n in sizes]


HEADER = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
struct Wide{float data[8];};
__device__ __forceinline__ void store_packet(uint4 value,char* base,int i){
 if(value.x==0xffffffff)value.x=0x7fc07fc0;if(value.y==0xffffffff)value.y=0x7fc07fc0;
 if(value.z==0xffffffff)value.z=0x7fc07fc0;if(value.w==0xffffffff)value.w=0x7fc07fc0;
 auto ptr=(uint4*)base+i;
 asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"::"l"(ptr),"r"(value.x),"r"(value.y),"r"(value.z),"r"(value.w):"memory");
}
__device__ __forceinline__ uint4 receive_packet(char* base,int i){
 uint4 value;auto ptr=(uint4*)base+i;
 do{asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(value.x),"=r"(value.y),"=r"(value.z),"=r"(value.w):"l"(ptr):"memory");}
 while(value.x==0xffffffff||value.y==0xffffffff||value.z==0xffffffff||value.w==0xffffffff);
 asm volatile("st.volatile.global.v4.u32 [%0], {%1,%1,%1,%1};"::"l"(ptr),"r"(0xffffffff):"memory");return value;
}
__device__ __forceinline__ uint4 add_half(uint4 a,uint4 b){
 uint4 out;
 #pragma unroll
 for(int j=0;j<4;++j)asm volatile("add.rn.f16x2 %0,%1,%2;":"=r"(((uint32_t*)&out)[j]):"r"(((uint32_t*)&a)[j]),"r"(((uint32_t*)&b)[j]));
 return out;
}
__device__ __forceinline__ Wide widen(uint4 a){
 Wide out;
 #pragma unroll
 for(int j=0;j<4;++j){float2 f=__half22float2(((half2*)&a)[j]);out.data[2*j]=f.x;out.data[2*j+1]=f.y;}return out;
}
__device__ __forceinline__ Wide widen(Wide a){return a;}
__device__ __forceinline__ Wide add_wide(Wide a,Wide b){
 Wide out;
 #pragma unroll
 for(int j=0;j<8;++j)out.data[j]=__fadd_rn(a.data[j],b.data[j]);return out;
}
__device__ __forceinline__ uint4 narrow(Wide a){
 uint4 out;
 #pragma unroll
 for(int j=0;j<4;++j)((half2*)&out)[j]=__floats2half2_rn(a.data[2*j],a.data[2*j+1]);return out;
}
__device__ __forceinline__ uint4 narrow(uint4 a){return a;}
"""


def generate(plans, topology, capacity):
    if not plans or len(plans) > 32 or capacity <= 0 or capacity % 16:
        raise ValueError("Invalid finite kernel batch")
    n = len(plans[0]["devices"])
    text = [HEADER, f"struct Boxes{{char* data[{n}];}};"]
    dispatch = []
    if any(len(p["devices"]) != n for p in plans):
        raise ValueError("One node count per compiled batch")
    for c, p in enumerate(plans):
        validate(p, topology)
        offsets, _ = layout(p, capacity)
        for rank, program in enumerate(p["programs"]):
            name = f"planned_{c}_rank_{rank}"
            leaf = "x[i]" if p["values"][rank]["mask"] else "make_uint4(0,0,0,0)"
            text.append(
                f"__global__ void {name}(const uint4* x,uint4* out,Boxes boxes,int* flags,int count){{\nint b=blockIdx.x,epoch=flags[b]&1;\nfor(int i=b*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x){{uint4 v{rank}={leaf};"
            )
            for op in program:
                kind = op["op"]
                v = op["value"]
                wide = p["values"][v]["dtype"] == "fp32"
                if kind == "add":
                    expression = (
                        f"add_wide(widen(v{op['left']}),widen(v{op['right']}))"
                        if wide
                        else f"add_half(v{op['left']},v{op['right']})"
                    )
                    text.append(f"{'Wide' if wide else 'uint4'} v{v}={expression};")
                else:
                    mid = op["message"]
                    m = p["messages"][mid]
                    width = 2 if wide else 1
                    address = f"boxes.data[{m['dest']}]+{offsets[mid]}+epoch*{capacity * width}"
                    if wide:
                        if kind == "recv":
                            text.append(f"Wide v{v};")
                        for j in range(2):
                            if kind == "send":
                                # Canonicalize only a NaN word that collides with
                                # readiness; finite numbers and infinities keep
                                # their exact bits. Half transport has its own
                                # safe two-half NaN representation in store_packet.
                                text.append(
                                    f"{{uint4 packet=((uint4*)&v{v})[{j}];uint32_t* words=(uint32_t*)&packet;for(int k=0;k<4;++k)if(words[k]==0xffffffff)words[k]=0x7fc00000;store_packet(packet,{address},2*i+{j});}}"
                                )
                            else:
                                text.append(
                                    f"((uint4*)&v{v})[{j}]=receive_packet({address},2*i+{j});"
                                )
                    elif kind == "send":
                        text.append(f"store_packet(v{v},{address},i);")
                    else:
                        text.append(f"uint4 v{v}=receive_packet({address},i);")
            if p["devices"][rank] in p["members"]:
                text.append(f"out[i]=narrow(v{p['outputs'][rank]});")
            text.append(
                "}\n__syncthreads();if(threadIdx.x==0)((uint32_t*)flags)[b]+=1u;\n}"
            )
            dispatch.append(
                f"if(candidate=={c}&&rank=={rank}){{{name}<<<blocks,threads,0,stream>>>(x,out,p,flags,count);return;}}"
            )
    text.append(f"""void planned_raw(const uint4* x,uint4* out,const std::vector<int64_t>& pointers,int* flags,int candidate,int rank,int threads,int blocks,int count,cudaStream_t stream){{
 Boxes p;for(int i=0;i<{n};++i)p.data[i]=(char*)pointers[i];
""")
    text.extend(dispatch)
    text.append('throw std::runtime_error("Unknown planned kernel");\n}')
    # stdexcept is needed by the host dispatch only.
    return "#include <stdexcept>\n" + "\n".join(text) + "\n"
