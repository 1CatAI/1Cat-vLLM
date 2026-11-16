# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMV_X: dmv13_x keeps dmv13's per-warp math and reduction order.

Text-level contract (CPU only): the Q5_K / Q8_0 decode, plane loads, mma
order, K-slot and split-K sums and output mapping in gguf_dmv13_x.cuh must
match gguf_dmv13_core.cuh statement for statement, so ACC=1 outputs are
bit-identical at the same (warps, split). The GPU check is
benchmarks/deepseek_v41/v41k/bench_dmv_x.py (bitwise diff per shape).
"""

from pathlib import Path

import pytest
import regex as re

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / "csrc" / "sm70_turbomind" / "ops"
CORE = (OPS / "gguf_dmv13_core.cuh").read_text()
X = (OPS / "gguf_dmv13_x.cuh").read_text()
V41K = (ROOT / "benchmarks" / "deepseek_v41" / "v41k" / "v41k.cu").read_text()

_HELPERS = r"\bx_(mma|lop_or|h2|u32|lo2|hi2|word|write)\b"


def _norm(text):
    """Strip comments, x_ helper prefixes and layout differences."""
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(_HELPERS, r"\1", text)
    text = text.replace("CW<FMT>::v", "CW").replace("L.hi[0]", "L.hi")
    text = text.replace("acc[st % ACC]", "acc").replace("float(&a)[8] = acc;", "")
    text = re.sub(r"\bmma\(a,", "mma(acc,", text)
    text = text.replace("kXQ5K", "Q5K").replace("kXQ8", "Q8")
    return re.sub(r"\s+", "", text)


def _between(text, start, end):
    i = text.index(start)
    return text[i : text.index(end, i)]


CORE_N, X_N = _norm(CORE), _norm(X)

# Statements that carry the arithmetic; each must appear in both headers.
STATEMENTS = [
    # Q5_K decode (5th bit from the high plane, scale/min half2 FMA)
    "consthalf2s2=lo2(sc),m2=hi2(sc);",
    "constints=4*c+j-4;",
    "s>=0?(hb>>(s>=0?s:0)):(hb<<(s<0?-s:0));",
    "lop_or(sh,0x00100010u,MAGIC);",
    "constuint32_th=lop_or(cr[c]>>(4*j),0x000f000fu,extra);",
    "hw[4*c+j]=u32(__hfma2(__hsub2(h2(h),h2(MAGIC)),s2,m2));",
    # Q8_0 decode
    "consthalf2bias=h2(0x64806480u);",
    "constuint32_tv=word(L.c[st][c>>2],c&3)^0x80808080u;",
    'asm("prmt.b32%0,%1,%2,0x4140;":"=r"(a):"r"(v),"r"(0x64646464u));',
    'asm("prmt.b32%0,%1,%2,0x4342;":"=r"(b):"r"(v),"r"(0x64646464u));',
    "hw[2*c]=u32(__hmul2(__hsub2(h2(a),bias),s2));",
    "hw[2*c+1]=u32(__hmul2(__hsub2(h2(b),bias),s2));",
    # plane loads
    "constsize_ttg=static_cast<size_t>(t)*G+g;",
    "L.sc=__ldcs(s.scale+tg*32+lane);",
    "L.hi=__ldcs(s.high+tg*32+lane);",
    "constsize_tSp=static_cast<size_t>(G)*4;",
    "__ldcs(s.codes+((static_cast<size_t>(t)*Sp+sidx)*CW+j)*32+lane);",
    # x staging and the per-warp mma chain (k order within a step)
    "constintr=(lane&3)+((lane&16)?4:0);",
    "constuint4*xp=slot+cur*128+(st*4)*8+r;",
    "for(intj=0;j<4;++j)xa[j]=xp[j*8];",
    "mma(acc,xa[j].x,xa[j].y,hw[4*j],hw[4*j+1]);",
    "mma(acc,xa[j].z,xa[j].w,hw[4*j+2],hw[4*j+3]);",
    "constintgn=g+KW;",
    # split ranges, output mapping
    "constintgps=(G+split-1)/split;",
    "constintg0=min(G,sp*gps),g1=min(G,g0+gps);",
    "consttoken=(i&2)|((lv&16)?4:0)|(lv&1);",
    "constintcol=(i&1)|(((lv>>1)&1)<<1)|((i>>2)<<2);",
    "constintrow=t*32+((lv>>2)&3)*8+col;",
    "sg.out[token*sg.out_ld+row]=__float2half_rn(val);",
    # split-K epilogue: fixed 8-slot sum with zero fill, counter reset
    "last=atomicAdd(cnt+tg,1)==split-1;",
    "for(intp=0;p<8;++p)s+=pv[p];",
    "if(threadIdx.x==0)cnt[tg]=0;",
]


@pytest.mark.parametrize("stmt", STATEMENTS)
def test_statement_in_both(stmt):
    stmt = stmt.replace("consttoken", "constinttoken")
    assert stmt in CORE_N, f"core changed: {stmt}"
    assert stmt in X_N, f"dmv13_x diverged: {stmt}"


def test_q8_decode_branch_is_verbatim():
    marker = "} else {  // Q8: int8 codes"
    core = _norm(_between(CORE, marker, "__device__ __constant__ int8_t kIQ4"))
    x = _norm(_between(X, marker, "template <int FMT>\n__device__ __forceinline__"))
    assert core == x


def test_reduction_order_matches():
    # K-slot sum in warp order, then split slots 0..7 in order.
    assert "for(intk=0;k<KW;++k)s+=red[(k*TN+i)*256+e];" in CORE_N
    assert "for(intk=0;k<KW;++k)s+=red[k*256+v];" in X_N
    pv = "pv[p]=p<split?__ldcg(ws+(static_cast<size_t>(tg)*split+p)*{}256+v):0.f;"
    assert pv.format("TN*") in CORE_N
    assert pv.format("") in X_N
    assert "__stcg(ws+(static_cast<size_t>(tg)*split+sp)*256+v,s);" in X_N
    # ACC=2 is the only order change, and it is a compile-time option.
    assert "if constexpr (ACC == 2) a += acc[1][e];" in X


def test_core_fmt_codes():
    assert re.search(
        r"Q4K\s*=\s*0,\s*Q5K\s*=\s*1,\s*Q6K\s*=\s*2,\s*LUT4\s*=\s*3,\s*Q8\s*=\s*4", CORE
    )
    assert "constexpr int kXQ5K = 1;" in X and "constexpr int kXQ8 = 4;" in X


def test_schema_matches_binding():
    block = _between(V41K, '"dmv13_x(', "&v41k_dmv13_x")
    joined = "".join(re.findall(r'"([^"]*)"', block))  # adjacent literals
    schema = re.fullmatch(r"dmv13_x\((.*)\) -> \(\)", joined).group(1)
    names = [arg.split()[-1] for arg in schema.split(", ")]
    assert names == [
        "x", "codes", "high", "scale", "out", "x_off", "fmt", "K", "split",
        "warps", "acc", "ws", "cnt", "probe",
    ]  # fmt: skip
    binding = _between(V41K, "void v41k_dmv13_x(", "{")
    params = [p.split()[-1] for p in re.sub(r"\s+", " ", binding).split("(")[1]
              .rstrip(") ").split(", ")]  # fmt: skip
    assert params == names
    assert 'm.impl("dmv13_x"' in V41K or "&v41k_dmv13_x" in V41K


def test_dispatch_covers_warps_formats_acc():
    for kw in (2, 4, 8):
        assert f"DMVX_W({kw});" in X
    for args in ("kXQ5K, 1, false", "kXQ5K, 2, false", "kXQ8, 1, false",
                 "kXQ8, 2, false", "kXQ5K, 1, true", "kXQ8, 1, true"):  # fmt: skip
        assert f"DMVX_GO(KWV, {args});" in X
