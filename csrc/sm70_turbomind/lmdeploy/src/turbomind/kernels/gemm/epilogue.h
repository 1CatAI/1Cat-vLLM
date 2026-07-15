// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/core/sync.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/predicate.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include "../../../../../../sm70_tile_runtime_signal.cuh"

namespace turbomind::gemm {

template<class Tc>
struct ChannelCombination_v3 {
    const Tc* __restrict__ scale_bias_ptr;

    template<class T, int V, int S, int C, int delta_c, int delta_s, class Pred>
    __device__ void operator()(Array<T, V> (&x)[S][C], int2 cs0, pair<delta_c, delta_s>, Pred& pred) const
    {
        __align__(16) Array<Tc, 2> scale_bias[S];

        if (scale_bias_ptr) {
            constexpr int ds  = sizeof(Tc) * delta_s;
            auto          ptr = reinterpret_cast<const char*>(scale_bias_ptr + cs0.y);
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                if (pred(s, 0)) {
                    Ldg(scale_bias[s], reinterpret_cast<const Tc*>(ptr));
                }
                ptr += ds;
            }
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                auto tmp = cast<T>(scale_bias[s]);
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    using namespace ops;
                    x[s][c] = x[s][c] * tmp[0] + tmp[1];
                }
            }
        }
    }
};

template<bool     scale_S,
         bool     scale_C,
         Striding mode_S,
         Striding mode_C,
         class T,
         int N,
         int S,
         int C,
         int delta_C,
         int delta_S,
         class Pred>
__device__ void Scale(pair<scale_S, scale_C>,
                      pair<mode_S, mode_C>,
                      pair<delta_C, delta_S>,
                      Array<T, N> (&x)[S][C],
                      const MatrixParam& param_S,
                      const MatrixParam& param_C,
                      int                gemm_id,
                      int2               cs0,
                      Pred&              pred)
{
    if (scale_S && param_S.ptr) {
        const auto mat = resolve<T, mode_S>(param_S, gemm_id);
        const T*   ptr = (const T*)mat.ptr.ptr;
        T          param[S];
        PRAGMA_UNROLL
        for (int s = 0; s < S; ++s) {
            const int ss  = cs0.y + s * delta_S;
            const int idx = mat.idxs ? __ldg(mat.idxs + ss) : ss;
            if (pred(s, 0)) {
                param[s] = __ldg((const T*)(ptr + idx));
            }
            PRAGMA_UNROLL
            for (int c = 0; c < C; ++c) {
                using namespace ops;
                x[s][c] = x[s][c] * param[s];
            }
        }
    }

    if (scale_C && param_C.ptr) {
        const T*      ptr = (const T*)resolve<T, mode_C>(param_C, gemm_id).ptr.ptr + cs0.x;
        constexpr int dc  = sizeof(Array<T, N>) * delta_C;
        Array<T, N>   param[C];
        PRAGMA_UNROLL
        for (int c = 0; c < C; ++c) {
            if (pred(0, c)) {
                Ldg(param[c], (const T*)(ptr + dc * c));
            }
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                using namespace ops;
                x[s][c] = x[s][c] * param[c];
            }
        }
    }
}

struct MatrixCombination_v3 {

    MatrixParam param_c;
    float       alpha;
    float       beta;

    template<class Tc, Striding mode, class T, int N, int S, int C, int delta_c, int delta_s, class Pred>
    __device__ void operator()(Tc*,  //
                               constant<mode>,
                               Array<T, N> (&x)[S][C],
                               int2 cs0,
                               int  gemm_id,
                               pair<delta_c, delta_s>,
                               Pred& pred) const
    {
        if (beta) {
            const auto c = resolve<Tc, mode>(param_c, gemm_id);

            Array<Tc, N>  frag[S][C];
            constexpr int dc  = sizeof(Tc) * delta_c;
            const int     ds  = sizeof(Tc) * delta_s * c.ptr.stride;
            const char*   ptr = (const char*)c.ptr.ptr + sizeof(Tc) * dot(cs0, long2{1, c.ptr.stride});
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    if (pred(s, c)) {
                        Load(frag[s][c], reinterpret_cast<const Tc*>(ptr));
                        using namespace ops;
                        x[s][c] = x[s][c] * alpha + cast<T>(frag[s][c]) * beta;
                    }
                    ptr += dc;
                }
                ptr -= dc * C;
                ptr += ds;
            }
        }
        else if (alpha != 1.f) {
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    using namespace ops;
                    x[s][c] = x[s][c] * alpha;
                }
            }
        }
    }
};

template<class Act>
struct GatedActivation {
    template<class OutT, class T, int N>
    __device__ static void apply(Array<T, N>& x)
    {
        static_assert(N % 2 == 0);
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 2) {
            const OutT gate = static_cast<OutT>(x[i]);
            const OutT up   = static_cast<OutT>(x[i + 1]);
            x[i / 2]        = static_cast<T>(Act::apply(gate) * up);
        }
    }
};

struct Silu {
    template<class T>
    __device__ static T apply(T x)
    {
        const float xf = static_cast<float>(x);
        return static_cast<T>(fdividef(xf, 1.f + expf(-xf)));
    }
};

struct EpilogueParam {
    MatrixParam c;
    MatrixParam partials;
    int*        locks;

    // MatrixParam scale_S;
    // MatrixParam scale_C;

    MatrixCombination_v3 combine_mat;

    bool         silu_act;
    bool         moe_weighted_reduce;
    void*        moe_reduce_out;
    const float* moe_sorted_weights;
    const int*   moe_offsets;

    bool tile_allreduce;
    TileAllReduceParam tile_allreduce_param;
};

template<class Tc_,
         int M,
         int N,
         int TM_,
         int TN_,
         int THREADS,
         class RearrangeC,
         class OperandC,
         Striding mode_C,
         bool     SplitK_>
struct Epilogue_ {

    using Dtype = typename OperandC::Dtype;

    static constexpr auto kOrder = OperandC::kOrder;
    static constexpr auto kMode  = mode_C;
    static constexpr bool SplitK = SplitK_;
    using Tc = Tc_;

    static constexpr int TM = TM_;
    static constexpr int TN = TN_;

    using SmemLayout = decltype(OperandC::GetSmemLayout::apply(pair<TM, TN>{}));

    using SmemAccessorV2 = SmemAccessorV2<Dtype, SmemLayout, kOrder>;

    using SharedStorage = Array<Dtype, SmemLayout::kSize>;

    using Map = decltype(OperandC::GetThreadMap::apply(pair<M, N>{}, constant<THREADS>{}));

    static constexpr int S       = Map::kIterS;
    static constexpr int C       = Map::kIterC;
    static constexpr int kAccess = Map::kAccessC;

    template<class T>
    using OutputC = Array<T, kAccess>;

    template<class FragC>
    __device__ void Rearrange(FragC& frag_C, SharedStorage& storage, OutputC<Dtype> (&out)[S][C])
    {
        SmemAccessorV2 smem_C{storage.data()};

        const int2 thr_cs = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE);

        constexpr int kPeriodC = ceil_div(SmemLayout::C0, Map::kDeltaC);
        constexpr int kPeriodS = ceil_div(SmemLayout::S0, Map::kDeltaS);

        int phases[kPeriodS][kPeriodC];
        PRAGMA_UNROLL
        for (int s = 0; s < kPeriodS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < kPeriodC; ++c) {
                phases[s][c] = SmemLayout::apply(s * Map::kDeltaS + thr_cs.y, c * Map::kDeltaC + thr_cs.x);
            }
        }

        constexpr bool kRaked = true;

        PRAGMA_UNROLL
        for (int m = 0; m < M; m += TM) {
            PRAGMA_UNROLL
            for (int n = 0; n < N; n += TN) {
                // Store to shared memory
                RearrangeC::apply(frag_C, smem_C, {m, n}, pair<TM, TN>{});

                // Load from shared memory
                PRAGMA_UNROLL
                for (int s = 0; s < S; ++s) {
                    PRAGMA_UNROLL
                    for (int c = 0; c < C; ++c) {
                        const int cc = c * Map::kDeltaC + thr_cs.x;
                        const int ss = s * Map::kDeltaS + thr_cs.y;

                        const int2 mn =
                            kRaked ? cs2mk<kOrder>(c * Map::kDeltaC, s * Map::kDeltaS) : cs2mk<kOrder>(cc, ss);
                        const int  mm   = mn.x - m;
                        const int  nn   = mn.y - n;
                        const bool mask = (M <= TM || (0 <= mm && mm < TM)) && ((N <= TN) || (0 <= nn && nn < TN));

                        const int2 _cs      = mk2cs<kOrder>(m, n);
                        const int  offset_0 = SmemLayout::apply(  //
                            s / kPeriodS * kPeriodS * Map::kDeltaS - _cs.y,
                            c / kPeriodC * kPeriodC * Map::kDeltaC - _cs.x);
                        const int  offset_p = phases[s % kPeriodS][c % kPeriodC];

                        if (mask) {
                            Load(out[s][c], &storage[offset_0 + offset_p]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    template<class T, class VecC, class Pred>
    __device__ void StoreC(const VecC& vec_C, const MatrixData& c, int2 cs0, Pred& pred)
    {
        constexpr int dc  = sizeof(T) * Map::kDeltaC;
        const int     ds  = sizeof(T) * Map::kDeltaS * c.ptr.stride;
        char*         ptr = (char*)c.ptr.ptr + sizeof(T) * dot(cs0, long2{1, c.ptr.stride});
        PRAGMA_UNROLL
        for (int s = 0; s < S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < C; ++c) {
                const auto tmp = cast<T>(vec_C[s][c]);
                if (pred(s, c)) {
                    Store(reinterpret_cast<T*>(ptr), tmp);
                }
                ptr += dc;
            }
            ptr -= dc * C;
            ptr += ds;
        }
    }

    template<class VecC, class Pred>
    __device__ void StoreMoeWeightedReduce(const VecC& vec_C,
                                           int2 cs0,
                                           int group_id,
                                           Pred& pred,
                                           const EpilogueParam& param)
    {
        static_assert(std::is_same_v<Tc, half_t>);
        __half* out = reinterpret_cast<__half*>(param.moe_reduce_out);
        const int base_row = param.moe_offsets ? __ldg(param.moe_offsets + group_id) : 0;
        PRAGMA_UNROLL
        for (int s = 0; s < S; ++s) {
            const int   row    = base_row + cs0.y + s * Map::kDeltaS;
            const float weight = __ldg(param.moe_sorted_weights + row);
            PRAGMA_UNROLL
            for (int c = 0; c < C; ++c) {
                if (pred(s, c)) {
                    const auto tmp = cast<float>(vec_C[s][c]);
                    PRAGMA_UNROLL
                    for (int i = 0; i < kAccess; ++i) {
                        const int col = cs0.x + c * Map::kDeltaC + i;
                        atomicAdd(out + col, __float2half_rn(tmp[i] * weight));
                    }
                }
            }
        }
    }

    __device__ void PublishTileAllReduceFlag(const int4&          tile_offset,
                                             int                  tile_id,
                                             bool                 is_last,
                                             const EpilogueParam& param)
    {
        if (!param.tile_allreduce || !is_last) {
            return;
        }
        const auto& tile_ar = param.tile_allreduce_param;
        if (tile_ar.world_size != 2 || tile_ar.tile_numel <= 0) {
            return;
        }
        // The first model path is intentionally M=1 and row-major N tiles.
        if (tile_offset.x != 0 || tile_offset.w != 0) {
            return;
        }
        const int col_begin = tile_offset.y * N;
        if (col_begin >= tile_ar.output_numel) {
            return;
        }
        const int col_end = min(col_begin + N, tile_ar.output_numel);

        // Make all CTA stores visible before any peer-visible flag is set.
        __threadfence_system();
        __syncthreads();

        const int first_signal = col_begin / tile_ar.tile_numel;
        const int last_signal  = (col_end - 1) / tile_ar.tile_numel;
        for (int signal_id = first_signal; signal_id <= last_signal; ++signal_id) {
            if (signal_id < 0 || signal_id >= vllm::sm70_tile_runtime::kMaxBlocks) {
                continue;
            }
            const int signal_begin = signal_id * tile_ar.tile_numel;
            const int signal_end   = min(signal_begin + tile_ar.tile_numel,
                                         tile_ar.output_numel);
            if (signal_begin < col_begin || signal_end > col_end) {
                continue;
            }

            const auto flag =
                reinterpret_cast<vllm::sm70_tile_runtime::Signal*>(
                    tile_ar.self_signal)->_flag[signal_id] +
                1;
            if (threadIdx.x < tile_ar.world_size) {
                auto* peer_signal =
                    reinterpret_cast<vllm::sm70_tile_runtime::Signal*>(
                        tile_ar.signals[threadIdx.x]);
                auto* peer_flag = &peer_signal->start[signal_id][tile_ar.rank];
                vllm::sm70_tile_runtime::store_flag_sys_visible(peer_flag, flag);
            }
        }
    }

    template<class VecC, class Pred>
    __device__ void StoreTileAllReduceC(const VecC&           vec_C,
                                        const MatrixData&     c,
                                        int2                  cs0,
                                        const int4&           tile_offset,
                                        bool                  is_last,
                                        Pred&                 pred,
                                        const EpilogueParam&  param)
    {
        if (!param.tile_allreduce || !is_last) {
            return;
        }
        const auto& tile_ar = param.tile_allreduce_param;
        if (tile_ar.world_size != 2 || tile_ar.tile_numel <= 0 ||
            tile_ar.rank_data == nullptr || tile_ar.output == nullptr) {
            return;
        }
        if (tile_offset.x != 0 || tile_offset.w != 0) {
            return;
        }

        const int col_begin = tile_offset.y * N;
        if (col_begin >= tile_ar.output_numel) {
            return;
        }
        const int col_end = min(col_begin + N, tile_ar.output_numel);
        const int first_signal = col_begin / tile_ar.tile_numel;
        const int last_signal  = (col_end - 1) / tile_ar.tile_numel;
        auto* self_signal =
            reinterpret_cast<vllm::sm70_tile_runtime::Signal*>(
                tile_ar.self_signal);

        for (int signal_id = first_signal; signal_id <= last_signal; ++signal_id) {
            if (signal_id < 0 ||
                signal_id >= vllm::sm70_tile_runtime::kMaxBlocks) {
                return;
            }
            const int signal_begin = signal_id * tile_ar.tile_numel;
            const int signal_end   = min(signal_begin + tile_ar.tile_numel,
                                         tile_ar.output_numel);
            if (signal_begin < col_begin || signal_end > col_end) {
                return;
            }

            const auto flag = self_signal->_flag[signal_id] + 1;
            if (threadIdx.x < tile_ar.world_size) {
                auto* self_flag = &self_signal->start[signal_id][threadIdx.x];
                while (vllm::sm70_tile_runtime::load_flag_sys_visible(
                           self_flag) != flag) {
                }
            }
            __syncthreads();
        }

        const auto rank_data =
            *reinterpret_cast<const vllm::sm70_tile_runtime::RankData*>(
                tile_ar.rank_data);
        const int peer_rank = 1 - tile_ar.rank;
        const auto* peer_base =
            reinterpret_cast<const char*>(rank_data.ptrs[peer_rank]);
        auto* output_base = reinterpret_cast<char*>(tile_ar.output);

        constexpr int dc = sizeof(Tc) * Map::kDeltaC;
        const int ds = sizeof(Tc) * Map::kDeltaS * c.ptr.stride;
        const auto offset =
            sizeof(Tc) * dot(cs0, long2{1, c.ptr.stride});
        const char* peer_ptr = peer_base + offset;
        char* output_ptr = output_base + offset;

        PRAGMA_UNROLL
        for (int s = 0; s < S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < C; ++c) {
                if (pred(s, c)) {
                    OutputC<Tc> peer;
                    Load(peer, reinterpret_cast<const Tc*>(peer_ptr));
                    using namespace ops;
                    const auto self = cast<Tc>(vec_C[s][c]);
                    const auto reduced =
                        cast<Tc>(cast<float>(self) + cast<float>(peer));
                    Store(reinterpret_cast<Tc*>(output_ptr), reduced);
                }
                peer_ptr += dc;
                output_ptr += dc;
            }
            peer_ptr -= dc * C;
            output_ptr -= dc * C;
            peer_ptr += ds;
            output_ptr += ds;
        }

        __syncthreads();
        if (threadIdx.x == 0) {
            for (int signal_id = first_signal; signal_id <= last_signal; ++signal_id) {
                self_signal->_flag[signal_id] = self_signal->_flag[signal_id] + 1;
            }
        }
    }

    __device__ bool StoreTailAllReduceC(const int4&          tile_offset,
                                        bool                 is_last,
                                        const EpilogueParam& param)
    {
        if (!param.tile_allreduce || !is_last) {
            return false;
        }
        const auto& tile_ar = param.tile_allreduce_param;
        if (tile_ar.world_size != 2 || tile_ar.rank_data == nullptr ||
            tile_ar.output == nullptr || tile_ar.output_numel <= 0 ||
            (tile_ar.output_numel & 1)) {
            return false;
        }
        // Narrow first model path: M=1, row-major dense down_proj output.
        if (tile_offset.x != 0 || tile_offset.w != 0) {
            return false;
        }
        const int cta_count = (tile_ar.output_numel + N - 1) / N;
        if (cta_count <= 0) {
            return false;
        }

        constexpr int signal_id = vllm::sm70_tile_runtime::kMaxBlocks - 1;
        auto* self_signal =
            reinterpret_cast<vllm::sm70_tile_runtime::Signal*>(
                tile_ar.self_signal);
        const auto epoch = self_signal->_flag[signal_id];
        const auto target = (epoch + 1) * cta_count;

        // Every producer CTA releases its staging stores before contributing to
        // the completion counter. Only one completed CTA stays alive as the
        // reduce worker; the rest exit the GEMM kernel normally.
        __syncthreads();
        __threadfence_system();
        if (threadIdx.x == 0) {
            atomicAdd(&self_signal->start[signal_id][tile_ar.rank],
                      static_cast<vllm::sm70_tile_runtime::FlagType>(1));
        }

        if (tile_offset.y != 0) {
            return true;
        }

        if (threadIdx.x < tile_ar.world_size) {
            auto* rank_signal =
                reinterpret_cast<vllm::sm70_tile_runtime::Signal*>(
                    tile_ar.signals[threadIdx.x]);
            auto* counter = &rank_signal->start[signal_id][threadIdx.x];
            while (vllm::sm70_tile_runtime::load_flag_sys_visible(counter) <
                   target) {
            }
        }
        __syncthreads();

        const auto rank_data =
            *reinterpret_cast<const vllm::sm70_tile_runtime::RankData*>(
                tile_ar.rank_data);
        const auto* rank0 =
            reinterpret_cast<const half2*>(rank_data.ptrs[0]);
        const auto* rank1 =
            reinterpret_cast<const half2*>(rank_data.ptrs[1]);
        auto* output = reinterpret_cast<half2*>(tile_ar.output);
        const int half2_count = tile_ar.output_numel / 2;
        for (int idx = threadIdx.x; idx < half2_count; idx += blockDim.x) {
            output[idx] = __hadd2(rank0[idx], rank1[idx]);
        }

        __syncthreads();
        if (threadIdx.x == 0) {
            self_signal->_flag[signal_id] = epoch + 1;
        }
        return true;
    }

#if 0
    template<class FragC, class Pred>
    __device__ void
    Reduce(FragC& frag_C, int splits, int64_t split_size, const int2& cta_cs, Pred& pred, const EpilogueParam& param)
    {
        using Vec         = OutputC<Dtype>;
        const int2 thr_cs = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE);
        for (int k = 0; k < splits; ++k) {
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    const int     ss  = thr_cs.y + s * Map::kDeltaS;
                    const int     cc  = thr_cs.x + c * Map::kDeltaC;
                    const int64_t idx = k * split_size + (cta_cs.y + ss) * param.partial_C_ld + (cta_cs.x + cc);
                    if (true) {
                        Vec tmp;
                        Load(tmp, &param.partial_C[idx]);
                        using namespace ops;
                        frag_C[s][c] = frag_C[s][c] + tmp;
                    }
                }
            }
        }
    }
#endif

    template<class FragC, class Pred>
    __device__ void Reduce(FragC& frag_C, const MatrixData& p, bool is_first, bool is_last, int2 cs0, Pred& pred)
    {
        constexpr int dc = sizeof(Dtype) * Map::kDeltaC;
        const int     ds = sizeof(Dtype) * Map::kDeltaS * p.ptr.stride;

        char* ptr = (char*)p.ptr.ptr + sizeof(Dtype) * dot(cs0, long2{1, p.ptr.stride});

        Pred ld_mask = is_first ? Pred{} : pred;
        Pred st_mask = is_last ? Pred{} : pred;

        PRAGMA_UNROLL
        for (int s = 0; s < S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < C; ++c) {
                OutputC<Dtype> tmp{};  // ! ZERO-filled
                if (ld_mask(s, c)) {
                    Load(tmp, reinterpret_cast<Dtype*>(ptr));
                }
                if (1) {
                    using namespace ops;
                    frag_C[s][c] = frag_C[s][c] + tmp;
                }
                if (st_mask(s, c)) {
                    Store(reinterpret_cast<Dtype*>(ptr), frag_C[s][c]);
                }
                ptr += dc;
            }
            ptr -= dc * C;
            ptr += ds;
        }
    }

    template<class FragC>
    __device__ void operator()(FragC&               frag_C,
                               const int4&          tile_offset,
                               const int2&          extents,
                               int                  splits,
                               int                  tile_id,
                               bool                 is_last,
                               const EpilogueParam& param,
                               SharedStorage&       storage)
    {
        const int2 cta_cs = mk2cs<kOrder>(tile_offset.x * M, tile_offset.y * N);
        const int2 end_cs = mk2cs<kOrder>(extents);

        OutputC<Dtype> tmp_C[S][C];

        Rearrange(frag_C, storage, tmp_C);

        Predicate<S, C, false, false> pred{};  //  1 regs

        const int2 thr_cs = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE);
        const int2 cs0    = {cta_cs.x + thr_cs.x, cta_cs.y + thr_cs.y};

        PRAGMA_UNROLL
        for (int s = 0; s < S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < C; ++c) {
                const int ss = thr_cs.y + s * Map::kDeltaS;
                const int cc = thr_cs.x + c * Map::kDeltaC;
                if (ss < end_cs.y && cc < end_cs.x) {
                    pred.set(s, c);
                }
            }
        }

        if (SplitK_ && splits > 1) {
            int* barrier = &param.locks[tile_id];

            sem_wait(barrier, tile_offset.z, threadIdx.x == 0);

            const MatrixData p = resolve<Dtype, kMode>(param.partials, tile_offset.w);

            Reduce(tmp_C, p, tile_offset.z == 0, is_last, cs0, pred);

            const int post_id = is_last ? 0 : tile_offset.z + 1;
            sem_post(barrier, post_id, threadIdx.x == 0);

            if (!is_last) {
                return;
            }
        }

        constexpr pair<Map::kDeltaC, Map::kDeltaS> delta_cs{};

        // opt-in scaling
        // Scale(scale_SC{}, mode_SC{}, delta_cs, tmp_C, param.scale_S, param.scale_C, tile_offset.w, cs0, pred);

        param.combine_mat((Tc*)0, constant<kMode>{}, tmp_C, cs0, tile_offset.w, delta_cs, pred);

        const MatrixData c = resolve<Tc, kMode>(param.c, tile_offset.w);

        bool publish_tile_allreduce = false;
        if (param.moe_weighted_reduce) {
            if constexpr (std::is_same_v<Tc, half_t>) {
                StoreMoeWeightedReduce(tmp_C, cs0, tile_offset.w, pred, param);
            }
        }
        else if (param.silu_act) {
            constexpr int dc  = sizeof(Tc) * Map::kDeltaC / 2;
            const int     ds  = sizeof(Tc) * Map::kDeltaS * c.ptr.stride;
            auto          ptr = (char*)c.ptr.ptr + sizeof(Tc) * dot({cs0.x / 2, cs0.y}, long2{1, c.ptr.stride});
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    GatedActivation<Silu>::template apply<Tc>(tmp_C[s][c]);
                    if (pred(s, c)) {
                        const auto tmp = cast<Tc>((Array<Dtype, kAccess / 2>&)tmp_C[s][c]);
                        Store(reinterpret_cast<Tc*>(ptr), tmp);
                    }
                    ptr += dc;
                }
                ptr -= dc * C;
                ptr += ds;
            }
        }
        else {
            StoreC<Tc>(tmp_C, c, cs0, pred);
            publish_tile_allreduce = true;
        }

        if (publish_tile_allreduce) {
            if (param.tile_allreduce && param.tile_allreduce_param.kernel_reducer_blocks > 0) {
                PublishTileAllReduceFlag(tile_offset, tile_id, is_last, param);
            }
            else if (!StoreTailAllReduceC(tile_offset, is_last, param)) {
                PublishTileAllReduceFlag(tile_offset, tile_id, is_last, param);
                StoreTileAllReduceC(tmp_C, c, cs0, tile_offset, is_last, pred, param);
            }
        }
    }
};

// Opt-in wrapper that changes only the serial split-K wait policy.
template<class Base>
struct EpilogueLane0Wait : Base {
    using Base::Base;

    using Dtype = typename Base::Dtype;
    using Tc    = typename Base::Tc;

    static constexpr auto kOrder = Base::kOrder;
    static constexpr auto kMode  = Base::kMode;
    static constexpr bool SplitK = Base::SplitK;

    using SharedStorage = typename Base::SharedStorage;
    using Map           = typename Base::Map;

    static constexpr int S       = Base::S;
    static constexpr int C       = Base::C;
    static constexpr int kAccess = Base::kAccess;
    static constexpr int2 kMN    = cs2mk<kOrder>(Map::kDimC, Map::kDimS);

    template<class T>
    using OutputC = typename Base::template OutputC<T>;

    template<class FragC>
    __device__ void operator()(FragC&               frag_C,
                               const int4&          tile_offset,
                               const int2&          extents,
                               int                  splits,
                               int                  tile_id,
                               bool                 is_last,
                               const EpilogueParam& param,
                               SharedStorage&       storage)
    {
        const int2 cta_cs = mk2cs<kOrder>(tile_offset.x * kMN.x, tile_offset.y * kMN.y);
        const int2 end_cs = mk2cs<kOrder>(extents);

        OutputC<Dtype> tmp_C[S][C];

        this->Rearrange(frag_C, storage, tmp_C);

        Predicate<S, C, false, false> pred{};

        const int2 thr_cs = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE);
        const int2 cs0    = {cta_cs.x + thr_cs.x, cta_cs.y + thr_cs.y};

        PRAGMA_UNROLL
        for (int s = 0; s < S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < C; ++c) {
                const int ss = thr_cs.y + s * Map::kDeltaS;
                const int cc = thr_cs.x + c * Map::kDeltaC;
                if (ss < end_cs.y && cc < end_cs.x) {
                    pred.set(s, c);
                }
            }
        }

        if (SplitK && splits > 1) {
            int* barrier = &param.locks[tile_id];

            sem_wait_lane0(barrier, tile_offset.z);

            const MatrixData p = resolve<Dtype, kMode>(param.partials, tile_offset.w);

            this->Reduce(tmp_C, p, tile_offset.z == 0, is_last, cs0, pred);

            const int post_id = is_last ? 0 : tile_offset.z + 1;
            sem_post(barrier, post_id, threadIdx.x == 0);

            if (!is_last) {
                return;
            }
        }

        constexpr pair<Map::kDeltaC, Map::kDeltaS> delta_cs{};

        param.combine_mat((Tc*)0, constant<kMode>{}, tmp_C, cs0, tile_offset.w, delta_cs, pred);

        const MatrixData c = resolve<Tc, kMode>(param.c, tile_offset.w);

        bool publish_tile_allreduce = false;
        if (param.moe_weighted_reduce) {
            if constexpr (std::is_same_v<Tc, half_t>) {
                this->StoreMoeWeightedReduce(tmp_C, cs0, tile_offset.w, pred, param);
            }
        }
        else if (param.silu_act) {
            constexpr int dc  = sizeof(Tc) * Map::kDeltaC / 2;
            const int     ds  = sizeof(Tc) * Map::kDeltaS * c.ptr.stride;
            auto          ptr = (char*)c.ptr.ptr + sizeof(Tc) * dot({cs0.x / 2, cs0.y}, long2{1, c.ptr.stride});
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    GatedActivation<Silu>::template apply<Tc>(tmp_C[s][c]);
                    if (pred(s, c)) {
                        const auto tmp = cast<Tc>((Array<Dtype, kAccess / 2>&)tmp_C[s][c]);
                        Store(reinterpret_cast<Tc*>(ptr), tmp);
                    }
                    ptr += dc;
                }
                ptr -= dc * C;
                ptr += ds;
            }
        }
        else {
            this->template StoreC<Tc>(tmp_C, c, cs0, pred);
            publish_tile_allreduce = true;
        }

        if (publish_tile_allreduce) {
            if (param.tile_allreduce && param.tile_allreduce_param.kernel_reducer_blocks > 0) {
                this->PublishTileAllReduceFlag(tile_offset, tile_id, is_last, param);
            }
            else if (!this->StoreTailAllReduceC(tile_offset, is_last, param)) {
                this->PublishTileAllReduceFlag(tile_offset, tile_id, is_last, param);
                this->StoreTileAllReduceC(tmp_C, c, cs0, tile_offset, is_last, pred, param);
            }
        }
    }
};

}  // namespace turbomind::gemm
