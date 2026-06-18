// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/gemm/context.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/dispatch_cache.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/tuner/params.h"
#include "src/turbomind/kernels/gemm/tuner/sampler.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iterator>
#include <mutex>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <vector>

namespace turbomind::gemm {

void ExportDispatchCache(std::ostream& os, const std::vector<std::pair<GemmDesc, LaunchSpec>>& entries);

void ImportDispatchCache(std::istream&                                 is,
                         std::vector<std::pair<GemmDesc, LaunchSpec>>& entries,
                         const std::vector<std::unique_ptr<Kernel>>&   kernels);

namespace {

template<class Cmp>
std::vector<int> ArgSort(size_t size, const Cmp& cmp)
{
    std::vector<int> idxs(size);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::stable_sort(idxs.begin(), idxs.end(), cmp);
    return idxs;
}

bool GemmTraceEnabled()
{
    const char* raw = std::getenv("TM_GEMM_TRACE");
    return raw && std::atoi(raw) != 0;
}

int GemmTraceLimit()
{
    const char* raw = std::getenv("TM_GEMM_TRACE_LIMIT");
    return raw ? std::max(std::atoi(raw), 0) : 256;
}

bool GemmTraceFilterAllows(const std::string& desc)
{
    const char* raw = std::getenv("TM_GEMM_TRACE_FILTER");
    return !raw || !*raw || desc.find(raw) != std::string::npos;
}

const char* ToString(DispatchPolicy policy)
{
    if ((policy & DispatchPolicy::kPreserveDefaultSplits)
        || (policy & DispatchPolicy::kPreserveDefaultSplitCount)) {
        static thread_local std::string text;
        auto base =
            static_cast<DispatchPolicy>((int)policy & ~(int)DispatchPolicy::kPreserveDefaultSplits
                                        & ~(int)DispatchPolicy::kPreserveDefaultSplitCount);
        text = std::string(ToString(base));
        if (policy & DispatchPolicy::kPreserveDefaultSplits) {
            text += "|preserve_default_splits";
        }
        if (policy & DispatchPolicy::kPreserveDefaultSplitCount) {
            text += "|preserve_default_split_count";
        }
        return text.c_str();
    }
    switch (policy) {
        case DispatchPolicy::kDefault:
            return "default";
        case DispatchPolicy::kMeasure:
            return "measure";
        case DispatchPolicy::kReuse:
            return "reuse";
        case DispatchPolicy::kAppend:
            return "append";
        default:
            return "unknown";
    }
}

void MaybeTraceGemmDispatch(const GemmDesc& desc, DispatchPolicy policy, const LaunchSpec& spec, bool measured)
{
    if (!GemmTraceEnabled()) {
        return;
    }
    const std::string desc_str = to_string(desc);
    if (!GemmTraceFilterAllows(desc_str)) {
        return;
    }
    static std::atomic<int> logged{0};
    const int limit = GemmTraceLimit();
    if (limit > 0 && logged.fetch_add(1, std::memory_order_relaxed) >= limit) {
        return;
    }

    std::cerr << "[TM_GEMM_TRACE] desc=" << desc_str << " policy=" << ToString(policy) << " measured="
              << (measured ? 1 : 0);
    if (spec.kernel) {
        const int3 cta = spec.kernel->cta_tile_size();
        const int3 mma = spec.kernel->warp_tile_size();
        std::cerr << " kernel=" << spec.kernel->name() << " splits=" << spec.splits << " swizzle=" << spec.swizzle
                  << " cta=" << cta.x << "x" << cta.y << "x" << cta.z << " mma=" << mma.x << "x" << mma.y
                  << "x" << mma.z << " stages=" << spec.kernel->stages() << " smem=" << spec.kernel->smem_size();
    }
    else {
        std::cerr << " kernel=<none>";
    }
    std::cerr << std::endl;
}

}  // namespace

struct Gemm::Impl {

    Impl():
        props_{GetCudaDeviceProps()},
        arch_{props_->major * 100 + props_->minor * 10},
        registry_{props_},
        cache_{registry_.kernels()}
    {
        if (arch_ == 700) {
            // V100 decode is dominated by many tiny GEMM/GEMV problems. A
            // broader search space consistently finds better launch specs than
            // the generic defaults for these SM70 workloads.
            tuning_.max_splits = 16;
            tuning_.max_waves  = 32;
            tuning_.swizzle    = {0, 1, 2, 3, 4};
            tuning_.top_k      = 0;
            tuning_.clusters   = 0;
            tuning_.min_iter   = 2;
            tuning_.max_iter   = 20;
            tuning_.max_time   = 2.f;
        }
        if (auto str = std::getenv("TM_GEMM_TUNE")) {
            try {
                ParseTuningParams(tuning_, str);
            }
            catch (...) {
                std::cerr << "[Gemm2] Failed to parse `TM_GEMM_TUNE`, default value will be used.\n";
                tuning_ = {};
            }
        }
        if (std::getenv("TM_GEMM_WARN_CACHE_MISS")) {
            warn_cache_miss_ = true;
        }
        measurer_.emplace(CreateStoppingCriterion(tuning_.min_iter, tuning_.max_iter, tuning_.max_time));
    }

    // find launch spec in dispatch cache, dispatch by heuristic on cache miss
    LaunchSpec Dispatch(Context& ctx, DispatchPolicy policy, size_t barriers_size, size_t partials_size)
    {
        const auto& desc = ctx.desc();
        if (policy & DispatchPolicy::kReuse) {
            if (auto spec = cache_.LowerBound(desc)) {
                return *spec;
            }
            if (warn_cache_miss_) {
                std::cerr << "Failed to find a feasible kernel in the cache, will dispatch by heuristic: "
                          << to_string(ctx.desc()) << std::endl;
            }
        }

        if (auto spec = cache_.Find(desc)) {
            return *spec;
        }

        auto specs = Find(ctx, barriers_size, partials_size, 1);
        if (!specs.empty()) {
            cache_.Insert(desc, specs.front());
            return specs.front();
        }
        return {};
    }

    std::vector<LaunchSpec> Find(Context& ctx, size_t barrier_size, size_t partials_size, int top_k)
    {
        std::vector<Kernel*> feasible = ctx.Filter(registry_.kernels());

        std::vector<std::vector<LaunchSpec>> clusters;
        {
            std::vector<LaunchSpec> tmp;
            tmp.reserve(feasible.size());
            for (const auto& k : feasible) {
                LaunchSpec spec{k};
                tmp.push_back(spec);
            }
            clusters = Cluster(tmp, ClusteringParam{false, true});
        }
        std::vector<Kernel*> proxies;
        proxies.reserve(clusters.size());

        for (const auto& c : clusters) {
            proxies.push_back(c.front().kernel);
        }

        std::vector<std::pair<int, LaunchSpec>> specs;

        PopulateParam param{};
        param.max_splits    = tuning_.max_splits;
        param.max_waves     = tuning_.max_waves;
        param.swizzle       = tuning_.swizzle.at(0);
        param.barriers_size = barrier_size;
        param.partials_size = partials_size;

        for (int cluster_id = 0; cluster_id < (int)proxies.size(); ++cluster_id) {
            auto& kernel = *proxies[cluster_id];

            auto tmp = ctx.Populate(kernel, param);
            for (const auto& s : tmp) {
                specs.emplace_back(cluster_id, s);
            }
        }

        // std::cerr << "#kernel: " << kernels.size() << ", #cluster: " << clusters.size()
        //           << ", #metric: " << metrics.size() << "\n";

        int64_t mio_max = 0;
        int64_t mma_max = 0;
        for (const auto& [_, s] : specs) {
            auto& [mio, mma] = s.estimated;
            mio_max          = std::max(mio_max, mio);
            mma_max          = std::max(mma_max, mma);
        }
        std::vector<float> mio_ratio;
        std::vector<float> mma_ratio;
        std::vector<float> avg_ratio;
        for (const auto& [_, s] : specs) {
            auto& [mio, mma] = s.estimated;
            mio_ratio.push_back((float)mio / mio_max);
            mma_ratio.push_back((float)mma / mma_max);
            avg_ratio.push_back(.5 * (mio_ratio.back() + mma_ratio.back()));
        }
        auto idxs = ArgSort(specs.size(), [&](int i, int j) {  //
            return avg_ratio[i] < avg_ratio[j];
        });

        // for (const auto& i : idxs) {
        //     auto [cid, s, m] = metrics[i];
        //     std::cout << clusters[cid].front().kernel->name() << " s" << s << " " << avg_ratio[i] << " " <<
        //     mio_ratio[i]
        //               << " " << mma_ratio[i] << " " << m.mio_cost << " " << m.mma_cost << "\n";
        // }

        top_k = top_k > 0 ? std::min<int>(idxs.size(), top_k) : (int)idxs.size();
        std::vector<LaunchSpec> ret;
        ret.reserve(top_k);
        for (int i = 0; i < top_k; ++i) {
            const auto& [cluster_id, spec] = specs[idxs[i]];
            // Apply `splits` to all kernels in the cluster
            for (const auto& s : clusters[cluster_id]) {
                auto tmp   = spec;
                tmp.kernel = s.kernel;
                ret.push_back(tmp);
            }
        }

        return ret;
    }

    template<class LaunchFunc>
    int Measure(
        Context& ctx, size_t barriers_size, size_t partials_size, int top_k, LaunchFunc launch_func, cudaStream_t st)
    {
        // Early exit on exact match
        if (cache_.Find(ctx.desc())) {
            return 0;
        }
        // std::cerr << "GEMM: " << desc.m << "x" << desc.n << "x" << desc.k << "\n";

        const auto tmp = Find(ctx, barriers_size, partials_size, tuning_.top_k);

        std::vector<LaunchSpec> specs;
        for (const auto& spec : tmp) {
            // populate swizzle parameters
            const auto swis = ctx.Swizzle(spec, tuning_.swizzle);
            specs.insert(specs.end(), swis.begin(), swis.end());
        }

        specs = Sampler{*measurer_, tuning_.clusters}.Run(specs, launch_func, st);

        // for (const auto& s : specs) {
        //     std::cout << s.kernel->name()          //
        //               << " swizzle=" << s.swizzle  //
        //               << ", splits=" << s.splits   //
        //               << ", measured=" << s.measured << "ms\n";
        //     break;
        // }

        if (!specs.empty()) {
            cache_.Insert(ctx.desc(), specs.front());
        }
        else {
            std::cerr << "No valid kernel found for the problem\n";
            return -1;
        }

        return 0;
    }

    /// TODO: move to cuda utils
    static std::unique_ptr<cudaDeviceProp> GetCudaDeviceProps()
    {
        auto props     = std::make_unique<cudaDeviceProp>();
        int  device_id = -1;
        cudaGetDevice(&device_id);
        cudaGetDeviceProperties(props.get(), device_id);
        return props;
    }

    std::shared_ptr<cudaDeviceProp> props_;

    int arch_;

    Registry registry_;

    TuningParams tuning_;

    bool warn_cache_miss_{};

    std::optional<Measurer> measurer_;

    DispatchCache cache_;

    std::mutex dispatch_mutex_;
};

// implementation of GEMM interfaces

Gemm::Gemm(): impl_{new Impl{}} {}

Gemm::~Gemm() = default;

int Gemm::Run(const Operation&    operation,
              float               alpha,
              const void*         A,
              const MatrixLayout& Adesc,
              const void*         U,
              const MatrixLayout& Udesc,
              const void*         B,
              const MatrixLayout& Bdesc,
              const void*         V,
              const MatrixLayout& Vdesc,
              float               beta,
              const void*         C,
              const MatrixLayout& Cdesc,
              void*               D,
              const MatrixLayout& Ddesc,
              const Workspace&    workspace,
              cudaStream_t        stream)
{

    Context context{*impl_->props_};

    const auto desc = context.Init(operation, Adesc, Udesc, Bdesc, Vdesc, Cdesc, Ddesc);

    if (!desc) {
        fprintf(stderr, "invalid argument.\n");
        TM_CHECK(0);
        return -1;
    }

    const auto launch = [=](LaunchSpec spec, cudaStream_t st) {
        auto _workspace = workspace;
        return spec.kernel->Launch(operation,
                                   alpha,
                                   A,
                                   Adesc,
                                   U,
                                   Udesc,
                                   B,
                                   Bdesc,
                                   V,
                                   Vdesc,
                                   beta,
                                   C,
                                   Cdesc,
                                   D,
                                   Ddesc,
                                   spec.swizzle,
                                   spec.splits,
                                   _workspace,
                                   st);
    };

    std::optional<Context> dispatch_context_storage;
    Context*               dispatch_context = &context;
    if (operation.dispatch_num_override > 0 && operation.dispatch_num_override != context.desc().num) {
        MatrixLayout dispatch_Adesc = Adesc;
        MatrixLayout dispatch_Bdesc = Bdesc;
        MatrixLayout dispatch_Ddesc = Ddesc;
        dispatch_Adesc.num          = operation.dispatch_num_override;
        dispatch_Bdesc.num          = operation.dispatch_num_override;
        dispatch_Ddesc.num          = operation.dispatch_num_override;
        dispatch_context_storage.emplace(*impl_->props_);
        if (dispatch_context_storage->Init(
                operation, dispatch_Adesc, Udesc, dispatch_Bdesc, Vdesc, Cdesc, dispatch_Ddesc)) {
            dispatch_context = &*dispatch_context_storage;
        }
        else {
            dispatch_context_storage.reset();
            dispatch_context = &context;
        }
    }

#if 0
    if (operation.reserved) {
        auto specs = impl_->Find(context, workspace.barriers_size, workspace.partials_size, 0);
        auto cases = (std::vector<std::function<LaunchSpec()>>*)operation.reserved;
        for (const auto& spec : specs) {
            cases->push_back([=] {
                launch(spec, stream);
                return spec;
            });
        }
        return -1;
    }
#endif

    LaunchSpec spec{};

    const bool measured = operation.dispatch & DispatchPolicy::kMeasure;
    {
        std::lock_guard<std::mutex> lock(impl_->dispatch_mutex_);
        if (measured) {
            impl_->Measure(*dispatch_context, workspace.barriers_size, workspace.partials_size, 1, launch, stream);
        }

        spec = impl_->Dispatch(*dispatch_context, operation.dispatch, workspace.barriers_size, workspace.partials_size);
        const bool preserve_default_kernel = operation.dispatch & DispatchPolicy::kPreserveDefaultSplits;
        const bool preserve_default_split_count = operation.dispatch & DispatchPolicy::kPreserveDefaultSplitCount;
        if (spec.kernel && (preserve_default_kernel || preserve_default_split_count)) {
            auto default_specs = impl_->Find(*dispatch_context, workspace.barriers_size, workspace.partials_size, 1);
            if (!default_specs.empty()) {
                const auto default_spec = default_specs.front();
                if (preserve_default_kernel) {
                    spec.kernel = default_spec.kernel;
                }
                spec.splits = default_spec.splits;
                const auto& default_desc = dispatch_context->get_desc(*spec.kernel);
                spec.swizzle = std::min(
                    spec.swizzle,
                    spec.kernel->GetMaxSwizzle({
                        default_desc.m,
                        default_desc.n,
                        default_desc.k,
                        default_desc.num,
                    }));
            }
        }
    }
    if (spec.kernel && dispatch_context != &context) {
        const auto& actual_desc = context.get_desc(*spec.kernel);
        spec.swizzle =
            std::min(spec.swizzle, spec.kernel->GetMaxSwizzle({actual_desc.m, actual_desc.n, actual_desc.k, actual_desc.num}));
    }
    MaybeTraceGemmDispatch(dispatch_context->desc(), operation.dispatch, spec, measured);

    if (spec.kernel) {
        // std::cout << "[Gemm] dispatch: " << spec.kernel->name()  //
        //           << " split_k=" << spec.splits                  //
        //           << " swizzle=" << spec.swizzle << std::endl;
        return launch(spec, stream);
    }

    TM_CHECK(0) << "No feasible kernel found for the problem: " << to_string(context.desc());

    return -1;
}

int Gemm::Export(std::ostream& os)
{
    return impl_->cache_.Export(os);
}

int Gemm::Import(std::istream& is)
{
    return impl_->cache_.Import(is);
}

std::vector<int> Gemm::GetTuningSeq() const
{
    return impl_->tuning_.seq;
}

}  // namespace turbomind::gemm
