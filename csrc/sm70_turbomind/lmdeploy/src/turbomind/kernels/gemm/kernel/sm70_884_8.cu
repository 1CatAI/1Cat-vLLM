// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/batch_kernel_sm70.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

using namespace sm70_s884;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

namespace {

template <class Gemm>
class Qwen38PrescaledFullTileKernelImpl final : public KernelImpl<Gemm> {
 public:
  Qwen38PrescaledFullTileKernelImpl() {
    this->info_.name += "_sm70_fp8_pscale_full";
  }

  bool is_feasible(const GemmDesc& desc) const noexcept override {
    return desc.m == 8000 && (desc.n == 4096 || desc.n == 3584) &&
           desc.k == 5120 && desc.num == 1 &&
           KernelImpl<Gemm>::is_feasible(desc);
  }
};

template <class Gemm>
class Fp8PrescaledM1KernelImpl final : public KernelImpl<Gemm> {
 public:
  Fp8PrescaledM1KernelImpl() {
    this->info_.name += "_sm70_fp8_pscale_m1";
  }

  bool is_feasible(const GemmDesc& desc) const noexcept override {
    if (desc.m != 1 || desc.num != 1) {
      return false;
    }
    const bool accepted_shape =
        (desc.n == 1536 && desc.k == 4096) ||
        (desc.n == 8192 && desc.k == 1024) ||
        (desc.n == 4096 && desc.k == 2048) ||
        (desc.n == 1024 && desc.k == 4096) ||
        (desc.n == 4096 && desc.k == 512);
    return accepted_shape && KernelImpl<Gemm>::is_feasible(desc);
  }
};

template <class Gemm>
class Fp8PrescaledBatchKernelImpl final : public KernelImpl<Gemm> {
 public:
  Fp8PrescaledBatchKernelImpl() {
    this->info_.name += "_sm70_fp8_pscale_batch";
  }

  bool is_feasible(const GemmDesc& desc) const noexcept override {
    return desc.m > 32 && desc.m <= 64 && desc.num == 1 &&
           ((desc.n == 5120 && desc.k == 1536) ||
            ((desc.n == 4096 || desc.n == 3584) && desc.k == 5120)) &&
           KernelImpl<Gemm>::is_feasible(desc);
  }
};

}  // namespace

void Registry::sm70_884_8() {
  {
    using B = Config_QuantizedBatch<fp8_e4m3_t, kColMajor>;
    using Rows32 = B::Type<32, 256, 32, 1, 4, 1, D, S, 2, true,
                          1, 128, 32, 128>;
    using Rows64 = B::Type<64, 256, 32, 1, 4, 1, D, S, 2, true,
                          1, 128, 64, 128>;
    using Full64 = B::Type<64, 128, 64, 2, 4, 1, D, S, 2, true,
                          1, 128, 64, 128, 1, true>;
    Add(std::make_unique<DenseBatchSupplyKernelImpl<typename Rows32::Kernel>>());
    Add(std::make_unique<DenseBatchSupplyKernelImpl<typename Rows64::Kernel>>());
    Add(std::make_unique<DenseBatchSupplyKernelImpl<typename Full64::Kernel, true>>());
  }

  if constexpr (1) {
    // clang-format off
        using C = Config_E4M3<kColMajor, 0>;
        Add<C::Type<128, 256,  16, 2, 4, 1, D, D, 2, true, 1, 128, 128, 128>>();
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 128,  64, 128>>();
        Add<C::Type< 96, 128,  32, 2, 2, 1, D, S, 2, true, 1, 128,  48, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>>();
        Add<C::Type< 64, 256,  16, 1, 4, 1, D, S, 2, true, 1, 128,  64, 128>>();
        Add<C::Type< 32, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 256,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();

        using CP = Config_E4M3_Prescaled<kColMajor, 0>;
        using Q38PrescaledFull = CP::Type<64, 256, 16, 1, 4, 1, D, S, 2, true, 1, 128, 64, 128, 1, true>;
        Add(std::make_unique<Qwen38PrescaledFullTileKernelImpl<typename Q38PrescaledFull::Kernel>>());

        using Fp8PrescaledM1 = CP::Type<8, 128, 64, 1, 4, 1, D, S, 2, true, 1, 128>;
        Add(std::make_unique<Fp8PrescaledM1KernelImpl<typename Fp8PrescaledM1::Kernel>>());

        using Fp8PrescaledBatch32 = CP::Type<32, 256, 32, 1, 4, 1, D, S, 2, true, 1, 128, 32, 128>;
        using Fp8PrescaledBatch16 = CP::Type<16, 256, 32, 1, 4, 1, D, S, 2, true, 1, 128>;
        Add(std::make_unique<Fp8PrescaledBatchKernelImpl<typename Fp8PrescaledBatch32::Kernel>>());
        Add(std::make_unique<Fp8PrescaledBatchKernelImpl<typename Fp8PrescaledBatch16::Kernel>>());
    // clang-format on
  }
}

}  // namespace turbomind::gemm
