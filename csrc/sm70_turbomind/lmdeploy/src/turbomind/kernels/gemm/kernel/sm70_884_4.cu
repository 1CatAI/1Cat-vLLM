// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/batch_kernel_sm70.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/types.h"

#include <cstdlib>

namespace turbomind::gemm {

using namespace sm70_s884;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

namespace {

// Keep shape-specific small-N tactics out of prefill and unrelated CUDA graph
// shapes.
template <class Gemm, int ExactM>
class ExactMKernelImpl final : public KernelImpl<Gemm> {
 public:
  bool is_feasible(const GemmDesc& desc) const noexcept override {
    return desc.m == ExactM && KernelImpl<Gemm>::is_feasible(desc);
  }
};

template <class Gemm, int ExactM, int ExactN, int ExactK>
class ExactMnkKernelImpl final : public KernelImpl<Gemm> {
 public:
  bool is_feasible(const GemmDesc& desc) const noexcept override {
    return desc.m == ExactM && desc.n == ExactN && desc.k == ExactK &&
           KernelImpl<Gemm>::is_feasible(desc);
  }
};

// Default-cache-B kernel for the exact Qwen3.8 TP4 W2 prefill descriptor.
// Expert-sorted prefill gives each expert several adjacent M tiles, so keeping
// B cacheable may reuse its FP4 weights across those tiles. The exact contract
// and default-on feature gate keep it out of unrelated descriptors and provide
// an operational rollback.
template <class Gemm>
class Qwen38Nvfp4W2CacheBKernelImpl final : public KernelImpl<Gemm> {
 public:
  bool is_feasible(const GemmDesc& desc) const noexcept override {
    const char* enabled =
        std::getenv("VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL");
    return (!enabled || std::atoi(enabled) != 0) && desc.m >= 1280 &&
           desc.num == 512 && desc.n == 2560 && desc.k == 160 &&
           KernelImpl<Gemm>::is_feasible(desc);
  }
};

// The full W13 N=320 shape is faster with N128 tiles, but its final tile does
// half-empty work. This N64 kernel is exposed only for the exact split-W13
// tail, while the first 256 columns retain the established N128 kernel.
template <class Gemm>
class Qwen38Nvfp4W13TailN64KernelImpl final : public KernelImpl<Gemm> {
 public:
  bool is_feasible(const GemmDesc& desc) const noexcept override {
    const char* enabled =
        std::getenv("VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL");
    return (!enabled || std::atoi(enabled) != 0) && desc.m >= 1280 &&
           desc.num == 512 && desc.n == 64 && desc.k == 2560 &&
           KernelImpl<Gemm>::is_feasible(desc);
  }
};

}  // namespace

void Registry::sm70_884_4() {
  {
    auto add_planes = [this]<class C, class G, int GroupSize>() {
      // Metadata holds the high plane for one full group. CTA K must be a
      // multiple of that group so the register code position restarts at 0.
      constexpr int PrefillK = GroupSize == 16 ? 16 : 32;
      Add<typename C::template Type<128, 256, PrefillK, 2, 4, 1, D, D, 2, true,
                                    1, GroupSize, 128, 128>>();
      Add<typename C::template Type<128, 128, 32, 2, 2, 1, D, S, 2, true, 1, GroupSize>>();
      Add<typename C::template Type<64, 128, 32, 1, 4, 1, D, S, 2, true, 1, GroupSize>>();
      Add<typename C::template Type<32, 128, 32, 1, 4, 1, D, S, 2, true, 1, GroupSize>>();
      Add<typename C::template Type<16, 128, 32, 1, 4, 1, D, S, 2, true, 1, GroupSize>>();
      Add<typename C::template Type<8, 128, 32, 1, 4, 1, D, S, 2, true, 1, GroupSize>>();
      Add<typename G::template Type<128, 128, 32, 2, 2, 1, D, S, 2, true, 1, GroupSize>>();
      Add<typename G::template Type<64, 128, 32, 1, 4, 1, D, S, 2, true, 1, GroupSize>>();
      Add<typename G::template Type<32, 128, 32, 1, 4, 1, D, S, 2, true, 1, GroupSize>>();
      Add<typename G::template Type<16, 128, 32, 1, 4, 1, D, S, 2, true, 1, GroupSize>>();
      Add<typename G::template Type<8, 128, 32, 1, 4, 1, D, S, 2, true, 1, GroupSize>>();
    };
    add_planes.template operator()<Config_GgufBitPlane<2, 1, 16, kColMajor>,
                                   Config_GgufBitPlane<2, 1, 16, kColMajor, 0>, 16>();
    add_planes.template operator()<Config_GgufBitPlane<4, 1, 32, kColMajor>,
                                   Config_GgufBitPlane<4, 1, 32, kColMajor, 0>, 32>();
    add_planes.template operator()<Config_GgufBitPlane<4, 2, 16, kColMajor>,
                                   Config_GgufBitPlane<4, 2, 16, kColMajor, 0>, 16>();
  }
  {
    // Two-bit affine weights share the FP16 MMA pipeline and coefficient
    // carrier. Keep separate dense/grouped descriptors for group16/group32.
    using C = Config_U2_Affine<kColMajor>;
    using G = Config_U2_Affine<kColMajor, 0>;
    // Keep the native prefill tile repertoire: the source code width does
    // not require halving N or doubling the MMA K tile.
    Add<C::Type<128, 256, 16, 2, 4, 1, D, D, 2, true, 1, 16, 128, 128>>();
    Add<C::Type<128, 128, 16, 2, 2, 1, D, D, 2, true, 1, 16, 64, 128>>();
    Add<C::Type<64, 128, 16, 1, 4, 1, D, S, 2, true, 1, 16, 32, 128>>();
    Add<G::Type<128, 128, 16, 2, 2, 1, D, D, 2, true, 1, 16, 64, 128>>();
    Add<C::Type<128, 128, 32, 2, 2, 1, D, S, 2, true, 1, 16>>();
    Add<C::Type<64, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16>>();
    Add<C::Type<32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16>>();
    Add<C::Type<16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16>>();
    Add<C::Type<8, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16>>();
    Add<G::Type<128, 128, 32, 2, 2, 1, D, S, 2, true, 1, 16>>();
    Add<G::Type<64, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16>>();
    Add<G::Type<32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16>>();
    Add<G::Type<16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16>>();
    Add<G::Type<8, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16>>();
    Add<C::Type<128, 256, 16, 2, 4, 1, D, D, 2, true, 1, 32, 128, 128>>();
    Add<C::Type<128, 128, 16, 2, 2, 1, D, D, 2, true, 1, 32, 64, 128>>();
    Add<C::Type<64, 128, 16, 1, 4, 1, D, S, 2, true, 1, 32, 32, 128>>();
    Add<G::Type<128, 128, 16, 2, 2, 1, D, D, 2, true, 1, 32, 64, 128>>();
    Add<C::Type<128, 128, 32, 2, 2, 1, D, S, 2, true, 1, 32>>();
    Add<C::Type<64, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
    Add<C::Type<32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
    Add<C::Type<16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
    Add<C::Type<8, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
    Add<G::Type<128, 128, 32, 2, 2, 1, D, S, 2, true, 1, 32>>();
    Add<G::Type<64, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
    Add<G::Type<32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
    Add<G::Type<16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
    Add<G::Type<8, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
  }
  {
    // Canonical GGUF u4 uses per-group scale plus additive bias, in the
    // existing affine metadata layout. Dense group-32 tiles complement the
    // grouped group-32 kernels below; they retain the same mma884 arithmetic.
    using C = Config_U4_d<kColMajor>;
    // Match the established dense affine tile repertoire. The metadata group
    // width is a decoding contract, not a reason to force narrower N tiles.
    Add<C::Type<128, 256, 16, 2, 4, 1, D, D, 2, true, 1, 32, 128, 128>>();
    Add<C::Type<128, 128, 16, 2, 2, 1, D, D, 2, true, 1, 32, 64, 128>>();
    Add<C::Type<64, 128, 16, 1, 4, 1, D, S, 2, true, 1, 32, 32, 128>>();
    Add<C::Type<32, 256, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
    Add<C::Type<16, 256, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
    Add<C::Type<8, 256, 64, 1, 4, 1, D, S, 2, true, 1, 32>>();
    Add<C::Type<8, 128, 64, 1, 4, 1, D, S, 2, true, 1, 32>>();
    Add<C::Type<128, 128, 32, 2, 2, 1, D, S, 2, true, 1, 32>>();
    Add<C::Type<64, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
    Add<C::Type<32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
    Add<C::Type<16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
    Add<C::Type<8, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
  }
  {
    using B = Config_QuantizedBatch<fp4_e2m1_t, kColMajor>;
    using Rows32 = B::Type<32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16, 32, 128>;
    using Full64 =
        B::Type<64, 128, 64, 2, 4, 1, D, S, 2, true, 1, 16, 64, 128, 1, true>;
    using Full48K32 =
        B::Type<48, 128, 32, 2, 4, 1, D, S, 2, true, 1, 16, 48, 128, 1, true>;
    using Full48K64 =
        B::Type<48, 128, 64, 2, 4, 1, D, S, 2, true, 1, 16, 48, 128, 1, true>;
    using Tail48K32 =
        B::Type<48, 128, 32, 2, 4, 1, D, S, 2, true, 1, 16, 48, 128, 1, true, true>;
    using Tail64 =
        B::Type<64, 128, 64, 2, 4, 1, D, S, 2, true, 1, 16, 64, 128, 1, true, true>;
    Add(std::make_unique<DenseBatchSupplyKernelImpl<typename Rows32::Kernel>>());
    Add(std::make_unique<DenseBatchSupplyKernelImpl<typename Full64::Kernel, true>>());
    Add(std::make_unique<DenseBatchSupplyKernelImpl<typename Full48K32::Kernel, true>>());
    Add(std::make_unique<DenseBatchSupplyKernelImpl<typename Full48K64::Kernel, true>>());
    Add(std::make_unique<DenseBatchSupplyKernelImpl<typename Tail48K32::Kernel, true, true>>());
    Add(std::make_unique<DenseBatchSupplyKernelImpl<typename Tail64::Kernel, true, true>>());

  }

  if constexpr (1) {
    // clang-format off
        using C = Config_U4_d<kColMajor>;
        Add<C::Type<128, 256, 16, 2, 4, 1, D, D, 2, true, 1, 128, 128, 128>>();
        Add<C::Type<128, 128, 16, 2, 2, 1, D, D, 2, true, 1, 128, 64, 128>>();
        Add<C::Type<128, 128, 16, 2, 2, 1, D, S, 2, true, 1, 128, 64, 128>>();
        Add<C::Type< 96, 128, 32, 2, 2, 1, D, S, 2, true, 1, 128, 48, 128>>();
        Add<C::Type< 64, 128, 32, 2, 2, 1, D, D, 2, true, 1, 128, 32, 128>>();
        Add<C::Type< 64, 128, 32, 2, 2, 1, D, S, 2, true, 1, 128, 32, 128>>();
        Add<C::Type< 64, 128, 16, 1, 4, 1, D, S, 2, true, 1, 128, 32, 128>>();
        Add<C::Type< 64, 256, 16, 1, 4, 1, D, S, 2, true, 1, 128, 64, 128>>();
        Add<C::Type< 32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 32, 256, 32, 1, 4, 1, D, S, 2, true, 1, 128, 32, 128>>();
        Add<C::Type< 16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 256, 32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 128, 64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 256, 64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 256, 64, 1, 4, 2, D, S, 2, true, 1, 128>>();

        using CS = Config_U4_d_A8x64Swizzle<kColMajor>;
        using C32 = CS::Type< 8, 32, 64, 1, 1, 1, D, S, 2, true, 1, 128>;
        using C64 = CS::Type< 8, 64, 64, 1, 2, 1, D, S, 2, true, 1, 128, -1, -1, 2>;
        Add(std::make_unique<ExactMKernelImpl<typename C32::Kernel, 5>>());
        Add(std::make_unique<ExactMKernelImpl<typename C64::Kernel, 5>>());
        Add(std::make_unique<ExactMnkKernelImpl<typename C64::Kernel, 1, 8704, 5120>>());
        Add(std::make_unique<ExactMnkKernelImpl<typename C64::Kernel, 1, 4096, 5120>>());

    // clang-format on
  }

  if constexpr (1) {
    // clang-format off
        // GroupSizeV=128
        using C = Config_U4_g<kColMajor>;
        Add<C::Type<128, 256,  16, 2, 4, 1, D, D, 2,   0 , 1, 128, 128, 128>>();
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 128,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>>();
        Add<C::Type< 64, 256,  16, 1, 4, 1, D, S, 2, true, 1, 128,  64, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 32, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 256,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 256,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        // GroupSizeV=64
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 64,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 64,  32, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 64>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 64>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 64>>();
        Add<C::Type<  8, 256,  64, 1, 4, 1, D, S, 2, true, 1, 64>>();
        // GroupSizeV=32
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 32,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32,  32, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 32>>();
        Add<C::Type<  8, 256,  64, 1, 4, 1, D, S, 2, true, 1, 32>>();
    // clang-format on
  }

  if constexpr (1) {
    // clang-format off
        using C = Config_MXF4<kColMajor, 0>;
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 32,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32,  32, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 32>>();
    // clang-format on
  }

  if constexpr (1) {
    // clang-format off
        using C = Config_NVF4<kColMajor, 0>;
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 16,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 16,  32, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 16>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 16>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 16>>();
        using Qwen38CacheB = C::Type<32, 128, 32, 1, 4, 1, D, D, 2, true, 1, 16>;
        Add(std::make_unique<Qwen38Nvfp4W2CacheBKernelImpl<typename Qwen38CacheB::Kernel>>());
        using Qwen38W13TailN64 = C::Type<32, 64, 32, 1, 2, 1, D, S, 2, true, 1, 16>;
        Add(std::make_unique<Qwen38Nvfp4W13TailN64KernelImpl<typename Qwen38W13TailN64::Kernel>>());
        using C32K64L1 = C::Type<8, 32, 64, 1, 1, 1, D, S, 2, true, 1, 16>;
        using C32K64L2 = C::Type<8, 32, 64, 1, 1, 1, D, S, 2, true, 1, 16, -1, -1, 2>;
        using C32K128L1 = C::Type<8, 32, 128, 1, 1, 1, D, S, 2, true, 1, 16>;
        using C32K128L2 = C::Type<8, 32, 128, 1, 1, 1, D, S, 2, true, 1, 16, -1, -1, 2>;
        Add(std::make_unique<ExactMnkKernelImpl<typename C32K64L1::Kernel, 1, 8704, 5120>>());
        Add(std::make_unique<ExactMnkKernelImpl<typename C32K64L2::Kernel, 1, 5120, 4352>>());
        Add(std::make_unique<ExactMnkKernelImpl<typename C32K128L1::Kernel, 1, 8704, 5120>>());
        Add(std::make_unique<ExactMnkKernelImpl<typename C32K128L2::Kernel, 1, 5120, 4352>>());
    // clang-format on
  }
}

namespace {

template <class Gemm>
class PrescaledNvfp4KernelImpl final : public KernelImpl<Gemm> {
 public:
  PrescaledNvfp4KernelImpl() { this->info_.name += "_sm70_nvfp4_prescaled"; }
};

template <class Ordinary, class Prescaled>
Kernel* MatchPrescaled(const Kernel& control, bool batch = false,
                       bool mask_m = false) {
  static thread_local KernelImpl<typename Ordinary::Kernel> ordinary;
  const std::string name =
      ordinary.name() + (mask_m ? "_sm70_batch_supply_mtail"
                               : (batch ? "_sm70_batch_supply" : ""));
  if (control.name() != name) {
    return nullptr;
  }
  static thread_local PrescaledNvfp4KernelImpl<typename Prescaled::Kernel>
      scaled;
  return &scaled;
}

}  // namespace

Kernel* Sm70Nvfp4PrescaledCounterpart(const Kernel& control) {
  using C = Config_NVF4<kColMajor, 0>;
  using P = Config_NVF4_Prescaled<kColMajor>;
  using B = Config_QuantizedBatch<fp4_e2m1_t, kColMajor>;
  using BP = Config_QuantizedBatch<fp4_e2m1_t, kColMajor,
                                   Transform_HMMA_SIMT_B_PrescaledE2M1>;
  // Match the ordinary tile and split partition before changing its transform.
  if (auto* k = MatchPrescaled<B::Type<32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16, 32, 128>, BP::Type<32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16, 32, 128>>(control, true))
    return k;
  if (auto* k = MatchPrescaled<B::Type<64, 128, 64, 2, 4, 1, D, S, 2, true, 1, 16, 64, 128, 1, true>, BP::Type<64, 128, 64, 2, 4, 1, D, S, 2, true, 1, 16, 64, 128, 1, true>>(control, true))
    return k;
  if (auto* k = MatchPrescaled<B::Type<48, 128, 32, 2, 4, 1, D, S, 2, true, 1, 16, 48, 128, 1, true>, BP::Type<48, 128, 32, 2, 4, 1, D, S, 2, true, 1, 16, 48, 128, 1, true>>(control, true))
    return k;
  if (auto* k = MatchPrescaled<B::Type<48, 128, 64, 2, 4, 1, D, S, 2, true, 1, 16, 48, 128, 1, true>, BP::Type<48, 128, 64, 2, 4, 1, D, S, 2, true, 1, 16, 48, 128, 1, true>>(control, true))
    return k;
  if (auto* k = MatchPrescaled<B::Type<48, 128, 32, 2, 4, 1, D, S, 2, true, 1, 16, 48, 128, 1, true, true>, BP::Type<48, 128, 32, 2, 4, 1, D, S, 2, true, 1, 16, 48, 128, 1, true, true>>(control, true, true))
    return k;
  if (auto* k = MatchPrescaled<B::Type<64, 128, 64, 2, 4, 1, D, S, 2, true, 1, 16, 64, 128, 1, true, true>, BP::Type<64, 128, 64, 2, 4, 1, D, S, 2, true, 1, 16, 64, 128, 1, true, true>>(control, true, true))
    return k;
  if (auto* k = MatchPrescaled<
          C::Type<128, 128, 16, 2, 2, 1, D, D, 2, true, 1, 16, 64, 128>,
          P::Type<128, 128, 16, 2, 2, 1, D, D, 2, true, 1, 16, 64, 128>>(
          control))
    return k;
  if (auto* k = MatchPrescaled<
          C::Type<64, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16, 32, 128>,
          P::Type<64, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16, 32, 128>>(
          control))
    return k;
  if (auto* k =
          MatchPrescaled<C::Type<32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16>,
                         P::Type<32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16>>(
              control))
    return k;
  if (auto* k =
          MatchPrescaled<C::Type<16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16>,
                         P::Type<16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 16>>(
              control))
    return k;
  return MatchPrescaled<C::Type<8, 128, 64, 1, 4, 1, D, S, 2, true, 1, 16>,
                        P::Type<8, 128, 64, 1, 4, 1, D, S, 2, true, 1, 16>>(
      control);
}

}  // namespace turbomind::gemm
