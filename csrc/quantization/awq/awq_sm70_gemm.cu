/*
 * SM70 AWQ GEMM integration using TurboMind s884h kernels.
 * Adapted from LMDeploy TurboMind (Apache-2.0).
 */

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind {
void unpack_awq_gemm(uint4_t* dst, const uint4_t* src, int rows, int cols, cudaStream_t st);
}  // namespace turbomind

namespace vllm {
namespace awq_sm70 {

namespace {

struct WorkspaceHolder {
  torch::Tensor barriers;
  torch::Tensor partials;
  torch::Tensor tensormaps;
  torch::Tensor flags;
  turbomind::gemm::Workspace workspace{};
};

struct GemmHolder {
  std::unique_ptr<turbomind::gemm::Gemm> gemm;
};

struct DenseTuneKey {
  int device;
  int m;
  int n;
  int k;
  int group_size;

  bool operator==(const DenseTuneKey& other) const {
    return device == other.device && m == other.m && n == other.n &&
           k == other.k && group_size == other.group_size;
  }
};

struct DenseTuneKeyHash {
  std::size_t operator()(const DenseTuneKey& key) const {
    std::size_t h = std::hash<int>()(key.device);
    h ^= std::hash<int>()(key.m) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.k) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.group_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct MoeTuneKey {
  int device;
  int total_tokens;
  int n;
  int k;
  int num_experts;
  int group_size;

  bool operator==(const MoeTuneKey& other) const {
    return device == other.device && total_tokens == other.total_tokens &&
           n == other.n && k == other.k &&
           num_experts == other.num_experts &&
           group_size == other.group_size;
  }
};

struct MoeTuneKeyHash {
  std::size_t operator()(const MoeTuneKey& key) const {
    std::size_t h = std::hash<int>()(key.device);
    h ^= std::hash<int>()(key.total_tokens) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.k) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.num_experts) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.group_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct Sm70F16WeightCacheKey {
  int device;
  const void* tensor_impl;
  int64_t rows;
  int64_t cols;

  bool operator==(const Sm70F16WeightCacheKey& other) const {
    return device == other.device && tensor_impl == other.tensor_impl &&
           rows == other.rows && cols == other.cols;
  }
};

struct Sm70F16WeightCacheKeyHash {
  std::size_t operator()(const Sm70F16WeightCacheKey& key) const {
    std::size_t h = std::hash<int>()(key.device);
    h ^= std::hash<const void*>()(key.tensor_impl) + 0x9e3779b9 + (h << 6) +
         (h >> 2);
    h ^= std::hash<int64_t>()(key.rows) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>()(key.cols) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct Sm70F16WeightCacheEntry {
  torch::Tensor tm_weight;
  int64_t k_ld;
};

// Per-stream workspace management to eliminate mutex contention
struct StreamWorkspaceKey {
  int device;
  cudaStream_t stream;

  bool operator==(const StreamWorkspaceKey& other) const {
    return device == other.device && stream == other.stream;
  }
};

struct StreamWorkspaceKeyHash {
  std::size_t operator()(const StreamWorkspaceKey& k) const {
    return std::hash<int>()(k.device) ^
           (std::hash<cudaStream_t>()(k.stream) << 1);
  }
};

std::mutex workspace_mutex;
std::mutex gemm_mutex;
std::mutex tune_mutex;
std::mutex sm70_f16_weight_cache_mutex;
std::unordered_map<StreamWorkspaceKey, WorkspaceHolder, StreamWorkspaceKeyHash> workspace_cache;
std::unordered_map<int, GemmHolder> gemm_cache;
std::unordered_set<DenseTuneKey, DenseTuneKeyHash> dense_tuned_shapes;
std::unordered_set<MoeTuneKey, MoeTuneKeyHash> moe_tuned_shapes;
std::unordered_set<int> imported_cache_devices;
std::unordered_map<Sm70F16WeightCacheKey,
                   Sm70F16WeightCacheEntry,
                   Sm70F16WeightCacheKeyHash>
    sm70_f16_weight_cache;

turbomind::gemm::DispatchPolicy select_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream);
turbomind::gemm::DispatchPolicy select_moe_dispatch_policy(
    int device,
    int total_tokens,
    int n,
    int k,
    int num_experts,
    int group_size,
    cudaStream_t stream);

bool tune_small_shapes_enabled() {
  const char* raw = std::getenv("VLLM_SM70_AWQ_TUNE_SMALL_SHAPES");
  return raw == nullptr || std::atoi(raw) != 0;
}

int dense_tune_max_m() {
  const char* raw = std::getenv("VLLM_SM70_AWQ_DENSE_TUNE_MAX_M");
  return raw ? std::max(std::atoi(raw), 0) : 16;
}

int moe_tune_max_tokens() {
  const char* raw = std::getenv("VLLM_SM70_AWQ_MOE_TUNE_MAX_TOKENS");
  return raw ? std::max(std::atoi(raw), 0) : 128;
}

int sm70_f16_dense_max_m() {
  const char* raw = std::getenv("VLLM_SM70_F16_DENSE_MAX_M");
  return raw ? std::max(std::atoi(raw), 0) : 64;
}

bool is_stream_capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  const auto ec = cudaStreamIsCapturing(stream, &status);
  if (ec != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return status != cudaStreamCaptureStatusNone;
}

bool has_imported_cache(int device) {
  std::lock_guard<std::mutex> lock(tune_mutex);
  return imported_cache_devices.find(device) != imported_cache_devices.end();
}

turbomind::gemm::DispatchPolicy select_dense_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream) {
  if (!tune_small_shapes_enabled() || m > dense_tune_max_m()) {
    return turbomind::gemm::DispatchPolicy::kDefault;
  }

  DenseTuneKey key{device, m, n, k, group_size};
  std::lock_guard<std::mutex> lock(tune_mutex);
  if (dense_tuned_shapes.find(key) != dense_tuned_shapes.end()) {
    return turbomind::gemm::DispatchPolicy::kReuse;
  }
  if (is_stream_capturing(stream)) {
    if (has_imported_cache(device)) {
      return turbomind::gemm::DispatchPolicy::kReuse;
    }
    return turbomind::gemm::DispatchPolicy::kDefault;
  }
  dense_tuned_shapes.insert(key);
  return turbomind::gemm::DispatchPolicy::kMeasure;
}

turbomind::gemm::DispatchPolicy select_moe_dispatch_policy(
    int device,
    int total_tokens,
    int n,
    int k,
    int num_experts,
    int group_size,
    cudaStream_t stream) {
  if (!tune_small_shapes_enabled() || total_tokens > moe_tune_max_tokens()) {
    return turbomind::gemm::DispatchPolicy::kDefault;
  }

  MoeTuneKey key{device, total_tokens, n, k, num_experts, group_size};
  std::lock_guard<std::mutex> lock(tune_mutex);
  if (moe_tuned_shapes.find(key) != moe_tuned_shapes.end()) {
    return turbomind::gemm::DispatchPolicy::kReuse;
  }
  if (is_stream_capturing(stream)) {
    if (has_imported_cache(device)) {
      return turbomind::gemm::DispatchPolicy::kReuse;
    }
    return turbomind::gemm::DispatchPolicy::kDefault;
  }
  moe_tuned_shapes.insert(key);
  return turbomind::gemm::DispatchPolicy::kMeasure;
}

WorkspaceHolder& get_workspace(int device, cudaStream_t stream) {
  thread_local int cached_device = -1;
  thread_local cudaStream_t cached_stream = nullptr;
  thread_local WorkspaceHolder* cached_holder = nullptr;
  if (cached_holder != nullptr && cached_device == device &&
      cached_stream == stream) {
    return *cached_holder;
  }

  StreamWorkspaceKey key{device, stream};

  // Fast path: check if workspace exists without lock
  {
    std::lock_guard<std::mutex> lock(workspace_mutex);
    auto it = workspace_cache.find(key);
    if (it != workspace_cache.end()) {
      cached_device = device;
      cached_stream = stream;
      cached_holder = &it->second;
      return it->second;
    }
  }

  // Slow path: create new workspace
  WorkspaceHolder holder;
  auto byte_opts = torch::TensorOptions()
                       .device(torch::Device(torch::kCUDA, device))
                       .dtype(torch::kUInt8);
  auto int_opts = torch::TensorOptions()
                      .device(torch::Device(torch::kCUDA, device))
                      .dtype(torch::kInt32);

  holder.barriers = torch::zeros(
      {(long long)turbomind::gemm::Gemm::kBarriersSize}, byte_opts);
  holder.partials = torch::zeros(
      {(long long)turbomind::gemm::Gemm::kPartialsSize}, byte_opts);
  // Keep same tensormap size as TurboMind LlamaLinear.
  holder.tensormaps = torch::empty({(long long)(8192 * 128)}, byte_opts);
  holder.flags = torch::zeros({1}, int_opts);

  holder.workspace.barriers = holder.barriers.data_ptr();
  holder.workspace.barriers_size = holder.barriers.numel();
  holder.workspace.partials = holder.partials.data_ptr();
  holder.workspace.partials_size = holder.partials.numel();
  holder.workspace.tensormaps = holder.tensormaps.data_ptr();
  holder.workspace.tensormaps_size = holder.tensormaps.numel();
  holder.workspace.flags = holder.flags.data_ptr<int>();

  std::lock_guard<std::mutex> lock(workspace_mutex);
  auto [insert_it, _] = workspace_cache.emplace(key, std::move(holder));
  cached_device = device;
  cached_stream = stream;
  cached_holder = &insert_it->second;
  return insert_it->second;
}

turbomind::gemm::Gemm& get_gemm(int device) {
  thread_local int cached_device = -1;
  thread_local turbomind::gemm::Gemm* cached_gemm = nullptr;
  if (cached_gemm != nullptr && cached_device == device) {
    return *cached_gemm;
  }

  std::lock_guard<std::mutex> lock(gemm_mutex);
  auto it = gemm_cache.find(device);
  if (it != gemm_cache.end()) {
    cached_device = device;
    cached_gemm = it->second.gemm.get();
    return *it->second.gemm;
  }
  GemmHolder holder;
  holder.gemm = std::make_unique<turbomind::gemm::Gemm>();
  auto [insert_it, _] = gemm_cache.emplace(device, std::move(holder));
  cached_device = device;
  cached_gemm = insert_it->second.gemm.get();
  return *insert_it->second.gemm;
}

void validate_awq_inputs(const torch::Tensor& qweight,
                         const torch::Tensor& scales,
                         const torch::Tensor& qzeros) {
  TORCH_CHECK(qweight.is_cuda(), "awq_sm70_prepare: qweight must be CUDA.");
  TORCH_CHECK(scales.is_cuda(), "awq_sm70_prepare: scales must be CUDA.");
  TORCH_CHECK(qzeros.is_cuda(), "awq_sm70_prepare: qzeros must be CUDA.");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt32,
              "awq_sm70_prepare: qweight must be int32.");
  TORCH_CHECK(qzeros.scalar_type() == torch::kInt32,
              "awq_sm70_prepare: qzeros must be int32.");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16,
              "awq_sm70_prepare: scales must be float16.");
}

void validate_fp8_inputs(const torch::Tensor& qweight,
                         const torch::Tensor& scales,
                         int64_t group_size) {
  TORCH_CHECK(qweight.is_cuda(), "fp8_sm70_prepare: qweight must be CUDA.");
  TORCH_CHECK(scales.is_cuda(), "fp8_sm70_prepare: scales must be CUDA.");
  TORCH_CHECK(qweight.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "fp8_sm70_prepare: qweight must be float8_e4m3fn.");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
              "fp8_sm70_prepare: scales must be float32.");
  TORCH_CHECK(qweight.dim() == 2, "fp8_sm70_prepare: qweight must be 2D.");
  TORCH_CHECK(scales.dim() == 2, "fp8_sm70_prepare: scales must be 2D.");
  TORCH_CHECK(group_size == 128,
              "fp8_sm70_prepare: only group_size=128 is supported.");
}

void validate_f16_weight(const torch::Tensor& weight,
                         const char* op_name) {
  TORCH_CHECK(weight.is_cuda(), op_name, ": weight must be CUDA.");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat16,
              op_name, ": weight must be float16.");
  TORCH_CHECK(weight.dim() == 2, op_name, ": weight must be 2D.");
}

void validate_f16_input(const torch::Tensor& in_feats,
                        const torch::Tensor& tm_weight,
                        const torch::Tensor& out,
                        bool gated_silu,
                        const char* op_name) {
  TORCH_CHECK(in_feats.is_cuda(), op_name, ": input must be CUDA.");
  TORCH_CHECK(tm_weight.is_cuda(), op_name, ": weight must be CUDA.");
  TORCH_CHECK(out.is_cuda(), op_name, ": output must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              op_name, ": input must be float16.");
  TORCH_CHECK(tm_weight.scalar_type() == torch::kFloat16,
              op_name, ": weight must be float16.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              op_name, ": output must be float16.");
  TORCH_CHECK(in_feats.dim() == 2, op_name, ": input must be 2D.");
  TORCH_CHECK(tm_weight.dim() == 2, op_name, ": weight must be 2D.");
  TORCH_CHECK(out.dim() == 2, op_name, ": output must be 2D.");
}

void validate_f16_gate_mul_input(const torch::Tensor& out,
                                 const torch::Tensor& in_feats,
                                 const torch::Tensor& gate_weight) {
  TORCH_CHECK(out.is_cuda(), "sm70_f16_gate_mul_out: out must be CUDA.");
  TORCH_CHECK(in_feats.is_cuda(),
              "sm70_f16_gate_mul_out: input must be CUDA.");
  TORCH_CHECK(gate_weight.is_cuda(),
              "sm70_f16_gate_mul_out: gate weight must be CUDA.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              "sm70_f16_gate_mul_out: out must be float16.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "sm70_f16_gate_mul_out: input must be float16.");
  TORCH_CHECK(gate_weight.scalar_type() == torch::kFloat16,
              "sm70_f16_gate_mul_out: gate weight must be float16.");
  TORCH_CHECK(out.dim() == 2, "sm70_f16_gate_mul_out: out must be 2D.");
  TORCH_CHECK(in_feats.dim() == 2,
              "sm70_f16_gate_mul_out: input must be 2D.");
  TORCH_CHECK(gate_weight.dim() == 2,
              "sm70_f16_gate_mul_out: gate weight must be 2D.");
  TORCH_CHECK(gate_weight.size(0) == 1,
              "sm70_f16_gate_mul_out: gate weight must have one output row.");
  TORCH_CHECK(out.size(0) == in_feats.size(0),
              "sm70_f16_gate_mul_out: out/input batch mismatch.");
  TORCH_CHECK(out.stride(1) == 1,
              "sm70_f16_gate_mul_out: out must be row-major contiguous.");
  TORCH_CHECK(in_feats.stride(1) == 1,
              "sm70_f16_gate_mul_out: input must be row-major contiguous.");
  TORCH_CHECK(gate_weight.stride(1) == 1,
              "sm70_f16_gate_mul_out: gate weight must be contiguous.");
  TORCH_CHECK(in_feats.size(1) == gate_weight.size(1),
              "sm70_f16_gate_mul_out: input/gate K mismatch.");
}

Sm70F16WeightCacheKey make_sm70_f16_weight_cache_key(
    const torch::Tensor& weight) {
  return Sm70F16WeightCacheKey{
      weight.get_device(),
      static_cast<const void*>(weight.unsafeGetTensorImpl()),
      weight.size(0),
      weight.size(1),
  };
}

Sm70F16WeightCacheEntry prepare_sm70_f16_weight(torch::Tensor weight,
                                                cudaStream_t stream) {
  const int64_t n = weight.size(0);
  const int64_t k = weight.size(1);

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kHalf, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  TORCH_CHECK(conv_w, "sm70_f16_prepare: no compatible TurboMind converter.");

  const auto order_w = conv_w->order;
  const bool is_A_w =
      turbomind::gemm::get_operand_tag(conv_w->pack) ==
      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout k_desc = w_desc;
  k_desc.type = turbomind::kHalf;
  k_desc.pack = conv_w->pack;
  if (is_A_w) {
    k_desc = turbomind::gemm::transpose(k_desc);
  }

  auto tm_weight = torch::empty_like(weight);
  TORCH_CHECK(
      conv_w->Convert(weight.data_ptr(),
                      w_desc,
                      tm_weight.data_ptr(),
                      k_desc,
                      stream) == 0,
      "sm70_f16_prepare: weight conversion failed.");

  return {std::move(tm_weight), static_cast<int64_t>(k_desc.ld)};
}

Sm70F16WeightCacheEntry get_sm70_f16_cached_weight(torch::Tensor weight,
                                                   cudaStream_t stream) {
  weight = weight.contiguous();
  const auto key = make_sm70_f16_weight_cache_key(weight);

  {
    std::lock_guard<std::mutex> lock(sm70_f16_weight_cache_mutex);
    auto it = sm70_f16_weight_cache.find(key);
    if (it != sm70_f16_weight_cache.end()) {
      return it->second;
    }
  }

  TORCH_CHECK(!is_stream_capturing(stream),
              "sm70_f16_prepare: cache miss during CUDA graph capture.");

  auto entry = prepare_sm70_f16_weight(weight, stream);

  std::lock_guard<std::mutex> lock(sm70_f16_weight_cache_mutex);
  auto [it, _] = sm70_f16_weight_cache.emplace(key, entry);
  return it->second;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <int THREADS>
__global__ void sm70_f16_gate_mul_kernel(half* out,
                                         const half* in_feats,
                                         const half* gate_weight,
                                         int out_rows,
                                         int out_cols,
                                         int k,
                                         int64_t out_row_stride,
                                         int64_t in_row_stride) {
  const int row = blockIdx.x;
  if (row >= out_rows) {
    return;
  }

  const int tid = threadIdx.x;
  const half* x_row = in_feats + row * in_row_stride;

  float dot = 0.f;
  const int k2 = k >> 1;
  const half2* x2 = reinterpret_cast<const half2*>(x_row);
  const half2* w2 = reinterpret_cast<const half2*>(gate_weight);

  for (int idx = tid; idx < k2; idx += THREADS) {
    const half2 a = x2[idx];
    const half2 b = w2[idx];
    const half2 prod = __hmul2(a, b);
    dot += __half2float(__low2half(prod));
    dot += __half2float(__high2half(prod));
  }
  if ((k & 1) && tid == 0) {
    dot += __half2float(x_row[k - 1]) * __half2float(gate_weight[k - 1]);
  }

  dot = warp_reduce_sum(dot);

  __shared__ float block_sum[THREADS / 32];
  if ((tid & 31) == 0) {
    block_sum[tid >> 5] = dot;
  }
  __syncthreads();

  if (tid < 32) {
    float sum = tid < (THREADS / 32) ? block_sum[tid] : 0.f;
    sum = warp_reduce_sum(sum);
    if (tid == 0) {
      block_sum[0] = 1.f / (1.f + __expf(-sum));
    }
  }
  __syncthreads();

  const float gate = block_sum[0];
  const half2 gate2 = __float2half2_rn(gate);
  half* out_row = out + row * out_row_stride;
  half2* out_row2 = reinterpret_cast<half2*>(out_row);
  const int out_cols2 = out_cols >> 1;
  for (int idx = tid; idx < out_cols2; idx += THREADS) {
    out_row2[idx] = __hmul2(out_row2[idx], gate2);
  }
  if ((out_cols & 1) && tid == 0) {
    out_row[out_cols - 1] =
        __float2half(__half2float(out_row[out_cols - 1]) * gate);
  }
}

}  // namespace

torch::Tensor interleave_gated_silu_cols(torch::Tensor tensor) {
  const int64_t n = tensor.size(-1);
  TORCH_CHECK((n % 2) == 0,
              "awq_sm70_prepare: gated_silu interleave requires even columns.");
  const int64_t half = n / 2;
  auto first = tensor.slice(-1, 0, half);
  auto second = tensor.slice(-1, half, n);
  return torch::stack({first, second}, -1).reshape(tensor.sizes());
}

std::vector<torch::Tensor> awq_sm70_prepare(torch::Tensor qweight,
                                            torch::Tensor scales,
                                            torch::Tensor qzeros,
                                            int64_t group_size,
                                            bool interleave_gated_silu) {
  validate_awq_inputs(qweight, scales, qzeros);

  qweight = qweight.contiguous();
  scales = scales.contiguous();
  qzeros = qzeros.contiguous();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t k = qweight.size(0);
  const int64_t n = qweight.size(1) * 8;
  const int64_t num_groups = scales.size(0);

  TORCH_CHECK(scales.size(1) == n,
              "awq_sm70_prepare: scales shape mismatch.");
  TORCH_CHECK(qzeros.size(0) == num_groups,
              "awq_sm70_prepare: qzeros group mismatch.");
  TORCH_CHECK(qzeros.size(1) * 8 == n,
              "awq_sm70_prepare: qzeros shape mismatch.");
  TORCH_CHECK(k % 8 == 0 && n % 8 == 0,
              "awq_sm70_prepare: K and N must be multiples of 8.");
  TORCH_CHECK(k % num_groups == 0,
              "awq_sm70_prepare: input dim must be divisible by groups.");

  if (group_size <= 0) {
    group_size = k / num_groups;
  }
  TORCH_CHECK(k / num_groups == group_size,
              "awq_sm70_prepare: group_size mismatch with scales.");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "awq_sm70_prepare: SM70 AWQ supports group_size=32/64/128, got ",
              group_size, ".");

  const bool grouped = (group_size != k);
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "awq_sm70_prepare: no compatible TurboMind converters.");

  const auto order_w = conv_w->order;
  const bool is_A_w =
      turbomind::gemm::get_operand_tag(conv_w->pack) ==
      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  auto packed_weight = torch::empty_like(qweight);
  turbomind::unpack_awq_gemm(
      reinterpret_cast<turbomind::uint4_t*>(packed_weight.data_ptr<int>()),
      reinterpret_cast<const turbomind::uint4_t*>(qweight.data_ptr<int>()),
      static_cast<int>(k), static_cast<int>(n), stream);

  auto u16_opts = torch::TensorOptions()
                      .device(qweight.device())
                      .dtype(torch::kInt16);
  auto tmp_u16 = torch::empty({k, n}, u16_opts);
  turbomind::extend_to_u16(
      reinterpret_cast<uint16_t*>(tmp_u16.data_ptr<int16_t>()),
      reinterpret_cast<const turbomind::uint4_t*>(
          packed_weight.data_ptr<int>()),
      tmp_u16.numel(), stream);
  if (interleave_gated_silu) {
    tmp_u16 = interleave_gated_silu_cols(tmp_u16);
  }

  torch::Tensor tmp_u16_conv = tmp_u16;
  if (order_w == turbomind::gemm::kRowMajor) {
    tmp_u16_conv = tmp_u16.transpose(0, 1).contiguous();
  }

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };

  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout k_desc = w_desc;
  k_desc.type = turbomind::data_type_v<turbomind::uint4_t>;
  k_desc.pack = conv_w->pack;
  if (is_A_w) {
    k_desc = turbomind::gemm::transpose(k_desc);
  }

  auto tm_weight = torch::empty_like(qweight);
  TORCH_CHECK(
      conv_w->Convert(tmp_u16_conv.data_ptr(),
                      w_desc,
                      tm_weight.data_ptr(),
                      k_desc,
                      stream) == 0,
      "awq_sm70_prepare: weight conversion failed.");

  // Unpack AWQ zeros using PyTorch tensor ops (matches lmdeploy's Python
  // approach).  The C++ unpack_awq_gemm() requires rows%8==0 which fails
  // when num_groups < 8 (e.g. Qwen3-30B-A3B w2: K=768, num_groups=6).
  const int awq_order[] = {0, 4, 1, 5, 2, 6, 3, 7};
  std::vector<torch::Tensor> zslices;
  auto zz = qzeros;
  for (int i = 0; i < 8; ++i) {
    zslices.push_back((zz & 0xF).to(torch::kUInt8));
    zz = zz.__rshift__(4);
  }
  std::vector<torch::Tensor> zordered;
  for (int i = 0; i < 8; ++i) {
    zordered.push_back(zslices[awq_order[i]]);
  }
  auto zeros_half = torch::stack(zordered, -1)
                        .reshape({num_groups, n})
                        .to(torch::kFloat16);
  if (interleave_gated_silu) {
    scales = interleave_gated_silu_cols(scales);
    zeros_half = interleave_gated_silu_cols(zeros_half);
  }

  auto fused = torch::empty({num_groups, n * 2},
                            torch::TensorOptions()
                                .device(scales.device())
                                .dtype(torch::kFloat16));
  turbomind::fuse_scales_and_zeros(
      reinterpret_cast<half*>(fused.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(zeros_half.data_ptr<at::Half>()),
      scales.numel(), stream);

  const auto order_s = conv_s->order;
  const bool is_A_s =
      turbomind::gemm::get_operand_tag(conv_s->pack) ==
      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint32,
      order_s,
      static_cast<int>(n),
      static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }

  turbomind::gemm::MatrixLayout q_desc = s_desc;
  q_desc.pack = conv_s->pack;
  if (is_A_s) {
    q_desc = turbomind::gemm::transpose(q_desc);
  }

  auto tm_scales = torch::empty(
      {num_groups, n},
      torch::TensorOptions()
          .device(scales.device())
          .dtype(torch::kInt32));
  TORCH_CHECK(
      conv_s->Convert(fused.data_ptr(),
                      s_desc,
                      tm_scales.data_ptr(),
                      q_desc,
                      stream) == 0,
      "awq_sm70_prepare: scale conversion failed.");

  auto meta = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64));
  meta.index_put_({0}, k_desc.ld);
  meta.index_put_({1}, q_desc.ld);

  return {tm_weight, tm_scales, meta};
}

std::vector<torch::Tensor> fp8_sm70_prepare(torch::Tensor qweight,
                                            torch::Tensor scales,
                                            int64_t group_size) {
  validate_fp8_inputs(qweight, scales, group_size);

  qweight = qweight.contiguous();
  scales = scales.contiguous();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t n = qweight.size(0);
  const int64_t k = qweight.size(1);
  TORCH_CHECK(k % 8 == 0 && n % 8 == 0,
              "fp8_sm70_prepare: K and N must be multiples of 8.");
  TORCH_CHECK(scales.size(0) == (n + group_size - 1) / group_size,
              "fp8_sm70_prepare: output scale block mismatch.");
  TORCH_CHECK(scales.size(1) == (k + group_size - 1) / group_size,
              "fp8_sm70_prepare: input scale block mismatch.");

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "fp8_sm70_prepare: no compatible TurboMind converters.");

  const auto order_w = conv_w->order;
  const bool is_A_w =
      turbomind::gemm::get_operand_tag(conv_w->pack) ==
      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  auto qweight_kn = qweight.transpose(0, 1).contiguous();
  auto i16_opts = torch::TensorOptions()
                      .device(qweight.device())
                      .dtype(torch::kInt16);
  auto tmp_u16 = torch::empty({k, n}, i16_opts);
  turbomind::extend_to_u16(
      reinterpret_cast<uint16_t*>(tmp_u16.data_ptr<int16_t>()),
      reinterpret_cast<const uint8_t*>(qweight_kn.data_ptr()),
      tmp_u16.numel(), stream);

  torch::Tensor tmp_u16_conv = tmp_u16;
  if (order_w == turbomind::gemm::kRowMajor) {
    tmp_u16_conv = tmp_u16.transpose(0, 1).contiguous();
  }

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout k_desc = w_desc;
  k_desc.type = turbomind::data_type_v<uint8_t>;
  k_desc.pack = conv_w->pack;
  if (is_A_w) {
    k_desc = turbomind::gemm::transpose(k_desc);
  }

  auto tm_weight = torch::empty(
      {k, n},
      torch::TensorOptions().device(qweight.device()).dtype(torch::kUInt8));
  TORCH_CHECK(
      conv_w->Convert(tmp_u16_conv.data_ptr(),
                      w_desc,
                      tm_weight.data_ptr(),
                      k_desc,
                      stream) == 0,
      "fp8_sm70_prepare: weight conversion failed.");

  const auto order_s = conv_s->order;
  const bool is_A_s =
      turbomind::gemm::get_operand_tag(conv_s->pack) ==
      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;
  const int64_t num_groups = (k + group_size - 1) / group_size;

  auto group_scales = scales.transpose(0, 1)
                          .contiguous()
                          .to(torch::kFloat16)
                          .repeat_interleave(group_size, 1)
                          .slice(1, 0, n)
                          .contiguous();

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint16,
      order_s,
      static_cast<int>(n),
      static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }

  turbomind::gemm::MatrixLayout q_desc = s_desc;
  q_desc.pack = conv_s->pack;
  if (is_A_s) {
    q_desc = turbomind::gemm::transpose(q_desc);
  }

  auto tm_scales = torch::empty_like(group_scales);
  TORCH_CHECK(
      conv_s->Convert(group_scales.data_ptr(),
                      s_desc,
                      tm_scales.data_ptr(),
                      q_desc,
                      stream) == 0,
      "fp8_sm70_prepare: scale conversion failed.");

  auto meta = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64));
  meta.index_put_({0}, k_desc.ld);
  meta.index_put_({1}, q_desc.ld);

  return {tm_weight, tm_scales, meta};
}

std::vector<torch::Tensor> sm70_f16_prepare(torch::Tensor weight) {
  validate_f16_weight(weight, "sm70_f16_prepare");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(weight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto entry = get_sm70_f16_cached_weight(weight, stream);

  auto meta = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64));
  meta.index_put_({0}, entry.k_ld);
  return {entry.tm_weight, meta};
}

void awq_gemm_sm70_out(torch::Tensor out,
                       torch::Tensor in_feats,
                       torch::Tensor tm_weight,
                       torch::Tensor tm_scales,
                       int64_t group_size,
                       int64_t k_ld,
                       int64_t q_ld,
                       bool gated_silu) {
  TORCH_CHECK(in_feats.is_cuda(), "awq_gemm_sm70: input must be CUDA.");
  TORCH_CHECK(tm_weight.is_cuda(), "awq_gemm_sm70: weight must be CUDA.");
  TORCH_CHECK(tm_scales.is_cuda(), "awq_gemm_sm70: scales must be CUDA.");
  TORCH_CHECK(out.is_cuda(), "awq_gemm_sm70: output must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "awq_gemm_sm70: input must be float16.");
  TORCH_CHECK(tm_weight.scalar_type() == torch::kInt32,
              "awq_gemm_sm70: weight must be int32.");
  TORCH_CHECK(tm_scales.scalar_type() == torch::kInt32,
              "awq_gemm_sm70: scales must be int32.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              "awq_gemm_sm70: output must be float16.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const int device = in_feats.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t k = in_feats.size(1);
  const int64_t n = tm_weight.size(1) * 8;
  TORCH_CHECK(tm_weight.size(0) == k,
              "awq_gemm_sm70: weight shape mismatch.");
  TORCH_CHECK(k % group_size == 0,
              "awq_gemm_sm70: input dim must be divisible by group size.");
  TORCH_CHECK(tm_scales.size(0) == k / group_size,
              "awq_gemm_sm70: scale groups mismatch.");
  TORCH_CHECK(tm_scales.size(1) == n,
              "awq_gemm_sm70: scale shape mismatch.");
  TORCH_CHECK(out.size(0) == m,
              "awq_gemm_sm70: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "awq_gemm_sm70: output must be row-major contiguous.");
  if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "awq_gemm_sm70: gated_silu requires even output dim.");
    TORCH_CHECK(out.size(1) == n / 2,
                "awq_gemm_sm70: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n,
                "awq_gemm_sm70: output cols must match n.");
  }

  const bool grouped = (group_size != k);
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "awq_gemm_sm70: no compatible TurboMind converters.");

  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(m),
      static_cast<int>(k),
      static_cast<int>(k),
  };
  turbomind::gemm::MatrixLayout desc_U{};

  const auto order_w = conv_w->order;
  const bool is_A_w =
      turbomind::gemm::get_operand_tag(conv_w->pack) ==
      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.type = turbomind::data_type_v<turbomind::uint4_t>;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = static_cast<int>(k_ld);

  const auto order_s = conv_s->order;
  const bool is_A_s =
      turbomind::gemm::get_operand_tag(conv_s->pack) ==
      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;

  const int64_t num_groups_raw = k / group_size;

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint32,
      order_s,
      static_cast<int>(n),
      static_cast<int>(num_groups_raw),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_V = s_desc;
  desc_V.pack = conv_s->pack;
  if (is_A_s) {
    desc_V = turbomind::gemm::transpose(desc_V);
  }
  desc_V.ld = static_cast<int>(q_ld);

  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(out.stride(0)),
  };

  turbomind::gemm::Operation op{};
  op.dispatch = select_dense_dispatch_policy(
      device, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
      static_cast<int>(group_size), stream);
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kK, static_cast<int>(group_size)};
  op.batch_dim = 0;

  auto& workspace_holder = get_workspace(device, stream);
  auto& gemm = get_gemm(device);

  const int ec = gemm.Run(op,
                          1.f,
                          in_feats.data_ptr(),
                          desc_A,
                          nullptr,
                          desc_U,
                          tm_weight.data_ptr(),
                          desc_B,
                          tm_scales.data_ptr(),
                          desc_V,
                          0.f,
                          out.data_ptr(),
                          desc_D,
                          out.data_ptr(),
                          desc_D,
                          workspace_holder.workspace,
                          stream);
  TORCH_CHECK(ec == 0, "awq_gemm_sm70: TurboMind GEMM failed.");
}

void fp8_gemm_sm70_out(torch::Tensor out,
                       torch::Tensor in_feats,
                       torch::Tensor tm_weight,
                       torch::Tensor tm_scales,
                       int64_t group_size,
                       int64_t k_ld,
                       int64_t q_ld) {
  TORCH_CHECK(in_feats.is_cuda(), "fp8_gemm_sm70: input must be CUDA.");
  TORCH_CHECK(tm_weight.is_cuda(), "fp8_gemm_sm70: weight must be CUDA.");
  TORCH_CHECK(tm_scales.is_cuda(), "fp8_gemm_sm70: scales must be CUDA.");
  TORCH_CHECK(out.is_cuda(), "fp8_gemm_sm70: output must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "fp8_gemm_sm70: input must be float16.");
  TORCH_CHECK(tm_weight.scalar_type() == torch::kUInt8,
              "fp8_gemm_sm70: weight must be uint8.");
  TORCH_CHECK(tm_scales.scalar_type() == torch::kFloat16,
              "fp8_gemm_sm70: scales must be float16.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              "fp8_gemm_sm70: output must be float16.");
  TORCH_CHECK(group_size == 128,
              "fp8_gemm_sm70: only group_size=128 is supported.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const int device = in_feats.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t k = in_feats.size(1);
  const int64_t n = tm_weight.size(1);
  TORCH_CHECK(tm_weight.size(0) == k,
              "fp8_gemm_sm70: weight shape mismatch.");
  TORCH_CHECK(tm_scales.size(0) == (k + group_size - 1) / group_size,
              "fp8_gemm_sm70: scale groups mismatch.");
  TORCH_CHECK(tm_scales.size(1) == n,
              "fp8_gemm_sm70: scale shape mismatch.");
  TORCH_CHECK(out.size(0) == m,
              "fp8_gemm_sm70: output rows must match input rows.");
  TORCH_CHECK(out.size(1) == n,
              "fp8_gemm_sm70: output cols must match weight output dim.");
  TORCH_CHECK(out.stride(1) == 1,
              "fp8_gemm_sm70: output must be row-major contiguous.");

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "fp8_gemm_sm70: no compatible TurboMind converters.");

  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(m),
      static_cast<int>(k),
      static_cast<int>(in_feats.stride(0)),
  };
  turbomind::gemm::MatrixLayout desc_U{};

  const auto order_w = conv_w->order;
  const bool is_A_w =
      turbomind::gemm::get_operand_tag(conv_w->pack) ==
      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;
  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }
  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.type = turbomind::kFloat8_e4m3;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = static_cast<int>(k_ld);

  const auto order_s = conv_s->order;
  const bool is_A_s =
      turbomind::gemm::get_operand_tag(conv_s->pack) ==
      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;
  const int64_t num_groups = (k + group_size - 1) / group_size;
  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint16,
      order_s,
      static_cast<int>(n),
      static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }
  turbomind::gemm::MatrixLayout desc_V = s_desc;
  desc_V.pack = conv_s->pack;
  if (is_A_s) {
    desc_V = turbomind::gemm::transpose(desc_V);
  }
  desc_V.ld = static_cast<int>(q_ld);

  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(out.stride(0)),
  };

  turbomind::gemm::Operation op{};
  op.dispatch = select_dense_dispatch_policy(
      device, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
      static_cast<int>(group_size), stream);
  op.epilogue = turbomind::gemm::Epilogue::kNone;
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kK, static_cast<int>(group_size)};
  op.batch_dim = 0;

  auto& workspace_holder = get_workspace(device, stream);
  auto& gemm = get_gemm(device);

  const int ec = gemm.Run(op,
                          1.f,
                          in_feats.data_ptr(),
                          desc_A,
                          nullptr,
                          desc_U,
                          tm_weight.data_ptr(),
                          desc_B,
                          tm_scales.data_ptr(),
                          desc_V,
                          0.f,
                          out.data_ptr(),
                          desc_D,
                          out.data_ptr(),
                          desc_D,
                          workspace_holder.workspace,
                          stream);
  TORCH_CHECK(ec == 0, "fp8_gemm_sm70: TurboMind GEMM failed.");
}

void fp8_gemm_sm70_out_auto(torch::Tensor out,
                            torch::Tensor in_feats,
                            torch::Tensor tm_weight,
                            torch::Tensor tm_scales) {
  constexpr int64_t group_size = 128;
  const int64_t k = in_feats.size(1);
  const int64_t n = tm_weight.size(1);

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kFloat8_e4m3, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "fp8_gemm_sm70: no compatible TurboMind converters.");

  const auto order_w = conv_w->order;
  const bool is_A_w =
      turbomind::gemm::get_operand_tag(conv_w->pack) ==
      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;
  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }
  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.type = turbomind::kFloat8_e4m3;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }

  const auto order_s = conv_s->order;
  const bool is_A_s =
      turbomind::gemm::get_operand_tag(conv_s->pack) ==
      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;
  const int64_t num_groups = (k + group_size - 1) / group_size;
  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint16,
      order_s,
      static_cast<int>(n),
      static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }
  turbomind::gemm::MatrixLayout desc_V = s_desc;
  desc_V.pack = conv_s->pack;
  if (is_A_s) {
    desc_V = turbomind::gemm::transpose(desc_V);
  }

  fp8_gemm_sm70_out(out, in_feats, tm_weight, tm_scales, group_size,
                    desc_B.ld, desc_V.ld);
}

void fp8_gemm_sm70_out_meta(torch::Tensor out,
                            torch::Tensor in_feats,
                            torch::Tensor tm_weight,
                            torch::Tensor tm_scales,
                            torch::Tensor meta) {
  TORCH_CHECK(meta.scalar_type() == torch::kInt64,
              "fp8_gemm_sm70: meta must be int64.");
  TORCH_CHECK(meta.numel() >= 2, "fp8_gemm_sm70: meta must have two values.");
  auto meta_cpu = meta.device().is_cpu() ? meta.contiguous()
                                        : meta.to(torch::kCPU).contiguous();
  const int64_t* meta_ptr = meta_cpu.data_ptr<int64_t>();
  fp8_gemm_sm70_out(out, in_feats, tm_weight, tm_scales, 128, meta_ptr[0],
                    meta_ptr[1]);
}

void sm70_f16_gemm_out(torch::Tensor out,
                       torch::Tensor in_feats,
                       torch::Tensor tm_weight,
                       int64_t k_ld,
                       bool gated_silu) {
  validate_f16_input(in_feats, tm_weight, out, gated_silu, "sm70_f16_gemm");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const int device = in_feats.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t k = in_feats.size(1);
  const int64_t n = tm_weight.size(0);

  TORCH_CHECK(tm_weight.size(1) == k,
              "sm70_f16_gemm: weight shape mismatch.");
  TORCH_CHECK(out.size(0) == m,
              "sm70_f16_gemm: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "sm70_f16_gemm: output must be row-major contiguous.");
  if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "sm70_f16_gemm: gated_silu requires even output dim.");
    TORCH_CHECK(out.size(1) == n / 2,
                "sm70_f16_gemm: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n,
                "sm70_f16_gemm: output cols must match n.");
  }

  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(m),
      static_cast<int>(k),
      static_cast<int>(in_feats.stride(0)),
  };
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kHalf, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  TORCH_CHECK(conv_w, "sm70_f16_gemm: no compatible TurboMind converter.");

  const auto order_w = conv_w->order;
  const bool is_A_w =
      turbomind::gemm::get_operand_tag(conv_w->pack) ==
      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = static_cast<int>(k_ld);
  turbomind::gemm::MatrixLayout desc_U{};
  turbomind::gemm::MatrixLayout desc_V{};
  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(out.stride(0)),
  };

  turbomind::gemm::Operation op{};
  op.dispatch = select_dense_dispatch_policy(
      device, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
      0, stream);
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kNone, 0};
  op.batch_dim = 0;

  auto& workspace_holder = get_workspace(device, stream);
  auto& gemm = get_gemm(device);

  const int ec = gemm.Run(op,
                          1.f,
                          in_feats.data_ptr(),
                          desc_A,
                          nullptr,
                          desc_U,
                          tm_weight.data_ptr(),
                          desc_B,
                          nullptr,
                          desc_V,
                          0.f,
                          out.data_ptr(),
                          desc_D,
                          out.data_ptr(),
                          desc_D,
                          workspace_holder.workspace,
                          stream);
  TORCH_CHECK(ec == 0, "sm70_f16_gemm: TurboMind GEMM failed.");
}

void sm70_f16_gate_mul_out(torch::Tensor out,
                           torch::Tensor in_feats,
                           torch::Tensor gate_weight) {
  validate_f16_gate_mul_input(out, in_feats, gate_weight);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t k = in_feats.size(1);
  const int64_t n = out.size(1);
  if (m == 0 || n == 0) {
    return;
  }

  constexpr int kThreads = 256;
  sm70_f16_gate_mul_kernel<kThreads>
      <<<static_cast<unsigned int>(m), kThreads, 0, stream>>>(
          reinterpret_cast<half*>(out.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(in_feats.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(gate_weight.data_ptr<at::Half>()),
          static_cast<int>(m),
          static_cast<int>(n),
          static_cast<int>(k),
          out.stride(0),
          in_feats.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor awq_gemm_sm70(torch::Tensor in_feats,
                            torch::Tensor tm_weight,
                            torch::Tensor tm_scales,
                            int64_t group_size,
                            int64_t k_ld,
                            int64_t q_ld) {
  const int64_t n = tm_weight.size(1) * 8;
  auto out = torch::empty(
      {in_feats.size(0), n},
      torch::TensorOptions().dtype(in_feats.dtype()).device(in_feats.device()));
  awq_gemm_sm70_out(out, in_feats, tm_weight, tm_scales, group_size, k_ld,
                    q_ld, false);
  return out;
}

torch::Tensor sm70_f16_gemm(torch::Tensor in_feats,
                            torch::Tensor weight) {
  validate_f16_weight(weight, "sm70_f16_gemm");
  TORCH_CHECK(in_feats.is_cuda(), "sm70_f16_gemm: input must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "sm70_f16_gemm: input must be float16.");
  TORCH_CHECK(in_feats.dim() == 2, "sm70_f16_gemm: input must be 2D.");
  TORCH_CHECK(in_feats.size(1) == weight.size(1),
              "sm70_f16_gemm: input/weight K mismatch.");
  if (in_feats.size(0) > sm70_f16_dense_max_m()) {
    return torch::mm(in_feats, weight.transpose(0, 1));
  }
  const at::cuda::OptionalCUDAGuard device_guard(device_of(weight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto entry = get_sm70_f16_cached_weight(weight, stream);
  const int64_t n = weight.size(0);
  auto out = torch::empty(
      {in_feats.size(0), n},
      torch::TensorOptions().dtype(in_feats.dtype()).device(in_feats.device()));
  sm70_f16_gemm_out(out, in_feats, entry.tm_weight, entry.k_ld, false);
  return out;
}

turbomind::gemm::DispatchPolicy awq_select_moe_dispatch_policy(
    int device,
    int total_tokens,
    int n,
    int k,
    int num_experts,
    int group_size,
    cudaStream_t stream) {
  return select_moe_dispatch_policy(
      device, total_tokens, n, k, num_experts, group_size, stream);
}

}  // namespace awq_sm70
}  // namespace vllm

std::vector<torch::Tensor> awq_sm70_prepare(torch::Tensor _kernel,
                                            torch::Tensor _scaling_factors,
                                            torch::Tensor _zeros,
                                            int64_t group_size,
                                            bool interleave_gated_silu) {
  return vllm::awq_sm70::awq_sm70_prepare(
      _kernel, _scaling_factors, _zeros, group_size,
      interleave_gated_silu);
}

std::vector<torch::Tensor> fp8_sm70_prepare(torch::Tensor _kernel,
                                            torch::Tensor _scaling_factors,
                                            int64_t group_size) {
  return vllm::awq_sm70::fp8_sm70_prepare(
      _kernel, _scaling_factors, group_size);
}

std::vector<torch::Tensor> sm70_f16_prepare(torch::Tensor _kernel) {
  return vllm::awq_sm70::sm70_f16_prepare(_kernel);
}

torch::Tensor awq_gemm_sm70(torch::Tensor _in_feats,
                            torch::Tensor _kernel,
                            torch::Tensor _scaling_factors,
                            int64_t group_size,
                            int64_t k_ld,
                            int64_t q_ld) {
  return vllm::awq_sm70::awq_gemm_sm70(
      _in_feats, _kernel, _scaling_factors, group_size, k_ld, q_ld);
}

torch::Tensor sm70_f16_gemm(torch::Tensor _in_feats,
                            torch::Tensor _kernel) {
  return vllm::awq_sm70::sm70_f16_gemm(_in_feats, _kernel);
}

void awq_gemm_sm70_out(torch::Tensor out,
                       torch::Tensor _in_feats,
                       torch::Tensor _kernel,
                       torch::Tensor _scaling_factors,
                       int64_t group_size,
                       int64_t k_ld,
                       int64_t q_ld,
                       bool gated_silu) {
  vllm::awq_sm70::awq_gemm_sm70_out(out, _in_feats, _kernel,
                                    _scaling_factors, group_size, k_ld, q_ld,
                                    gated_silu);
}

void fp8_gemm_sm70_out(torch::Tensor out,
                       torch::Tensor _in_feats,
                       torch::Tensor _kernel,
                       torch::Tensor _scaling_factors,
                       int64_t group_size,
                       int64_t k_ld,
                       int64_t q_ld) {
  vllm::awq_sm70::fp8_gemm_sm70_out(out, _in_feats, _kernel,
                                    _scaling_factors, group_size, k_ld, q_ld);
}

void fp8_gemm_sm70_out_auto(torch::Tensor out,
                            torch::Tensor _in_feats,
                            torch::Tensor _kernel,
                            torch::Tensor _scaling_factors) {
  vllm::awq_sm70::fp8_gemm_sm70_out_auto(out, _in_feats, _kernel,
                                         _scaling_factors);
}

void fp8_gemm_sm70_out_meta(torch::Tensor out,
                            torch::Tensor _in_feats,
                            torch::Tensor _kernel,
                            torch::Tensor _scaling_factors,
                            torch::Tensor _meta) {
  vllm::awq_sm70::fp8_gemm_sm70_out_meta(out, _in_feats, _kernel,
                                         _scaling_factors, _meta);
}

void sm70_f16_gemm_out(torch::Tensor out,
                       torch::Tensor _in_feats,
                       torch::Tensor _kernel,
                       int64_t k_ld,
                       bool gated_silu) {
  vllm::awq_sm70::sm70_f16_gemm_out(out, _in_feats, _kernel, k_ld, gated_silu);
}

void sm70_f16_gate_mul_out(torch::Tensor out,
                           torch::Tensor _in_feats,
                           torch::Tensor _gate_weight) {
  vllm::awq_sm70::sm70_f16_gate_mul_out(out, _in_feats, _gate_weight);
}

int64_t sm70_gemm_import_cache(torch::Tensor device_hint,
                               const std::string& path) {
  TORCH_CHECK(device_hint.is_cuda(),
              "sm70_gemm_import_cache: device_hint must be CUDA.");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(device_hint));
  const int device = device_hint.get_device();

  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.good()) {
    return 0;
  }

  auto& gemm = vllm::awq_sm70::get_gemm(device);
  const int64_t imported = gemm.Import(ifs);
  if (imported > 0) {
    std::lock_guard<std::mutex> lock(vllm::awq_sm70::tune_mutex);
    vllm::awq_sm70::imported_cache_devices.insert(device);
  }
  return imported;
}

int64_t sm70_gemm_export_cache(torch::Tensor device_hint,
                               const std::string& path) {
  TORCH_CHECK(device_hint.is_cuda(),
              "sm70_gemm_export_cache: device_hint must be CUDA.");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(device_hint));
  const int device = device_hint.get_device();

  try {
    const std::filesystem::path fs_path(path);
    if (fs_path.has_parent_path()) {
      std::filesystem::create_directories(fs_path.parent_path());
    }
  } catch (const std::exception& e) {
    TORCH_CHECK(false, "sm70_gemm_export_cache: failed to create parent "
                       "directory for ",
                path, " (", e.what(), ").");
  }

  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  TORCH_CHECK(ofs.good(),
              "sm70_gemm_export_cache: failed to open ", path,
              " for writing.");

  auto& gemm = vllm::awq_sm70::get_gemm(device);
  return gemm.Export(ofs);
}

// ---------------------------------------------------------------------------
// MoE batched GEMM support
// ---------------------------------------------------------------------------

std::vector<torch::Tensor> awq_moe_build_strided_ptrs(
    torch::Tensor tm_weights,   // [E, ...]  stacked TM weights
    torch::Tensor tm_scales,    // [E, ...]  stacked TM scales
    int64_t k_ld,
    int64_t q_ld,
    int64_t num_experts) {
  TORCH_CHECK(tm_weights.is_cuda(), "awq_moe_build_strided_ptrs: weights must be CUDA.");
  TORCH_CHECK(tm_scales.is_cuda(), "awq_moe_build_strided_ptrs: scales must be CUDA.");
  TORCH_CHECK(num_experts > 0, "awq_moe_build_strided_ptrs: num_experts must be > 0.");
  TORCH_CHECK(tm_weights.size(0) == num_experts,
              "awq_moe_build_strided_ptrs: weights dim0 != num_experts.");
  TORCH_CHECK(tm_scales.size(0) == num_experts,
              "awq_moe_build_strided_ptrs: scales dim0 != num_experts.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(tm_weights));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Build {ptr, stride} pairs for each expert
  std::vector<std::pair<void*, int>> w_ptrs;
  std::vector<std::pair<void*, int>> s_ptrs;
  w_ptrs.reserve(num_experts);
  s_ptrs.reserve(num_experts);

  const int64_t w_expert_stride = tm_weights.stride(0) * tm_weights.element_size();
  const int64_t s_expert_stride = tm_scales.stride(0) * tm_scales.element_size();
  char* w_base = static_cast<char*>(tm_weights.data_ptr());
  char* s_base = static_cast<char*>(tm_scales.data_ptr());

  for (int64_t e = 0; e < num_experts; ++e) {
    w_ptrs.emplace_back(w_base + e * w_expert_stride, static_cast<int>(k_ld));
    s_ptrs.emplace_back(s_base + e * s_expert_stride, static_cast<int>(q_ld));
  }

  // MakeStridedPtrs allocates GPU memory via cudaMallocAsync
  void* w_gpu = turbomind::gemm::MakeStridedPtrs(w_ptrs, stream);
  void* s_gpu = turbomind::gemm::MakeStridedPtrs(s_ptrs, stream);

  // Wrap in torch tensors for lifetime management.
  // StridedPtr is 16 bytes (__align__(16): void* ptr + int stride + padding).
  const int64_t buf_bytes = num_experts * 16;
  auto opts = torch::TensorOptions()
                  .device(tm_weights.device())
                  .dtype(torch::kUInt8);

  // Copy into torch-managed tensors so cudaFree of the original is safe.
  auto w_tensor = torch::empty({buf_bytes}, opts);
  auto s_tensor = torch::empty({buf_bytes}, opts);
  cudaMemcpyAsync(w_tensor.data_ptr(), w_gpu, buf_bytes,
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(s_tensor.data_ptr(), s_gpu, buf_bytes,
                  cudaMemcpyDeviceToDevice, stream);
  cudaFreeAsync(w_gpu, stream);
  cudaFreeAsync(s_gpu, stream);

  return {w_tensor, s_tensor};
}

template <typename index_t>
__global__ void awq_moe_single_token_compact_prepare_kernel(
    const index_t* topk_ids,
    const uint8_t* src_w13_ptrs_w_rows,
    const uint8_t* src_w13_ptrs_s_rows,
    const uint8_t* src_w2_ptrs_w_rows,
    const uint8_t* src_w2_ptrs_s_rows,
    uint8_t* dst_w13_ptrs_w_rows,
    uint8_t* dst_w13_ptrs_s_rows,
    uint8_t* dst_w2_ptrs_w_rows,
    uint8_t* dst_w2_ptrs_s_rows,
    int* inv_permuted_idx,
    int top_k,
    int src_row_stride,
    int dst_row_stride,
    int row_bytes) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int sorted_ids[32];
  int sorted_src[32];

  for (int i = 0; i < top_k; ++i) {
    sorted_ids[i] = static_cast<int>(topk_ids[i]);
    sorted_src[i] = i;
  }

  // Stable insertion sort to match the native MoE single-token permutation.
  for (int i = 1; i < top_k; ++i) {
    const int expert_id = sorted_ids[i];
    const int src_idx = sorted_src[i];
    int j = i - 1;
    while (j >= 0 && sorted_ids[j] > expert_id) {
      sorted_ids[j + 1] = sorted_ids[j];
      sorted_src[j + 1] = sorted_src[j];
      --j;
    }
    sorted_ids[j + 1] = expert_id;
    sorted_src[j + 1] = src_idx;
  }

  for (int sorted_pos = 0; sorted_pos < top_k; ++sorted_pos) {
    const int expert_id = sorted_ids[sorted_pos];
    const int src_idx = sorted_src[sorted_pos];
    inv_permuted_idx[src_idx] = sorted_pos;

    const uint8_t* src_w13_w =
        src_w13_ptrs_w_rows + expert_id * src_row_stride;
    const uint8_t* src_w13_s =
        src_w13_ptrs_s_rows + expert_id * src_row_stride;
    const uint8_t* src_w2_w =
        src_w2_ptrs_w_rows + expert_id * src_row_stride;
    const uint8_t* src_w2_s =
        src_w2_ptrs_s_rows + expert_id * src_row_stride;

    uint8_t* dst_w13_w = dst_w13_ptrs_w_rows + sorted_pos * dst_row_stride;
    uint8_t* dst_w13_s = dst_w13_ptrs_s_rows + sorted_pos * dst_row_stride;
    uint8_t* dst_w2_w = dst_w2_ptrs_w_rows + sorted_pos * dst_row_stride;
    uint8_t* dst_w2_s = dst_w2_ptrs_s_rows + sorted_pos * dst_row_stride;

    for (int byte_idx = 0; byte_idx < row_bytes; ++byte_idx) {
      dst_w13_w[byte_idx] = src_w13_w[byte_idx];
      dst_w13_s[byte_idx] = src_w13_s[byte_idx];
      dst_w2_w[byte_idx] = src_w2_w[byte_idx];
      dst_w2_s[byte_idx] = src_w2_s[byte_idx];
    }
  }
}

void awq_moe_single_token_compact_prepare(
    torch::Tensor topk_ids,
    torch::Tensor src_w13_ptrs_w_rows,
    torch::Tensor src_w13_ptrs_s_rows,
    torch::Tensor src_w2_ptrs_w_rows,
    torch::Tensor src_w2_ptrs_s_rows,
    torch::Tensor dst_w13_ptrs_w_rows,
    torch::Tensor dst_w13_ptrs_s_rows,
    torch::Tensor dst_w2_ptrs_w_rows,
    torch::Tensor dst_w2_ptrs_s_rows,
    torch::Tensor inv_permuted_idx) {
  TORCH_CHECK(topk_ids.is_cuda(),
              "awq_moe_single_token_compact_prepare: topk_ids must be CUDA.");
  TORCH_CHECK(
      topk_ids.scalar_type() == torch::kInt32 ||
          topk_ids.scalar_type() == torch::kInt64,
      "awq_moe_single_token_compact_prepare: topk_ids must be int32/int64.");
  TORCH_CHECK(src_w13_ptrs_w_rows.is_cuda() && src_w13_ptrs_s_rows.is_cuda() &&
                  src_w2_ptrs_w_rows.is_cuda() && src_w2_ptrs_s_rows.is_cuda(),
              "awq_moe_single_token_compact_prepare: source ptr rows must be CUDA.");
  TORCH_CHECK(dst_w13_ptrs_w_rows.is_cuda() && dst_w13_ptrs_s_rows.is_cuda() &&
                  dst_w2_ptrs_w_rows.is_cuda() && dst_w2_ptrs_s_rows.is_cuda(),
              "awq_moe_single_token_compact_prepare: destination ptr rows must be CUDA.");
  TORCH_CHECK(inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32,
              "awq_moe_single_token_compact_prepare: inv_permuted_idx "
              "must be CUDA int32.");

  topk_ids = topk_ids.contiguous().view({-1});
  TORCH_CHECK(topk_ids.numel() > 0,
              "awq_moe_single_token_compact_prepare: topk_ids must be non-empty.");
  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(top_k <= 32,
              "awq_moe_single_token_compact_prepare: top_k > 32 is unsupported.");

  const int64_t row_bytes = src_w13_ptrs_w_rows.size(1);
  TORCH_CHECK(src_w13_ptrs_w_rows.scalar_type() == torch::kUInt8 &&
                  src_w13_ptrs_s_rows.scalar_type() == torch::kUInt8 &&
                  src_w2_ptrs_w_rows.scalar_type() == torch::kUInt8 &&
                  src_w2_ptrs_s_rows.scalar_type() == torch::kUInt8 &&
                  dst_w13_ptrs_w_rows.scalar_type() == torch::kUInt8 &&
                  dst_w13_ptrs_s_rows.scalar_type() == torch::kUInt8 &&
                  dst_w2_ptrs_w_rows.scalar_type() == torch::kUInt8 &&
                  dst_w2_ptrs_s_rows.scalar_type() == torch::kUInt8,
              "awq_moe_single_token_compact_prepare: ptr rows must be uint8.");
  TORCH_CHECK(src_w13_ptrs_w_rows.dim() == 2 && src_w13_ptrs_s_rows.dim() == 2 &&
                  src_w2_ptrs_w_rows.dim() == 2 && src_w2_ptrs_s_rows.dim() == 2 &&
                  dst_w13_ptrs_w_rows.dim() == 2 && dst_w13_ptrs_s_rows.dim() == 2 &&
                  dst_w2_ptrs_w_rows.dim() == 2 && dst_w2_ptrs_s_rows.dim() == 2,
              "awq_moe_single_token_compact_prepare: ptr rows must be 2D.");
  TORCH_CHECK(src_w13_ptrs_w_rows.size(1) == row_bytes &&
                  src_w13_ptrs_s_rows.size(1) == row_bytes &&
                  src_w2_ptrs_w_rows.size(1) == row_bytes &&
                  src_w2_ptrs_s_rows.size(1) == row_bytes &&
                  dst_w13_ptrs_w_rows.size(0) == top_k &&
                  dst_w13_ptrs_s_rows.size(0) == top_k &&
                  dst_w2_ptrs_w_rows.size(0) == top_k &&
                  dst_w2_ptrs_s_rows.size(0) == top_k &&
                  dst_w13_ptrs_w_rows.size(1) == row_bytes &&
                  dst_w13_ptrs_s_rows.size(1) == row_bytes &&
                  dst_w2_ptrs_w_rows.size(1) == row_bytes &&
                  dst_w2_ptrs_s_rows.size(1) == row_bytes,
              "awq_moe_single_token_compact_prepare: ptr row shapes mismatch.");
  TORCH_CHECK(inv_permuted_idx.numel() == top_k,
              "awq_moe_single_token_compact_prepare: inv_permuted_idx size mismatch.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(topk_ids));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int src_row_stride = static_cast<int>(src_w13_ptrs_w_rows.stride(0));
  const int dst_row_stride = static_cast<int>(dst_w13_ptrs_w_rows.stride(0));

  if (topk_ids.scalar_type() == torch::kInt32) {
    awq_moe_single_token_compact_prepare_kernel<int32_t>
        <<<1, 1, 0, stream>>>(
            topk_ids.data_ptr<int32_t>(),
            src_w13_ptrs_w_rows.data_ptr<uint8_t>(),
            src_w13_ptrs_s_rows.data_ptr<uint8_t>(),
            src_w2_ptrs_w_rows.data_ptr<uint8_t>(),
            src_w2_ptrs_s_rows.data_ptr<uint8_t>(),
            dst_w13_ptrs_w_rows.data_ptr<uint8_t>(),
            dst_w13_ptrs_s_rows.data_ptr<uint8_t>(),
            dst_w2_ptrs_w_rows.data_ptr<uint8_t>(),
            dst_w2_ptrs_s_rows.data_ptr<uint8_t>(),
            inv_permuted_idx.data_ptr<int32_t>(),
            top_k,
            src_row_stride,
            dst_row_stride,
            static_cast<int>(row_bytes));
  } else {
    awq_moe_single_token_compact_prepare_kernel<int64_t>
        <<<1, 1, 0, stream>>>(
            topk_ids.data_ptr<int64_t>(),
            src_w13_ptrs_w_rows.data_ptr<uint8_t>(),
            src_w13_ptrs_s_rows.data_ptr<uint8_t>(),
            src_w2_ptrs_w_rows.data_ptr<uint8_t>(),
            src_w2_ptrs_s_rows.data_ptr<uint8_t>(),
            dst_w13_ptrs_w_rows.data_ptr<uint8_t>(),
            dst_w13_ptrs_s_rows.data_ptr<uint8_t>(),
            dst_w2_ptrs_w_rows.data_ptr<uint8_t>(),
            dst_w2_ptrs_s_rows.data_ptr<uint8_t>(),
            inv_permuted_idx.data_ptr<int32_t>(),
            top_k,
            src_row_stride,
            dst_row_stride,
            static_cast<int>(row_bytes));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void awq_moe_gemm_sm70_out(
    torch::Tensor out,
    torch::Tensor sorted_input,
    torch::Tensor expert_offsets,
    torch::Tensor strided_ptrs_w,
    torch::Tensor strided_ptrs_s,
    int64_t num_experts,
    int64_t k,
    int64_t n,
    int64_t group_size,
    bool gated_silu);

template <typename index_t>
__global__ void awq_moe_single_token_prepare_kernel(
    const index_t* topk_ids,
    const __half* x,
    const uint8_t* src_w13_ptrs_w_rows,
    const uint8_t* src_w13_ptrs_s_rows,
    const uint8_t* src_w2_ptrs_w_rows,
    const uint8_t* src_w2_ptrs_s_rows,
    uint8_t* dst_w13_ptrs_w_rows,
    uint8_t* dst_w13_ptrs_s_rows,
    uint8_t* dst_w2_ptrs_w_rows,
    uint8_t* dst_w2_ptrs_s_rows,
    __half* compact_input,
    int* expert_offsets,
    int* inv_permuted_idx,
    int top_k,
    int hidden_size,
    int src_row_stride,
    int dst_row_stride,
    int row_bytes) {
  __shared__ int sorted_ids[32];
  __shared__ int sorted_src[32];

  if (threadIdx.x == 0) {
    expert_offsets[0] = 0;
    for (int i = 0; i < top_k; ++i) {
      sorted_ids[i] = static_cast<int>(topk_ids[i]);
      sorted_src[i] = i;
      expert_offsets[i + 1] = i + 1;
    }

    // Stable insertion sort to match native single-token MoE permutation.
    for (int i = 1; i < top_k; ++i) {
      const int expert_id = sorted_ids[i];
      const int src_idx = sorted_src[i];
      int j = i - 1;
      while (j >= 0 && sorted_ids[j] > expert_id) {
        sorted_ids[j + 1] = sorted_ids[j];
        sorted_src[j + 1] = sorted_src[j];
        --j;
      }
      sorted_ids[j + 1] = expert_id;
      sorted_src[j + 1] = src_idx;
    }

    for (int sorted_pos = 0; sorted_pos < top_k; ++sorted_pos) {
      const int expert_id = sorted_ids[sorted_pos];
      const int src_idx = sorted_src[sorted_pos];
      inv_permuted_idx[src_idx] = sorted_pos;

      const uint8_t* src_w13_w =
          src_w13_ptrs_w_rows + expert_id * src_row_stride;
      const uint8_t* src_w13_s =
          src_w13_ptrs_s_rows + expert_id * src_row_stride;
      const uint8_t* src_w2_w =
          src_w2_ptrs_w_rows + expert_id * src_row_stride;
      const uint8_t* src_w2_s =
          src_w2_ptrs_s_rows + expert_id * src_row_stride;

      uint8_t* dst_w13_w = dst_w13_ptrs_w_rows + sorted_pos * dst_row_stride;
      uint8_t* dst_w13_s = dst_w13_ptrs_s_rows + sorted_pos * dst_row_stride;
      uint8_t* dst_w2_w = dst_w2_ptrs_w_rows + sorted_pos * dst_row_stride;
      uint8_t* dst_w2_s = dst_w2_ptrs_s_rows + sorted_pos * dst_row_stride;

      for (int byte_idx = 0; byte_idx < row_bytes; ++byte_idx) {
        dst_w13_w[byte_idx] = src_w13_w[byte_idx];
        dst_w13_s[byte_idx] = src_w13_s[byte_idx];
        dst_w2_w[byte_idx] = src_w2_w[byte_idx];
        dst_w2_s[byte_idx] = src_w2_s[byte_idx];
      }
    }
  }
  __syncthreads();

  const int total_elems = top_k * hidden_size;
  for (int idx = threadIdx.x; idx < total_elems; idx += blockDim.x) {
    const int col = idx % hidden_size;
    compact_input[idx] = x[col];
  }
}

__global__ void awq_moe_single_token_weighted_reduce_kernel(
    const __half* sorted_output,
    const float* topk_weights,
    const int* inv_permuted_idx,
    __half* out,
    int top_k,
    int hidden_logical_size,
    int sorted_output_row_stride) {
  for (int col = blockIdx.x * blockDim.x + threadIdx.x;
       col < hidden_logical_size;
       col += blockDim.x * gridDim.x) {
    float acc = 0.f;
    for (int route_idx = 0; route_idx < top_k; ++route_idx) {
      const int sorted_pos = inv_permuted_idx[route_idx];
      const __half value =
          sorted_output[sorted_pos * sorted_output_row_stride + col];
      acc = fmaf(topk_weights[route_idx], __half2float(value), acc);
    }
    out[col] = __float2half(acc);
  }
}

template <int TOPK>
__global__ void awq_moe_single_token_weighted_reduce_half2_kernel(
    const __half* sorted_output,
    const float* topk_weights,
    const int* inv_permuted_idx,
    __half* out,
    int hidden_logical_size,
    int sorted_output_row_stride) {
  __shared__ int sorted_pos_sh[TOPK];
  __shared__ float weight_sh[TOPK];

  if (threadIdx.x < TOPK) {
    const int route_idx = threadIdx.x;
    sorted_pos_sh[route_idx] = inv_permuted_idx[route_idx];
    weight_sh[route_idx] = topk_weights[route_idx];
  }
  __syncthreads();

  const int hidden_pairs = hidden_logical_size >> 1;
  for (int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
       pair_idx < hidden_pairs;
       pair_idx += blockDim.x * gridDim.x) {
    float2 acc = {0.f, 0.f};
#pragma unroll
    for (int route_idx = 0; route_idx < TOPK; ++route_idx) {
      const __half2 value = reinterpret_cast<const __half2*>(
          sorted_output + sorted_pos_sh[route_idx] * sorted_output_row_stride)[pair_idx];
      const float2 value_f = __half22float2(value);
      const float weight = weight_sh[route_idx];
      acc.x = fmaf(weight, value_f.x, acc.x);
      acc.y = fmaf(weight, value_f.y, acc.y);
    }
    reinterpret_cast<__half2*>(out)[pair_idx] = __floats2half2_rn(acc.x, acc.y);
  }
}

void awq_moe_single_token_sm70_out(
    torch::Tensor out,
    torch::Tensor x,
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    torch::Tensor src_w13_ptrs_w_rows,
    torch::Tensor src_w13_ptrs_s_rows,
    torch::Tensor src_w2_ptrs_w_rows,
    torch::Tensor src_w2_ptrs_s_rows,
    torch::Tensor compact_input,
    torch::Tensor intermediate,
    torch::Tensor sorted_output,
    torch::Tensor dst_w13_ptrs_w_rows,
    torch::Tensor dst_w13_ptrs_s_rows,
    torch::Tensor dst_w2_ptrs_w_rows,
    torch::Tensor dst_w2_ptrs_s_rows,
    torch::Tensor expert_offsets,
    torch::Tensor inv_permuted_idx,
    int64_t w13_k,
    int64_t w13_n,
    int64_t w2_k,
    int64_t w2_n,
    int64_t group_size,
    int64_t hidden_logical_size) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_sm70_out: out must be CUDA float16.");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_sm70_out: x must be CUDA float16.");
  TORCH_CHECK(topk_weights.is_cuda() &&
                  topk_weights.scalar_type() == torch::kFloat32,
              "awq_moe_single_token_sm70_out: topk_weights must be CUDA float32.");
  TORCH_CHECK(topk_ids.is_cuda() &&
                  (topk_ids.scalar_type() == torch::kInt32 ||
                   topk_ids.scalar_type() == torch::kInt64),
              "awq_moe_single_token_sm70_out: topk_ids must be CUDA int32/int64.");
  TORCH_CHECK(compact_input.is_cuda() &&
                  compact_input.scalar_type() == torch::kFloat16 &&
                  intermediate.is_cuda() &&
                  intermediate.scalar_type() == torch::kFloat16 &&
                  sorted_output.is_cuda() &&
                  sorted_output.scalar_type() == torch::kFloat16,
              "awq_moe_single_token_sm70_out: scratch buffers must be CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32 &&
                  inv_permuted_idx.is_cuda() &&
                  inv_permuted_idx.scalar_type() == torch::kInt32,
              "awq_moe_single_token_sm70_out: index buffers must be CUDA int32.");
  TORCH_CHECK(src_w13_ptrs_w_rows.is_cuda() && src_w13_ptrs_s_rows.is_cuda() &&
                  src_w2_ptrs_w_rows.is_cuda() && src_w2_ptrs_s_rows.is_cuda() &&
                  dst_w13_ptrs_w_rows.is_cuda() && dst_w13_ptrs_s_rows.is_cuda() &&
                  dst_w2_ptrs_w_rows.is_cuda() && dst_w2_ptrs_s_rows.is_cuda(),
              "awq_moe_single_token_sm70_out: ptr rows must be CUDA.");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
              "awq_moe_single_token_sm70_out: x must have shape [1, hidden].");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == 1,
              "awq_moe_single_token_sm70_out: out must have shape [1, hidden].");
  TORCH_CHECK(out.size(1) == hidden_logical_size,
              "awq_moe_single_token_sm70_out: out cols must match hidden_logical_size.");

  topk_ids = topk_ids.contiguous().view({-1});
  topk_weights = topk_weights.contiguous().view({-1});
  inv_permuted_idx = inv_permuted_idx.contiguous().view({-1});

  const int top_k = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(top_k > 0 && top_k <= 32,
              "awq_moe_single_token_sm70_out: top_k must be in [1, 32].");
  TORCH_CHECK(static_cast<int>(topk_weights.numel()) == top_k,
              "awq_moe_single_token_sm70_out: topk_weights size mismatch.");
  TORCH_CHECK(compact_input.dim() == 2 &&
                  compact_input.size(0) == top_k &&
                  compact_input.size(1) == x.size(1),
              "awq_moe_single_token_sm70_out: compact_input shape mismatch.");
  TORCH_CHECK(intermediate.dim() == 2 &&
                  intermediate.size(0) == top_k &&
                  intermediate.size(1) == w13_n / 2,
              "awq_moe_single_token_sm70_out: intermediate shape mismatch.");
  TORCH_CHECK(sorted_output.dim() == 2 &&
                  sorted_output.size(0) == top_k &&
                  sorted_output.size(1) == w2_n,
              "awq_moe_single_token_sm70_out: sorted_output shape mismatch.");
  TORCH_CHECK(expert_offsets.numel() == top_k + 1,
              "awq_moe_single_token_sm70_out: expert_offsets size mismatch.");
  TORCH_CHECK(inv_permuted_idx.numel() == top_k,
              "awq_moe_single_token_sm70_out: inv_permuted_idx size mismatch.");

  const int64_t row_bytes = src_w13_ptrs_w_rows.size(1);
  TORCH_CHECK(src_w13_ptrs_w_rows.dim() == 2 && src_w13_ptrs_s_rows.dim() == 2 &&
                  src_w2_ptrs_w_rows.dim() == 2 && src_w2_ptrs_s_rows.dim() == 2 &&
                  dst_w13_ptrs_w_rows.dim() == 2 && dst_w13_ptrs_s_rows.dim() == 2 &&
                  dst_w2_ptrs_w_rows.dim() == 2 && dst_w2_ptrs_s_rows.dim() == 2,
              "awq_moe_single_token_sm70_out: ptr row tensors must be 2D.");
  TORCH_CHECK(dst_w13_ptrs_w_rows.size(0) == top_k &&
                  dst_w13_ptrs_s_rows.size(0) == top_k &&
                  dst_w2_ptrs_w_rows.size(0) == top_k &&
                  dst_w2_ptrs_s_rows.size(0) == top_k &&
                  dst_w13_ptrs_w_rows.size(1) == row_bytes &&
                  dst_w13_ptrs_s_rows.size(1) == row_bytes &&
                  dst_w2_ptrs_w_rows.size(1) == row_bytes &&
                  dst_w2_ptrs_s_rows.size(1) == row_bytes,
              "awq_moe_single_token_sm70_out: destination ptr row shapes mismatch.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int src_row_stride = static_cast<int>(src_w13_ptrs_w_rows.stride(0));
  const int dst_row_stride = static_cast<int>(dst_w13_ptrs_w_rows.stride(0));
  constexpr int kThreads = 256;

  if (topk_ids.scalar_type() == torch::kInt32) {
    awq_moe_single_token_prepare_kernel<int32_t>
        <<<1, kThreads, 0, stream>>>(
            topk_ids.data_ptr<int32_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            src_w13_ptrs_w_rows.data_ptr<uint8_t>(),
            src_w13_ptrs_s_rows.data_ptr<uint8_t>(),
            src_w2_ptrs_w_rows.data_ptr<uint8_t>(),
            src_w2_ptrs_s_rows.data_ptr<uint8_t>(),
            dst_w13_ptrs_w_rows.data_ptr<uint8_t>(),
            dst_w13_ptrs_s_rows.data_ptr<uint8_t>(),
            dst_w2_ptrs_w_rows.data_ptr<uint8_t>(),
            dst_w2_ptrs_s_rows.data_ptr<uint8_t>(),
            reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
            expert_offsets.data_ptr<int32_t>(),
            inv_permuted_idx.data_ptr<int32_t>(),
            top_k,
            static_cast<int>(x.size(1)),
            src_row_stride,
            dst_row_stride,
            static_cast<int>(row_bytes));
  } else {
    awq_moe_single_token_prepare_kernel<int64_t>
        <<<1, kThreads, 0, stream>>>(
            topk_ids.data_ptr<int64_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            src_w13_ptrs_w_rows.data_ptr<uint8_t>(),
            src_w13_ptrs_s_rows.data_ptr<uint8_t>(),
            src_w2_ptrs_w_rows.data_ptr<uint8_t>(),
            src_w2_ptrs_s_rows.data_ptr<uint8_t>(),
            dst_w13_ptrs_w_rows.data_ptr<uint8_t>(),
            dst_w13_ptrs_s_rows.data_ptr<uint8_t>(),
            dst_w2_ptrs_w_rows.data_ptr<uint8_t>(),
            dst_w2_ptrs_s_rows.data_ptr<uint8_t>(),
            reinterpret_cast<__half*>(compact_input.data_ptr<at::Half>()),
            expert_offsets.data_ptr<int32_t>(),
            inv_permuted_idx.data_ptr<int32_t>(),
            top_k,
            static_cast<int>(x.size(1)),
            src_row_stride,
            dst_row_stride,
            static_cast<int>(row_bytes));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  awq_moe_gemm_sm70_out(
      intermediate,
      compact_input,
      expert_offsets,
      dst_w13_ptrs_w_rows,
      dst_w13_ptrs_s_rows,
      top_k,
      w13_k,
      w13_n,
      group_size,
      true);
  awq_moe_gemm_sm70_out(
      sorted_output,
      intermediate,
      expert_offsets,
      dst_w2_ptrs_w_rows,
      dst_w2_ptrs_s_rows,
      top_k,
      w2_k,
      w2_n,
      group_size,
      false);

  const int hidden_logical_size_i = static_cast<int>(hidden_logical_size);
  const int sorted_output_row_stride = static_cast<int>(sorted_output.stride(0));
  if (top_k == 8 && (hidden_logical_size_i % 2) == 0 &&
      (sorted_output_row_stride % 2) == 0) {
    const int blocks = std::max<int>(
        1, ((hidden_logical_size_i >> 1) + kThreads - 1) / kThreads);
    awq_moe_single_token_weighted_reduce_half2_kernel<8>
        <<<blocks, kThreads, 0, stream>>>(
            reinterpret_cast<const __half*>(sorted_output.data_ptr<at::Half>()),
            topk_weights.data_ptr<float>(),
            inv_permuted_idx.data_ptr<int32_t>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            hidden_logical_size_i,
            sorted_output_row_stride);
  } else {
    const int blocks = std::max<int>(
        1, (hidden_logical_size_i + kThreads - 1) / kThreads);
    awq_moe_single_token_weighted_reduce_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(sorted_output.data_ptr<at::Half>()),
        topk_weights.data_ptr<float>(),
        inv_permuted_idx.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        top_k,
        hidden_logical_size_i,
        sorted_output_row_stride);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void awq_moe_gemm_sm70_out(
    torch::Tensor out,
    torch::Tensor sorted_input,      // [total_tokens, K] float16
    torch::Tensor expert_offsets,    // [num_experts + 1] int32
    torch::Tensor strided_ptrs_w,    // [num_experts * 16] uint8 (StridedPtr array)
    torch::Tensor strided_ptrs_s,    // [num_experts * 16] uint8 (StridedPtr array)
    int64_t num_experts,
    int64_t k,
    int64_t n,
    int64_t group_size,
    bool gated_silu) {
  TORCH_CHECK(sorted_input.is_cuda() && sorted_input.scalar_type() == torch::kFloat16,
              "awq_moe_gemm_sm70: input must be CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() && expert_offsets.scalar_type() == torch::kInt32,
              "awq_moe_gemm_sm70: expert_offsets must be CUDA int32.");
  TORCH_CHECK(strided_ptrs_w.is_cuda() && strided_ptrs_s.is_cuda(),
              "awq_moe_gemm_sm70: strided_ptrs must be CUDA.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "awq_moe_gemm_sm70: output must be CUDA float16.");
  TORCH_CHECK(num_experts > 0 && k > 0 && n > 0,
              "awq_moe_gemm_sm70: invalid dimensions.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(sorted_input));
  const int device = sorted_input.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t total_tokens = sorted_input.size(0);
  TORCH_CHECK(out.size(0) == total_tokens,
              "awq_moe_gemm_sm70: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "awq_moe_gemm_sm70: output must be row-major contiguous.");
  if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "awq_moe_gemm_sm70: gated_silu requires even output dim.");
    TORCH_CHECK(out.size(1) == n / 2,
                "awq_moe_gemm_sm70: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n,
                "awq_moe_gemm_sm70: output cols must match n.");
  }

  if (total_tokens == 0) return;

  const bool grouped = (group_size != k);
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "awq_moe_gemm_sm70: no compatible TurboMind converters.");

  // desc_A: input activations with offsets (kBlocked mode)
  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(k),
      static_cast<int>(k),
  };
  desc_A.num = static_cast<int>(num_experts);
  desc_A.offsets = expert_offsets.data_ptr<int>();

  turbomind::gemm::MatrixLayout desc_U{};

  // desc_B: weights via StridedPtr (ld=0 triggers StridedPtr resolution)
  const auto order_w = conv_w->order;
  const bool is_A_w =
      turbomind::gemm::get_operand_tag(conv_w->pack) ==
      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf, order_w,
      static_cast<int>(n), static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.type = turbomind::data_type_v<turbomind::uint4_t>;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = 0;  // StridedPtr mode
  desc_B.num = static_cast<int>(num_experts);

  // desc_V: scales via StridedPtr
  const auto order_s = conv_s->order;
  const bool is_A_s =
      turbomind::gemm::get_operand_tag(conv_s->pack) ==
      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;

  const int64_t num_groups_raw = k / group_size;

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint32, order_s,
      static_cast<int>(n), static_cast<int>(num_groups_raw),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_V = s_desc;
  desc_V.pack = conv_s->pack;
  if (is_A_s) {
    desc_V = turbomind::gemm::transpose(desc_V);
  }
  desc_V.ld = 0;  // StridedPtr mode
  desc_V.num = static_cast<int>(num_experts);

  // desc_D: output with offsets (same as A)
  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(n),
      static_cast<int>(out.stride(0)),
  };
  desc_D.num = static_cast<int>(num_experts);
  desc_D.offsets = expert_offsets.data_ptr<int>();

  turbomind::gemm::Operation op{};
  op.dispatch = vllm::awq_sm70::awq_select_moe_dispatch_policy(
      device, static_cast<int>(total_tokens), static_cast<int>(n),
      static_cast<int>(k), static_cast<int>(num_experts),
      static_cast<int>(group_size), stream);
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kK, static_cast<int>(group_size)};
  op.batch_dim = 0;

  auto& workspace_holder = vllm::awq_sm70::get_workspace(device, stream);
  auto& gemm = vllm::awq_sm70::get_gemm(device);

  const int ec = gemm.Run(op, 1.f,
      sorted_input.data_ptr(), desc_A,
      nullptr, desc_U,
      strided_ptrs_w.data_ptr(), desc_B,
      strided_ptrs_s.data_ptr(), desc_V,
      0.f,
      out.data_ptr(), desc_D,
      out.data_ptr(), desc_D,
      workspace_holder.workspace, stream);

  TORCH_CHECK(ec == 0, "awq_moe_gemm_sm70: TurboMind batched GEMM failed (ec=",
              ec, ").");
}

torch::Tensor awq_moe_gemm_sm70(
    torch::Tensor sorted_input,
    torch::Tensor expert_offsets,
    torch::Tensor strided_ptrs_w,
    torch::Tensor strided_ptrs_s,
    int64_t num_experts,
    int64_t k,
    int64_t n,
    int64_t group_size) {
  auto out = torch::empty(
      {sorted_input.size(0), n},
      torch::TensorOptions()
          .dtype(sorted_input.dtype())
          .device(sorted_input.device()));
  awq_moe_gemm_sm70_out(out, sorted_input, expert_offsets, strided_ptrs_w,
                        strided_ptrs_s, num_experts, k, n, group_size, false);
  return out;
}
