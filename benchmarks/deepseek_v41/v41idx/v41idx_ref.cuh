// Copy of index_scores_decode from benchmarks/deepseek_v41/v41k/v41k.cu (#1277).
// Wrapped in namespace v41idx_ref; used only by the check op index_scores_ref. Do not edit.
namespace v41idx_ref {
// ---- G4b decode indexer: scores over packed FP4 index keys ----------------
// Packed key row (68 B): 64 B E2M1 nibbles (even element in the low nibble)
// then 4 power-of-two scale exponents (biased) for groups of 32. Rows q/w are
// (token, head) pairs padded to 64. Output fp16 out[t, n] =
//   sum_h fp16(fp16(relu(fp16(q[t,h] . k[n]))) * w[t,h]), for n < live end.
__device__ __forceinline__ float v41k_e2m1(int code) {
  const int m = code & 7;
  const float mag = m < 2 ? 0.5f * m
                          : __uint_as_float(uint32_t(((m >> 1) + 126) << 23) |
                                            uint32_t((m & 1) << 22));
  return (code & 8) ? -mag : mag;
}

template <int KB>
__global__ void __launch_bounds__(256)
    index_scores_decode(const half* __restrict__ q, const half* __restrict__ w,
                        const uint8_t* __restrict__ keys,
                        const int64_t* __restrict__ positions,
                        half* __restrict__ out, int tokens, int heads,
                        int64_t n_out, int ratio) {
  __shared__ half qs[64][136];
  __shared__ float ks[KB][129];
  __shared__ float ws[64];
  const int t = threadIdx.x;
  const int64_t end = (positions[tokens - 1] + 1) / ratio;
  const int64_t n0 = int64_t(blockIdx.x) * KB;
  if (n0 >= end) return;
  const int rows = tokens * heads;
  for (int i = t; i < 64 * 128; i += 256) {
    const int r = i / 128, d = i % 128;
    qs[r][d] = r < rows ? q[r * 128 + d] : __float2half(0.f);
  }
  if (t < 64) ws[t] = t < rows ? __half2float(w[t]) : 0.f;
  for (int i = t; i < KB * 64; i += 256) {
    const int k = i / 64, b = i % 64;
    const int64_t n = n0 + k;
    float lo = 0.f, hi = 0.f;
    if (n < end) {
      const uint8_t* row = keys + n * 68;
      const int byte = row[b];
      const float scale =
          __uint_as_float(uint32_t(row[64 + (2 * b) / 32]) << 23);
      lo = v41k_e2m1(byte & 15) * scale;
      hi = v41k_e2m1((byte >> 4) & 15) * scale;
    }
    // Round like the FP16 cache values (exact for these products).
    ks[k][2 * b] = __half2float(__float2half_rn(lo));
    ks[k][2 * b + 1] = __half2float(__float2half_rn(hi));
  }
  __syncthreads();
  // Each thread: one key column (k), a stripe of (token) outputs.
  for (int item = t; item < KB * tokens; item += 256) {
    const int k = item % KB, tok = item / KB;
    const int64_t n = n0 + k;
    if (n >= end) continue;
    float total = 0.f;
    for (int h = 0; h < heads; ++h) {
      const int r = tok * heads + h;
      float dot = 0.f;
#pragma unroll 16
      for (int d = 0; d < 128; ++d)
        dot = fmaf(__half2float(qs[r][d]), ks[k][d], dot);
      float s = __half2float(__float2half_rn(dot));
      s = s > 0.f ? s : 0.f;
      total += __half2float(__float2half_rn(s * ws[r]));
    }
    out[int64_t(tok) * n_out + n] = __float2half_rn(total);
  }
}
}  // namespace v41idx_ref
