// SPDX-License-Identifier: Apache-2.0
#include <climits>
#include <cmath>
#include <flashinfer/sm70/qsa.cuh>

#include "common.h"

namespace {
void qsa_scores(TensorView scores, TensorView visible, TensorView grouped_rows, TensorView counts,
                TensorView q, TensorView cache, TensorView table, TensorView requests,
                TensorView positions, TensorView lengths, double divisor, int64_t compress) {
  const auto device = q.device();
  check_tensor(q, device, kDLFloat, 16);
  check_tensor(cache, device, kDLFloat, 16);
  check_tensor(scores, device, kDLFloat, 32);
  for (auto t : {visible, grouped_rows, counts, table, requests, lengths})
    check_tensor(t, device, kDLInt, 32);
  TVM_FFI_ICHECK(positions.dtype().bits == 32 || positions.dtype().bits == 64);
  check_tensor(positions, device, kDLInt, positions.dtype().bits);
  TVM_FFI_ICHECK(q.ndim() == 3 && q.size(0) <= 32 && q.size(1) == 4 && q.size(2) == 128);
  TVM_FFI_ICHECK(cache.ndim() == 4 && cache.size(2) == 1 && cache.size(3) == 128);
  TVM_FFI_ICHECK(table.ndim() == 2 && grouped_rows.ndim() == 2 && scores.ndim() == 2);
  const auto m = q.size(0), nreq = table.size(0), columns = scores.size(1);
  const auto capacity = grouped_rows.size(1);
  TVM_FFI_ICHECK(capacity >= m && capacity <= 32 && capacity > 0 && nreq > 0 &&
                 nreq <= 65535 / ((capacity + 7) / 8) && grouped_rows.size(0) >= nreq &&
                 grouped_rows.stride(1) == 1 && grouped_rows.stride(0) == capacity &&
                 scores.size(0) == m && scores.stride(1) == 1 && scores.stride(0) >= columns &&
                 columns <= INT32_MAX && cache.size(0) <= INT32_MAX && cache.size(1) > 0 &&
                 cache.size(1) <= INT32_MAX && table.size(1) <= INT32_MAX &&
                 columns <= table.size(1) * cache.size(1) && std::isfinite(divisor) &&
                 divisor > 0 && compress > 0 && compress <= INT32_MAX);
  for (auto t : {requests, positions, visible})
    TVM_FFI_ICHECK(t.ndim() == 1 && t.size(0) == m && t.stride(0) == 1);
  for (auto t : {counts, lengths})
    TVM_FFI_ICHECK(t.ndim() == 1 && t.size(0) >= nreq && t.stride(0) == 1);
  tvm::ffi::CUDADeviceGuard guard(device.device_id);
  check_sm70();
  if (!m) return;
  const auto stream = stream_for(q);
  const dim3 grid((columns + 31) / 32, nreq * ((capacity + 7) / 8));
#define SCORE(P, G, GRID)                                                                      \
  flashinfer::sm70::mqa_multiquery<P, G><<<GRID, 128, 0, stream>>>(                            \
      static_cast<half*>(q.data_ptr()), static_cast<half*>(cache.data_ptr()),                  \
      static_cast<int*>(table.data_ptr()), static_cast<int*>(grouped_rows.data_ptr()),         \
      static_cast<int*>(counts.data_ptr()), static_cast<int*>(requests.data_ptr()),            \
      static_cast<int*>(visible.data_ptr()), m, static_cast<P*>(positions.data_ptr()),         \
      static_cast<int*>(lengths.data_ptr()), static_cast<float*>(scores.data_ptr()), capacity, \
      columns, scores.stride(0), cache.size(0), cache.size(1), table.size(1), q.stride(0),     \
      q.stride(1), q.stride(2), cache.stride(0), cache.stride(1), cache.stride(3),             \
      table.stride(0), table.stride(1), divisor, compress)
#define RUN(P)                                                                                    \
  if (nreq == 1 && m <= 8 && columns) {                                                           \
    const dim3 single_grid((columns + 31) / 32);                                                  \
    SCORE(P, false, single_grid);                                                                 \
  } else {                                                                                        \
    flashinfer::sm70::group_qsa_rows<P><<<nreq, 32, 0, stream>>>(                                 \
        static_cast<int*>(requests.data_ptr()), static_cast<P*>(positions.data_ptr()),            \
        static_cast<int*>(lengths.data_ptr()), static_cast<int*>(grouped_rows.data_ptr()),        \
        static_cast<int*>(counts.data_ptr()), static_cast<int*>(visible.data_ptr()), m, capacity, \
        nreq, compress);                                                                          \
    if (columns) {                                                                                \
      SCORE(P, true, grid);                                                                       \
    }                                                                                             \
  }
  if (positions.dtype().bits == 64) {
    RUN(int64_t);
  } else {
    RUN(int32_t);
  }
#undef RUN
#undef SCORE
  check_cuda(cudaGetLastError());
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(qsa_scores, qsa_scores);
