// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

// Per-rank background job lane (VLLM_V41_ASYNC_EVICT). The coordinator
// publishes a save only after every rank's GPU committed the plan that
// dropped the expert, so decode kernels no longer read that slot; a load
// (prefetch) targets an empty slot that no table references. This thread
// copies each job's rank-local bytes with copy-engine DMA, saves and loads on
// two streams (both copy engines), and completion counts per kind are
// written by the streams themselves (cuStreamWriteValue64). No GPU kernel
// waits on this lane; the coordinator may wait for a load only once every
// rank has issued it, which then depends on DMA progress alone.
#include <cuda.h>
#include <cuda_runtime.h>

#include "gguf_shared_expert_lane.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace vllm::gguf {

class SharedExpertJobLane {
 public:
  // banks[layer] = 3 bank base pointers then 3 slot strides (bytes);
  // layouts[layer] = 3 x (S, R, F, D, W, N, P), as the device swap kernels.
  SharedExpertJobLane(uint64_t control, uint64_t arena, int device,
                      unsigned rank, std::vector<std::vector<int64_t>> banks,
                      std::vector<std::vector<int64_t>> layouts,
                      uint64_t timeout_ns)
      : control_(reinterpret_cast<uint64_t*>(control)),
        arena_(reinterpret_cast<uint8_t*>(arena)),
        device_(device),
        rank_(rank),
        banks_(std::move(banks)),
        layouts_(std::move(layouts)),
        timeout_ns_(timeout_ns) {
    if (!control_ || !arena_ || rank >= kSharedMaxRanks ||
        banks_.size() != layouts_.size())
      throw std::invalid_argument("invalid shared expert job lane");
    for (size_t i = 0; i < banks_.size(); ++i)
      if (banks_[i].size() != 6 || layouts_[i].size() != 21)
        throw std::invalid_argument("shared expert job lane layout");
    issued_ = load(issued_word());
    for (int k = 0; k < 2; ++k) ordinal_[k] = load(done_word(k));
    thread_ = std::thread([this] { run(); });
  }

  ~SharedExpertJobLane() { stop(); }

  void stop() {
    stopped_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
  }

  std::string error() {
    std::lock_guard<std::mutex> guard(mutex_);
    return error_;
  }

  uint64_t completed() { return load(done_word(0)) + load(done_word(1)); }

 private:
  static uint64_t load(const uint64_t* p) {
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
  }
  static uint64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  uint64_t* jobs() { return control_ + kSharedJobStart; }
  uint64_t* done_word(int kind) { return jobs() + 1 + 4 * kind + rank_; }
  uint64_t* issued_word() { return jobs() + 9 + rank_; }

  using WriteValue64 = CUresult (*)(CUstream, CUdeviceptr, cuuint64_t,
                                    unsigned int);
  WriteValue64 stream_write_ = nullptr;
  CUdeviceptr done_device_[2] = {0, 0};
  void init_stream_write() {
    if (std::getenv("VLLM_V41_JOB_EVENTS")) return;
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult status;
    if (cudaGetDriverEntryPoint("cuStreamWriteValue64", &fn, cudaEnableDefault,
                                &status) != cudaSuccess ||
        status != cudaDriverEntryPointSuccess || !fn)
      return;
    for (int k = 0; k < 2; ++k) {
      void* device = nullptr;
      if (cudaHostGetDevicePointer(&device, done_word(k), 0) != cudaSuccess) {
        cudaGetLastError();
        return;
      }
      done_device_[k] = reinterpret_cast<CUdeviceptr>(device);
    }
    stream_write_ = reinterpret_cast<WriteValue64>(fn);
  }

  void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
      throw std::runtime_error(std::string("job lane ") + what + ": " +
                               cudaGetErrorString(e));
  }

  void issue(const uint64_t* entry, cudaStream_t stream) {
    const uint64_t kind = entry[1], layer = entry[2], slot = entry[3],
                   at = entry[4];
    const auto& L = layouts_[layer];
    const auto& B = banks_[layer];
    for (int i = 0; i < 3; ++i) {
      const int64_t S = L[7 * i], R = L[7 * i + 1], F = L[7 * i + 2],
                    D = L[7 * i + 3], W = L[7 * i + 4], N = L[7 * i + 5],
                    P = L[7 * i + 6];
      if (!N) continue;
      auto* dev = reinterpret_cast<uint8_t*>(B[i]) + int64_t(slot) * B[3 + i];
      auto* host = arena_ + at + P + R * S + F;
      if (kind == 0) {
        if (W)
          check(cudaMemcpy2DAsync(host, S, dev, D, W, N, cudaMemcpyDeviceToHost,
                                  stream),
                "save");
      } else {
        check(cudaMemcpy2DAsync(dev, D, host, S, D, N, cudaMemcpyHostToDevice,
                                stream),
              "load");
      }
    }
  }

  void run() {
    try {
      check(cudaSetDevice(device_), "set device");
      // The model captures CUDA graphs in global mode on another thread.
      cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
      check(cudaThreadExchangeStreamCaptureMode(&mode), "capture mode");
      cudaStream_t streams[2];
      for (auto& s : streams)
        check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
      init_stream_write();
      if (std::getenv("VLLM_V41_DMA_DEBUG"))
        std::fprintf(stderr, "[job lane r%u] stream write %s\n", rank_,
                     stream_write_ ? "on" : "off (events)");
      std::deque<std::pair<cudaEvent_t, uint64_t>> inflight[2];
      std::vector<cudaEvent_t> spare;
      uint64_t active_until = 0;
      while (!stopped_.load(std::memory_order_acquire)) {
        bool busy = false;
        const uint64_t head = load(jobs());
        if (issued_ < head) {
          bool used[2] = {false, false};
          for (; issued_ < head; ++issued_) {
            const uint64_t* entry =
                jobs() + kSharedJobHeader +
                (issued_ % kSharedJobEntries) * kSharedJobWords;
            if (load(entry) != issued_ + 1)
              throw std::runtime_error("job lane: torn or lapped job entry");
            const uint64_t kind = entry[1];
            if (entry[2] >= layouts_.size() || kind > 1 ||
                entry[6] != ordinal_[kind])
              throw std::runtime_error("job lane: invalid job");
            issue(entry, streams[kind]);
            ++ordinal_[kind];
            used[kind] = true;
          }
          for (int k = 0; k < 2; ++k) {
            if (!used[k]) continue;
            if (stream_write_) {
              if (stream_write_(reinterpret_cast<CUstream>(streams[k]),
                                done_device_[k], ordinal_[k],
                                0) != CUDA_SUCCESS)
                throw std::runtime_error("job lane: stream write");
              continue;
            }
            cudaEvent_t event;
            if (spare.empty()) {
              check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                    "event");
            } else {
              event = spare.back();
              spare.pop_back();
            }
            check(cudaEventRecord(event, streams[k]), "record");
            inflight[k].push_back({event, ordinal_[k]});
          }
          __atomic_store_n(issued_word(), issued_, __ATOMIC_RELEASE);
          if (debug_)
            std::fprintf(stderr, "[job lane r%u] issued %llu ord %llu/%llu\n",
                         rank_, (unsigned long long)issued_,
                         (unsigned long long)ordinal_[0],
                         (unsigned long long)ordinal_[1]);
          busy = true;
        }
        for (int k = 0; k < 2; ++k) {
          while (!inflight[k].empty()) {
            const auto status = cudaEventQuery(inflight[k].front().first);
            if (status == cudaErrorNotReady) break;
            check(status, "query");
            __atomic_store_n(done_word(k), inflight[k].front().second,
                             __ATOMIC_RELEASE);
            spare.push_back(inflight[k].front().first);
            inflight[k].pop_front();
            busy = true;
          }
        }
        if (busy) {
          active_until = now() + 4'000'000;
          continue;
        }
        if (!inflight[0].empty() || !inflight[1].empty() ||
            now() < active_until) {
#if defined(__x86_64__) || defined(__i386__)
          _mm_pause();
#else
          std::this_thread::yield();
#endif
        } else {
          std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
      }
      for (auto& q : inflight)
        for (auto& e : q) cudaEventDestroy(e.first);
      for (auto e : spare) cudaEventDestroy(e);
      for (auto s : streams) cudaStreamDestroy(s);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[SharedExpertJobLane rank %u] %s\n", rank_,
                   e.what());
      std::lock_guard<std::mutex> guard(mutex_);
      error_ = e.what();
    }
  }

  uint64_t* const control_;
  uint8_t* const arena_;
  const int device_;
  const unsigned rank_;
  const std::vector<std::vector<int64_t>> banks_, layouts_;
  const uint64_t timeout_ns_;
  uint64_t issued_ = 0;
  uint64_t ordinal_[2] = {0, 0};
  const bool debug_ = std::getenv("VLLM_V41_DMA_DEBUG") != nullptr;
  std::atomic<bool> stopped_{false};
  std::thread thread_;
  std::mutex mutex_;
  std::string error_;
};

}  // namespace vllm::gguf
