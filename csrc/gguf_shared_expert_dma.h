// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

// Per-rank writeback lane for the in-place expert exchange. After every rank
// of the group has read a plan's incoming bytes (each rank publishes its
// read-done sequence in the control page), this host thread copies the
// victims' exclusive bytes from the rank's device scratch into the shared
// arena with copy-engine DMA, then publishes the rank's completion sequence.
// SM zero-copy stores here would throttle the GPU memory pipeline for every
// concurrently running kernel; the copy engine does not.
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#if defined(__x86_64__) || defined(__i386__)
  #include <immintrin.h>
#endif

namespace vllm::gguf {

class SharedExpertDmaLane {
 public:
  // layouts[layer] = 3 x (S, R, F, D, W, N, P), as the device swap kernels.
  SharedExpertDmaLane(uint64_t control, uint64_t arena, uint64_t scratch,
                      int device, unsigned rank, unsigned ranks,
                      std::vector<std::vector<int64_t>> layouts,
                      uint64_t scratch_per_op, unsigned rank_start,
                      unsigned rank_words, unsigned op_start, unsigned op_words,
                      unsigned flag_word, uint64_t timeout_ns)
      : control_(reinterpret_cast<uint64_t*>(control)),
        arena_(reinterpret_cast<uint8_t*>(arena)),
        scratch_(reinterpret_cast<uint8_t*>(scratch)),
        device_(device),
        rank_(rank),
        ranks_(ranks),
        layouts_(std::move(layouts)),
        per_op_(scratch_per_op),
        rank_start_(rank_start),
        rank_words_(rank_words),
        op_start_(op_start),
        op_words_(op_words),
        flag_word_(flag_word),
        timeout_ns_(timeout_ns) {
    if (!control_ || !arena_ || !scratch_ || rank >= ranks || ranks > 8)
      throw std::invalid_argument("invalid shared expert DMA lane");
    for (const auto& l : layouts_)
      if (l.size() != 21) throw std::invalid_argument("DMA lane layout");
    last_ = load(flag(rank_));
    thread_ = std::thread([this] { run(); });
  }

  ~SharedExpertDmaLane() { stop(); }

  void stop() {
    stopped_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
  }

  std::string error() {
    std::lock_guard<std::mutex> guard(mutex_);
    return error_;
  }

  uint64_t completed() const { return done_.load(std::memory_order_acquire); }

 private:
  static uint64_t load(const uint64_t* p) {
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
  }
  static void relax() {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
  }
  static uint64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  uint64_t* rank_words(unsigned r) {
    return control_ + rank_start_ + r * rank_words_;
  }
  uint64_t* flag(unsigned r) { return rank_words(r) + flag_word_; }

  void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
      throw std::runtime_error(std::string("DMA lane ") + what + ": " +
                               cudaGetErrorString(e));
  }

  void run() {
    try {
      check(cudaSetDevice(device_), "set device");
      // The model captures CUDA graphs in global mode on another thread; the
      // copies and syncs on this lane's own stream must not be prohibited.
      cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
      check(cudaThreadExchangeStreamCaptureMode(&mode), "capture mode");
      cudaStream_t stream;
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "stream");
      uint64_t active_until = 0;
      while (!stopped_.load(std::memory_order_acquire)) {
        const uint64_t seq = load(flag(rank_));
        if (seq == last_) {
          if (now() < active_until)
            relax();
          else
            std::this_thread::sleep_for(std::chrono::microseconds(10));
          continue;
        }
        const uint64_t begin = now();
        if (debug_)
          std::fprintf(
              stderr,
              "[dma r%u] seq %llu seen (last %llu) flags %llu %llu %llu %llu\n",
              rank_, (unsigned long long)seq, (unsigned long long)last_,
              (unsigned long long)load(flag(0)),
              (unsigned long long)load(flag(1)),
              (unsigned long long)load(flag(2)),
              (unsigned long long)load(flag(3)));
        // Boundary down blocks are read by two ranks: no rank may overwrite
        // the incoming allocation before every rank has read it.
        for (;;) {
          bool all = true;
          for (unsigned r = 0; r < ranks_; ++r) all &= load(flag(r)) >= seq;
          if (all) break;
          if (stopped_.load(std::memory_order_acquire)) return;
          if (now() - begin > timeout_ns_)
            throw std::runtime_error("DMA lane: ranks did not finish reading");
          relax();
        }
        const uint64_t layer = rank_words(rank_)[2];
        const uint64_t copies = load(control_ + 3);
        if (layer >= layouts_.size() || copies > 64)
          throw std::runtime_error("DMA lane: invalid plan");
        const auto& L = layouts_[layer];
        uint64_t prefix[3], off = 0;
        for (int i = 0; i < 3; ++i) {
          prefix[i] = off;
          off += uint64_t(L[7 * i + 5] * L[7 * i + 4]);
        }
        const uint64_t layer_per_op = (off + 255) / 256 * 256;  // as swap_stage
        for (uint64_t op = 0; op < copies; ++op) {
          const uint64_t dst0 = load(control_ + op_start_ + op * op_words_ + 3);
          for (int i = 0; i < 3; ++i) {
            const int64_t S = L[7 * i], R = L[7 * i + 1], F = L[7 * i + 2],
                          W = L[7 * i + 4], N = L[7 * i + 5], P = L[7 * i + 6];
            if (!W || !N) continue;
            check(cudaMemcpy2DAsync(arena_ + dst0 + R * S + F + P, S,
                                    scratch_ + op * layer_per_op + prefix[i], W,
                                    W, N, cudaMemcpyDeviceToHost, stream),
                  "writeback");
          }
        }
        if (debug_)
          std::fprintf(stderr,
                       "[dma r%u] seq %llu flags ok after %llu us, layer %llu "
                       "copies %llu\n",
                       rank_, (unsigned long long)seq,
                       (unsigned long long)((now() - begin) / 1000),
                       (unsigned long long)layer, (unsigned long long)copies);
        check(cudaStreamSynchronize(stream), "writeback sync");
        if (debug_)
          std::fprintf(stderr, "[dma r%u] seq %llu acked after %llu us\n",
                       rank_, (unsigned long long)seq,
                       (unsigned long long)((now() - begin) / 1000));
        __atomic_store_n(rank_words(rank_) + 1, seq, __ATOMIC_RELEASE);
        last_ = seq;
        done_.fetch_add(1, std::memory_order_release);
        active_until = now() + 4'000'000;
      }
      cudaStreamDestroy(stream);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[SharedExpertDmaLane rank %u] %s\n", rank_,
                   e.what());
      std::lock_guard<std::mutex> guard(mutex_);
      error_ = e.what();
    }
  }

  uint64_t* const control_;
  uint8_t* const arena_;
  uint8_t* const scratch_;
  const int device_;
  const unsigned rank_, ranks_;
  const std::vector<std::vector<int64_t>> layouts_;
  const uint64_t per_op_;
  const unsigned rank_start_, rank_words_, op_start_, op_words_, flag_word_;
  const uint64_t timeout_ns_;
  uint64_t last_ = 0;
  const bool debug_ = std::getenv("VLLM_V41_DMA_DEBUG") != nullptr;
  std::atomic<bool> stopped_{false};
  std::atomic<uint64_t> done_{0};
  std::thread thread_;
  std::mutex mutex_;
  std::string error_;
};

}  // namespace vllm::gguf
