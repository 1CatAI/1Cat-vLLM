// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

// Lifetime fence for the TP-group coordinator. This class does not issue CUDA
// work or store process-local pointers. The coordinator owns residency and the
// arena allocator, publishes copies using these 64-bit ranges, and recycles
// input ranges only after finish() succeeds. It is not a per-rank swap lane.
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vllm::gguf {

struct SharedExpertRange {
  uint64_t offset;
  uint64_t bytes;
};

class SharedExpertSwapState {
 public:
  SharedExpertSwapState(uint64_t arena_bytes, unsigned ranks)
      : arena_bytes_(arena_bytes), ranks_(ranks) {
    if (!arena_bytes || !ranks || ranks > 64)
      throw std::invalid_argument("invalid shared expert group dimensions");
  }

  // The prefetcher must hold a lease until all its asynchronous reads finish.
  // Lease acquisition and publication of a swap are serialized here. No new
  // lease can extend the lifetime of an input after that swap is published.
  uint64_t acquire(SharedExpertRange range) {
    std::lock_guard<std::mutex> guard(mutex_);
    healthy();
    validate(range);
    if (active_) {
      for (auto input : inputs_)
        if (overlap(range, input))
          throw std::runtime_error(
              "prefetch input is fenced by an active swap");
      for (auto output : outputs_)
        if (overlap(range, output))
          throw std::runtime_error(
              "prefetch cannot read an unfinished writeback");
    }
    if (next_lease_ == std::numeric_limits<uint64_t>::max())
      poison("shared expert lease sequence exhausted");
    const uint64_t id = ++next_lease_;
    leases_.emplace(id, range);
    return id;
  }

  void release(uint64_t lease) {
    std::lock_guard<std::mutex> guard(mutex_);
    healthy();
    if (leases_.erase(lease) != 1) poison("unknown shared expert read lease");
  }

  // Input and output allocations must be distinct even when all four devices
  // appear to execute the same plan. A slow rank may still read any input.
  void begin(uint64_t sequence, std::vector<SharedExpertRange> inputs,
             std::vector<SharedExpertRange> outputs, uint64_t now_ns,
             uint64_t timeout_ns) {
    std::lock_guard<std::mutex> guard(mutex_);
    healthy();
    if (active_) throw std::runtime_error("shared expert swap already active");
    if (!sequence || sequence_ == std::numeric_limits<uint64_t>::max() ||
        sequence != sequence_ + 1)
      poison("shared expert plan sequence mismatch");
    // Free-slot loads have an input without an output (no victim to save).
    if (outputs.size() > inputs.size() || !timeout_ns ||
        timeout_ns > std::numeric_limits<uint64_t>::max() - now_ns)
      throw std::invalid_argument("invalid shared expert swap plan");
    for (size_t i = 0; i < inputs.size(); ++i) {
      validate(inputs[i]);
      for (size_t j = 0; j < i; ++j)
        if (overlap(inputs[i], inputs[j]))
          throw std::invalid_argument("shared expert plan allocations overlap");
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      validate(outputs[i]);
      for (size_t j = 0; j < inputs.size(); ++j) {
        const bool inplace = outputs[i].offset == inputs[j].offset &&
                             outputs[i].bytes == inputs[j].bytes;
        if (!inplace && overlap(outputs[i], inputs[j]))
          throw std::invalid_argument("shared writeback aliases an input");
      }
      for (size_t j = 0; j < i; ++j)
        if (overlap(outputs[i], outputs[j]))
          throw std::invalid_argument("shared expert plan allocations overlap");
      for (auto& entry : leases_)
        if (overlap(outputs[i], entry.second))
          throw std::invalid_argument(
              "shared writeback aliases a live prefetch");
    }
    sequence_ = sequence;
    deadline_ns_ = now_ns + timeout_ns;
    inputs_ = std::move(inputs);
    outputs_ = std::move(outputs);
    completed_ = 0;
    active_ = true;
  }

  // A rank acknowledges only after its reads AND its unique-owner writes
  // complete. CPU completion publication uses acquire/release outside this
  // object; receiving a CUDA launch notification is insufficient.
  void acknowledge(unsigned rank, uint64_t sequence, uint64_t now_ns) {
    std::lock_guard<std::mutex> guard(mutex_);
    check_plan(sequence, now_ns);
    if (rank >= ranks_) poison("invalid shared expert completion rank");
    const uint64_t bit = uint64_t{1} << rank;
    if (completed_ & bit) poison("duplicate shared expert completion");
    completed_ |= bit;
  }

  bool ready(uint64_t sequence, uint64_t now_ns) {
    std::lock_guard<std::mutex> guard(mutex_);
    check_plan(sequence, now_ns);
    return ready_locked();
  }

  // Returned ranges are the only old input allocations the caller may recycle.
  // Retain all allocations on failure: devices may already have modified slots,
  // so an aborted plan requires explicit cache reinitialization.
  std::vector<SharedExpertRange> finish(uint64_t sequence, uint64_t now_ns) {
    std::lock_guard<std::mutex> guard(mutex_);
    check_plan(sequence, now_ns);
    if (!ready_locked())
      throw std::runtime_error("shared expert readers have not completed");
    active_ = false;
    outputs_.clear();
    return std::exchange(inputs_, {});
  }

  void stop() {
    std::lock_guard<std::mutex> guard(mutex_);
    failure_ = "shared expert coordinator stopped";
  }

 private:
  static bool overlap(SharedExpertRange a, SharedExpertRange b) {
    // Validated ranges cannot wrap. Use subtraction to remain overflow-safe
    // even while checking an as-yet unvalidated prospective plan.
    return a.offset <= b.offset ? b.offset - a.offset < a.bytes
                                : a.offset - b.offset < b.bytes;
  }

  void validate(SharedExpertRange range) const {
    if (!range.bytes || range.offset >= arena_bytes_ ||
        range.bytes > arena_bytes_ - range.offset)
      throw std::invalid_argument("shared expert range exceeds arena");
  }

  void healthy() const {
    if (!failure_.empty()) throw std::runtime_error(failure_);
  }

  [[noreturn]] void poison(const char* reason) {
    failure_ = reason;
    throw std::runtime_error(failure_);
  }

  void check_plan(uint64_t sequence, uint64_t now_ns) {
    healthy();
    if (!active_ || sequence != sequence_)
      poison("shared expert completion sequence mismatch");
    if (now_ns >= deadline_ns_) poison("shared expert plan timed out");
  }

  bool ready_locked() const {
    const uint64_t expected =
        ranks_ == 64 ? ~uint64_t{0} : (uint64_t{1} << ranks_) - 1;
    if (completed_ != expected) return false;
    for (auto& entry : leases_)
      for (auto input : inputs_)
        if (overlap(entry.second, input)) return false;
    return true;
  }

  const uint64_t arena_bytes_;
  const unsigned ranks_;
  std::mutex mutex_;
  uint64_t next_lease_ = 0, sequence_ = 0, deadline_ns_ = 0, completed_ = 0;
  bool active_ = false;
  std::string failure_;
  std::vector<SharedExpertRange> inputs_, outputs_;
  std::unordered_map<uint64_t, SharedExpertRange> leases_;
};

}  // namespace vllm::gguf
