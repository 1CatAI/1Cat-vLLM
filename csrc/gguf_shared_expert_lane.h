// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include "gguf_shared_expert_planner.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <chrono>
#include <thread>
#if defined(__x86_64__) || defined(__i386__)
  #include <immintrin.h>
#endif

namespace vllm::gguf {

// All offsets below are uint64 words within one separate MAP_SHARED page.
// Only the owner CPU lane writes the plan/commit/failure header. Each rank
// writes its own request and completion sequence with a system-scope release.
constexpr size_t kSharedMaxRanks = 4;
constexpr size_t kSharedMaxRoutes = 36;
constexpr size_t kSharedMaxCopies = 44;
constexpr size_t kSharedOpStart = 8;
constexpr size_t kSharedOpWords = 5;  // slot, incoming, victim, load, save
constexpr size_t kSharedRankStart = 232;
constexpr size_t kSharedRankWords = 64;
constexpr size_t kSharedControlWords = 512;
static_assert(kSharedOpStart + kSharedMaxCopies * kSharedOpWords <=
              kSharedRankStart);
static_assert(kSharedRankStart + kSharedMaxRanks * kSharedRankWords <=
              kSharedControlWords);
// GPU-resolved hit log, after the control words: word 0 is the producer
// head (entries written), then a ring of entries {tag=index+1, layer, count,
// ids[36]}. Rank 0 alone produces; the lane consumes it for LRU uses only.
constexpr size_t kSharedHitStart = kSharedControlWords;
constexpr size_t kSharedHitHeader = 8;
constexpr size_t kSharedHitEntries = 256;
constexpr size_t kSharedHitWords = 40;
constexpr size_t kSharedHitTotalWords =
    kSharedHitStart + kSharedHitHeader + kSharedHitEntries * kSharedHitWords;
static_assert(3 + kSharedMaxRoutes <= kSharedHitWords);
// Background job ring (async eviction / prefetch), after the hit log: header
// word 0 = jobs published (coordinator), word 1 + 4 * kind + r = jobs of that
// kind completed on rank r (written by that rank's copy stream itself when
// supported), word 9 + r = jobs whose copies rank r's lane has issued;
// entries {tag=index+1, kind, layer, slot, offset, expert, kind ordinal}.
constexpr size_t kSharedJobStart = (kSharedHitTotalWords + 7) / 8 * 8;
constexpr size_t kSharedJobHeader = 16;
constexpr size_t kSharedJobEntries = 256;
constexpr size_t kSharedJobWords = 8;
// Per-rank prefetch hint rings (VLLM_V41_PREFETCH), after the job ring: each
// rank publishes, per decode layer i, its own expert quarter's top candidates
// per verify token for layers i+1 and i+2: header word 0 = entries written;
// entry {tag=index+1, layer, tokens, trunc, words[2][8][8] = expert << 32 |
// float bits of the biased selection score}.
constexpr size_t kSharedHintStart =
    kSharedJobStart + kSharedJobHeader + kSharedJobEntries * kSharedJobWords;
constexpr size_t kSharedHintHeader = 8;
constexpr size_t kSharedHintEntries = 32;
constexpr size_t kSharedHintTokens = 8;
constexpr size_t kSharedHintCands = 8;
constexpr size_t kSharedHintWords =
    4 + 2 * kSharedHintTokens * kSharedHintCands + 4;
constexpr size_t kSharedHintRankWords =
    kSharedHintHeader + kSharedHintEntries * kSharedHintWords;
constexpr size_t kSharedTotalWords =
    kSharedHintStart + kSharedMaxRanks * kSharedHintRankWords;

class SharedExpertCoordinator {
 public:
  SharedExpertCoordinator(uint8_t* arena, uint64_t arena_bytes,
                          uint64_t* control,
                          std::vector<SharedExpertLayer> layers, unsigned ranks,
                          unsigned max_routes, uint64_t exchange_bytes,
                          uint64_t timeout_ns)
      : control_(control),
        ranks_(ranks),
        timeout_ns_(timeout_ns),
        planner_(arena, arena_bytes, std::move(layers), ranks, max_routes,
                 exchange_bytes) {
    if (!control || reinterpret_cast<uintptr_t>(control) % 64 || ranks != 4 ||
        max_routes > kSharedMaxRoutes || !timeout_ns)
      throw std::invalid_argument("invalid shared expert control mapping");
    // Construct before sharing the descriptor or publishing any GPU requests.
    // The control mapping must cover the hit log (kSharedHitTotalWords).
    std::fill_n(control_, kSharedTotalWords, 0);
    control_[4] = max_routes;
    layer_count_ = planner_.configs().size();
    for (auto& v : last_hint_) v.resize(layer_count_);
    open_trace();
    thread_ = std::thread([this] { run(); });
  }

  ~SharedExpertCoordinator() {
    stop();
    if (trace_) std::fclose(trace_);
  }

  void stop() {
    stopped_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
    planner_.stop();
    store(control_ + 2, 1);
  }

  std::string error() {
    std::lock_guard<std::mutex> guard(error_mutex_);
    return error_;
  }

  int32_t slot(size_t layer, int32_t expert) {
    return planner_.slot(layer, expert);
  }
  uint64_t cold_offset(size_t layer, int32_t expert) {
    return planner_.cold_offset(layer, expert);
  }
  std::pair<uint64_t, uint64_t> acquire(size_t layer, int32_t expert) {
    return planner_.acquire(layer, expert);
  }
  void release(uint64_t lease) { planner_.release(lease); }

 private:
  static uint64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  static uint64_t load(const uint64_t* value) {
    return __atomic_load_n(value, __ATOMIC_ACQUIRE);
  }

  static void store(uint64_t* destination, uint64_t value) {
    __atomic_store_n(destination, value, __ATOMIC_RELEASE);
  }

  static void relax() {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
  }

  uint64_t* rank(unsigned index) {
    return control_ + kSharedRankStart + index * kSharedRankWords;
  }

  void check(uint64_t started, const char* where = "request") {
    if (stopped_.load(std::memory_order_acquire))
      throw std::runtime_error(
          "shared expert coordinator stopped during a plan");
    if (now() - started >= timeout_ns_) {
      char text[512];
      std::snprintf(
          text, sizeof(text),
          "shared expert group %s timed out: req %llu %llu %llu %llu "
          "done %llu %llu %llu %llu layer %llu %llu %llu %llu copies %llu",
          where, (unsigned long long)rank(0)[0], (unsigned long long)rank(1)[0],
          (unsigned long long)rank(2)[0], (unsigned long long)rank(3)[0],
          (unsigned long long)rank(0)[1], (unsigned long long)rank(1)[1],
          (unsigned long long)rank(2)[1], (unsigned long long)rank(3)[1],
          (unsigned long long)rank(0)[2], (unsigned long long)rank(1)[2],
          (unsigned long long)rank(2)[2], (unsigned long long)rank(3)[2],
          (unsigned long long)control_[3]);
      throw std::runtime_error(text);
    }
  }

  // Debug route trace (VLLM_V41_ROUTE_TRACE=<path prefix>): one fixed record
  // per decode layer request: kind (0 GPU-resolved hit, 1 plan), layer, route
  // count, copies, routes[36], miss bitmask over routes. Header: layer configs.
  struct TraceRecord {
    int32_t kind, layer, count, copies;
    int32_t routes[kSharedMaxRoutes];
    uint64_t miss_mask;
  };
  void open_trace() {
    const char* prefix = std::getenv("VLLM_V41_ROUTE_TRACE");
    if (!prefix || !*prefix) return;
    char path[512];
    std::snprintf(path, sizeof(path), "%s.%d.bin", prefix, int(getpid()));
    trace_ = std::fopen(path, "wb");
    if (!trace_) return;
    const auto configs = planner_.configs();
    const int32_t magic = 0x56344154, n = int32_t(configs.size());
    std::fwrite(&magic, 4, 1, trace_);
    std::fwrite(&n, 4, 1, trace_);
    for (const auto& c : configs) {
      const int32_t v[2] = {c.experts, c.resident};
      std::fwrite(v, 4, 2, trace_);
    }
    std::fflush(trace_);
  }
  void trace(int32_t kind, uint64_t layer, const int32_t* routes, size_t count,
             const std::vector<int32_t>& incoming) {
    if (!trace_) return;
    TraceRecord r{};
    r.kind = kind;
    r.layer = int32_t(layer);
    r.count = int32_t(count);
    r.copies = int32_t(incoming.size());
    for (size_t i = 0; i < count && i < kSharedMaxRoutes; ++i) {
      r.routes[i] = routes[i];
      for (auto e : incoming)
        if (e == routes[i]) r.miss_mask |= uint64_t{1} << i;
    }
    std::fwrite(&r, sizeof(r), 1, trace_);
    if (++trace_count_ % 256 == 0) std::fflush(trace_);
  }

  // Apply GPU-resolved hits to the LRU clock. A lapped reader skips to the
  // oldest entry still present; an entry is used only if its tag matches.
  void drain_hits() {
    const uint64_t head = load(control_ + kSharedHitStart);
    if (head - hits_read_ > kSharedHitEntries)
      hits_read_ = head - kSharedHitEntries;
    int32_t routes[kSharedMaxRoutes];
    while (hits_read_ < head) {
      const uint64_t* entry =
          control_ + kSharedHitStart + kSharedHitHeader +
          (hits_read_ % kSharedHitEntries) * kSharedHitWords;
      const uint64_t tag = load(entry);
      const uint64_t layer = entry[1], count = entry[2];
      if (tag == hits_read_ + 1 && count && count <= kSharedMaxRoutes) {
        for (size_t i = 0; i < count; ++i) routes[i] = int32_t(entry[3 + i]);
        if (load(entry) == tag) {
          planner_.touch(layer, routes, count);
          evaluate(layer, routes, count);
          trace(0, layer, routes, count, {});
        }
      }
      ++hits_read_;
    }
  }

  uint64_t* jobs() { return control_ + kSharedJobStart; }

  uint64_t kind_done(int kind) {
    uint64_t done = load(jobs() + 1 + 4 * kind);
    for (unsigned r = 1; r < ranks_; ++r)
      done = std::min(done, load(jobs() + 1 + 4 * kind + r));
    return done;
  }

  // Free slots return once every rank's lane completed a job.
  void drain_jobs() {
    if (!jobs_published_) return;
    for (int kind = 0; kind < 2; ++kind) {
      const uint64_t done = kind_done(kind);
      if (done > kind_completed_[kind]) {
        planner_.complete_jobs(kind, done);
        kind_completed_[kind] = done;
      }
    }
    uint64_t issued = load(jobs() + 9);
    for (unsigned r = 1; r < ranks_; ++r)
      issued = std::min(issued, load(jobs() + 9 + r));
    jobs_completed_ = issued;  // lanes copy an entry when they issue it
  }

  // A prefetch every rank has issued completes on DMA progress alone (the
  // copy streams write completion), so a bounded wait cannot deadlock and
  // is cheaper than transferring the expert again.
  void wait_prefetches(uint64_t layer, const std::vector<int32_t>& routes) {
    if (!prefetch_) return;
    const auto waits = planner_.inflight_for(layer, routes);
    if (waits.empty()) return;
    uint64_t issued = load(jobs() + 9);
    for (unsigned r = 1; r < ranks_; ++r)
      issued = std::min(issued, load(jobs() + 9 + r));
    const uint64_t begin = now();
    for (const auto& [id, ordinal] : waits) {
      if (issued <= id) continue;
      while (kind_done(1) <= ordinal && now() - begin < prefetch_wait_ns_)
        relax();
    }
    stat_wait_ns_ += now() - begin;
    ++stat_waits_;
    drain_jobs();
  }

  // Prefetch evaluation (VLLM_V41_PREFETCH_EVAL): of the candidates that
  // were cold when hinted, how many the layer's next request routes, per
  // lookahead and weight bucket.
  static int bucket(float w) {
    static const float edges[] = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    int b = 0;
    while (b < 7 && w >= edges[b]) ++b;
    return b;
  }
  void evaluate(uint64_t layer, const int32_t* routes, size_t count) {
    if (!prefetch_eval_ || layer >= layer_count_) return;
    for (int d = 0; d < 2; ++d) {
      auto& keep = last_hint_[d][layer];
      for (const auto& c : keep) {
        const bool used =
            std::find(routes, routes + count, c.e) != routes + count;
        ++eval_[d][bucket(c.w)][used];
      }
      keep.clear();
    }
  }
  void print_eval() {
    if (!prefetch_eval_) return;
    for (int d = 0; d < 2; ++d) {
      std::fprintf(stderr, "[coord] eval d%d", d + 1);
      for (int b = 0; b < 8; ++b)
        std::fprintf(stderr, " %llu/%llu", (unsigned long long)eval_[d][b][1],
                     (unsigned long long)(eval_[d][b][0] + eval_[d][b][1]));
      std::fprintf(stderr, "\n");
    }
  }

  uint64_t* hints(unsigned r) {
    return control_ + kSharedHintStart + r * kSharedHintRankWords;
  }

  // Merge the ranks' per-quarter candidates for each verified position into
  // the global top-k, weigh by position, and prefetch the best for layers
  // i+1 and i+2.
  void drain_hints() {
    if (!prefetch_) return;
    uint64_t head = load(hints(0));
    for (unsigned r = 1; r < ranks_; ++r) head = std::min(head, load(hints(r)));
    if (head - hints_read_ > kSharedHintEntries)
      hints_read_ = head - kSharedHintEntries;
    const size_t layers = layer_count_;
    while (hints_read_ < head) {
      const size_t at = kSharedHintHeader +
                        (hints_read_ % kSharedHintEntries) * kSharedHintWords;
      bool ok = true;
      for (unsigned r = 0; r < ranks_; ++r)
        ok &= load(hints(r) + at) == hints_read_ + 1;
      const uint64_t layer = hints(0)[at + 1], tokens = hints(0)[at + 2];
      const uint64_t trunc = hints(0)[at + 3];
      ++hints_read_;
      if (!ok || layer >= layers) continue;
      const size_t n = std::min<uint64_t>({tokens, trunc, kSharedHintTokens});
      for (int d = 0; d < 2; ++d) {
        const size_t target = layer + 1 + d;
        if (target >= layers) continue;
        std::unordered_map<int32_t, float> weight;
        for (size_t t = 0; t < n; ++t) {
          std::pair<float, int32_t> cand[kSharedMaxRanks * kSharedHintCands];
          size_t m = 0;
          for (unsigned r = 0; r < ranks_; ++r)
            for (size_t c = 0; c < kSharedHintCands; ++c) {
              const uint64_t w =
                  hints(r)[at + 4 +
                           (d * kSharedHintTokens + t) * kSharedHintCands + c];
              float key;
              const uint32_t bits = uint32_t(w);
              std::memcpy(&key, &bits, 4);
              cand[m++] = {key, int32_t(w >> 32)};
            }
          const size_t k = std::min<size_t>(prefetch_topk_, m);
          std::partial_sort(
              cand, cand + k, cand + m,
              [](const auto& a, const auto& b) { return a.first > b.first; });
          const float pos = 1.0f / float(1 + t);
          for (size_t c = 0; c < k; ++c)
            weight[cand[c].second] += pos * (1.0f + float(k - c) / float(k));
        }
        std::vector<std::pair<float, int32_t>> order;
        for (const auto& [e, w] : weight) order.push_back({w, e});
        std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
          return a.first > b.first;
        });
        std::vector<int32_t> ids;
        for (const auto& [w, e] : order)
          if (w >= prefetch_min_weight_) ids.push_back(e);
        if (prefetch_eval_) {
          auto& keep = last_hint_[d][target];
          keep.clear();
          for (const auto& [w, e] : order)
            if (planner_.cold(target, e)) keep.push_back({e, w});
        }
        planner_.prefetch(target, ids, prefetch_budget_, prefetch_inflight_);
      }
    }
    publish_jobs();
  }

  void publish_jobs() {
    auto list = planner_.take_jobs();
    if (list.empty()) return;
    for (const auto& job : list) {
      if (job.id != jobs_published_ ||
          job.id - jobs_completed_ >= kSharedJobEntries)
        throw std::runtime_error("shared expert job ring overflow");
      auto* entry = jobs() + kSharedJobHeader +
                    (job.id % kSharedJobEntries) * kSharedJobWords;
      entry[1] = uint64_t(job.kind);
      entry[2] = uint64_t(job.layer);
      entry[3] = uint64_t(job.slot);
      entry[4] = job.offset;
      entry[5] = uint64_t(job.expert);
      entry[6] = job.ordinal;
      store(entry, job.id + 1);
      ++jobs_published_;
    }
    store(jobs(), jobs_published_);
  }

  void run() {
    uint64_t sequence = 1;
    uint64_t active_until = 0;
    try {
      while (!stopped_.load(std::memory_order_acquire)) {
        drain_hits();
        drain_jobs();
        drain_hints();
        bool started = false;
        for (unsigned r = 0; r < ranks_; ++r) {
          const auto tag = load(rank(r));
          if (tag > sequence)
            throw std::runtime_error(
                "shared expert request skipped a sequence");
          started |= tag == sequence;
        }
        if (!started) {
          // A running decode reaches the next layer within a few ms. Avoid
          // paying a scheduler wakeup on every plan, while sleeping once the
          // group becomes idle. This lane still never submits CUDA work.
          if (now() < active_until)
            relax();
          else
            std::this_thread::sleep_for(std::chrono::microseconds(10));
          continue;
        }
        const auto begin = now();
        for (;;) {
          bool all = true;
          for (unsigned r = 0; r < ranks_; ++r) {
            const auto tag = load(rank(r));
            if (tag > sequence)
              throw std::runtime_error("shared expert ranks diverged");
            all &= tag == sequence;
          }
          if (all) break;
          check(begin, "publish");
          relax();
        }
        const uint64_t layer = rank(0)[2], count = rank(0)[3];
        if (!count || count > kSharedMaxRoutes)
          throw std::runtime_error("invalid shared expert route count");
        std::vector<int32_t> routes;
        for (size_t i = 0; i < count; ++i) {
          if (rank(0)[4 + i] > uint64_t(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error("invalid shared expert route id");
          routes.push_back(int32_t(rank(0)[4 + i]));
        }
        for (unsigned r = 1; r < ranks_; ++r) {
          if (rank(r)[2] != layer || rank(r)[3] != count ||
              !std::equal(rank(0) + 4, rank(0) + 4 + count, rank(r) + 4))
            throw std::runtime_error("shared expert rank routes differ");
        }
        evaluate(layer, routes.data(), routes.size());
        wait_prefetches(layer, routes);
        const int32_t cutoff = int32_t(load(control_ + 5));
        auto copies = planner_.plan(layer, sequence, routes, begin, timeout_ns_,
                                    cutoff > 0 && cutoff < 64 ? cutoff : 0, 6,
                                    rank(0)[41]);
        control_[6] = uint64_t(planner_.trunc());
        if (coord_debug_)
          std::fprintf(stderr,
                       "[coord] seq %llu layer %llu routes %zu copies %zu "
                       "trunc %d cutoff %d round %llu\n",
                       (unsigned long long)sequence, (unsigned long long)layer,
                       routes.size(), copies.size(), planner_.trunc(), cutoff,
                       (unsigned long long)rank(0)[41]);
        if (trace_) {
          std::vector<int32_t> incoming;
          for (const auto& c : copies) incoming.push_back(c.incoming);
          trace(1, layer, routes.data(), routes.size(), incoming);
        }
        // Demand loads by the copy engines: every rank's lane must ISSUE the
        // copies within a bound (a stalled lane falls back to the GPU's SM
        // copy, which rewrites the same bytes); once issued, completion is
        // written by the copy streams, so waiting depends on DMA alone.
        std::vector<bool> by_dma(copies.size(), false);
        if (dma_demand_) {
          const auto dj = planner_.demand_jobs();
          uint64_t last_id = 0, last_ord = 0;
          bool any = false;
          for (const auto& [id, ord] : dj)
            if (id != ~uint64_t{0}) {
              any = true;
              last_id = std::max(last_id, id);
              last_ord = std::max(last_ord, ord);
            }
          if (any) {
            publish_jobs();
            const uint64_t t0 = now();
            auto issued = [&] {
              uint64_t v = load(jobs() + 9);
              for (unsigned r = 1; r < ranks_; ++r)
                v = std::min(v, load(jobs() + 9 + r));
              return v;
            };
            while (issued() <= last_id && now() - t0 < dma_issue_ns_) relax();
            // Completion is bounded too: a lane stream that shares a hardware
            // queue with the waiting decode stream (false dependency) cannot
            // finish until the GPU moves on, so fall back to the SM copy (same
            // bytes; the late DMA is benign and its slot reuse is deferred).
            bool done = false;
            if (issued() > last_id) {
              while (kind_done(1) <= last_ord && now() - t0 < dma_done_ns_)
                relax();
              done = kind_done(1) > last_ord;
            }
            if (done) {
              for (size_t i = 0; i < dj.size(); ++i)
                by_dma[i] = dj[i].first != ~uint64_t{0};
              ++stat_dma_;
            } else {
              ++stat_dma_fallback_;
            }
            stat_dma_ns_ += now() - t0;
          }
        }
        const auto evicts = planner_.evicts();
        if (copies.size() + evicts.size() > kSharedMaxCopies)
          throw std::runtime_error("shared expert plan exceeds op capacity");
        for (size_t i = 0; i < copies.size() + evicts.size(); ++i) {
          auto* op = control_ + kSharedOpStart + i * kSharedOpWords;
          const auto& copy =
              i < copies.size() ? copies[i] : evicts[i - copies.size()];
          op[0] = uint64_t(int64_t(copy.slot));
          op[1] = uint64_t(int64_t(copy.incoming));
          op[2] = uint64_t(int64_t(copy.victim));
          op[3] =
              i < copies.size() && by_dma[i] ? uint64_t(-2) : copy.load_offset;
          op[4] = copy.save_offset;
        }
        // Low 16 bits: copies (swap kernel ops); high: evictions after them.
        control_[3] = copies.size() | (evicts.size() << 16);
        store(control_, sequence);
        uint64_t completed = 0;
        const uint64_t all_ranks = (uint64_t{1} << ranks_) - 1;
        // A plan without copies (every miss truncated) has no writeback, so
        // no rank publishes a completion for it.
        if (copies.empty() && evicts.empty()) {
          for (unsigned r = 0; r < ranks_; ++r)
            planner_.acknowledge(r, sequence, now());
          completed = all_ranks;
        }
        while (completed != all_ranks) {
          for (unsigned r = 0; r < ranks_; ++r) {
            const uint64_t tag = load(rank(r) + 1), bit = uint64_t{1} << r;
            if (tag > sequence)
              throw std::runtime_error(
                  "shared expert completion skipped a sequence");
            if (tag == sequence && !(completed & bit)) {
              planner_.acknowledge(r, sequence, now());
              completed |= bit;
            }
          }
          check(begin, "ack");
          relax();
        }
        while (!planner_.ready(sequence, now())) {
          check(begin, "ready");
          relax();
        }
        planner_.finish(sequence, now());
        {
          const auto background = planner_.take_background();
          if (!background.empty())
            stat_background_ += planner_.prefetch(
                layer, background, background.size(), bg_inflight_);
        }
        publish_jobs();
        if (plan_stats_ && sequence % 2000 == 0)
          std::fprintf(stderr,
                       "[coord] plans %llu free %llu exchange %llu rescue %llu "
                       "ghost %llu evict %llu pending %llu prefetch %llu hit "
                       "%llu waits %llu wait_ms %.1f\n",
                       (unsigned long long)sequence,
                       (unsigned long long)planner_.stat_free_,
                       (unsigned long long)planner_.stat_exchange_,
                       (unsigned long long)planner_.stat_rescue_,
                       (unsigned long long)planner_.stat_ghost_,
                       (unsigned long long)planner_.stat_evict_,
                       (unsigned long long)planner_.pending_jobs(),
                       (unsigned long long)planner_.stat_prefetch_,
                       (unsigned long long)planner_.stat_prefetch_hit_,
                       (unsigned long long)stat_waits_, stat_wait_ns_ / 1e6);
        if (plan_stats_ && sequence % 2000 == 0 && dma_demand_)
          std::fprintf(
              stderr, "[coord] dma plans %llu fallback %llu dma_ms %.1f\n",
              (unsigned long long)stat_dma_,
              (unsigned long long)stat_dma_fallback_, stat_dma_ns_ / 1e6);
        if (plan_stats_ && sequence % 2000 == 0) print_eval();
        if (plan_stats_ && sequence % 2000 == 0)
          std::fprintf(stderr,
                       "[coord] skip adopt %llu background loads %llu\n",
                       (unsigned long long)planner_.stat_adopt_,
                       (unsigned long long)stat_background_);
        store(control_ + 1, sequence);
        if (sequence == std::numeric_limits<uint64_t>::max())
          throw std::runtime_error("shared expert request sequence exhausted");
        ++sequence;
        active_until = now() + 4'000'000;
      }
    } catch (const std::exception& exception) {
      std::fprintf(stderr, "[SharedExpertCoordinator] %s\n", exception.what());
      {
        std::lock_guard<std::mutex> guard(error_mutex_);
        error_ = exception.what();
      }
      planner_.stop();
      store(control_ + 2, 1);
    }
  }

  uint64_t* const control_;
  uint64_t hits_read_ = 0;
  uint64_t jobs_published_ = 0, jobs_completed_ = 0;
  uint64_t kind_completed_[2] = {0, 0};
  uint64_t hints_read_ = 0, stat_waits_ = 0, stat_wait_ns_ = 0;
  size_t layer_count_ = 0;
  struct HintCand {
    int32_t e;
    float w;
  };
  std::vector<std::vector<HintCand>> last_hint_[2];
  uint64_t eval_[2][8][2] = {};
  const bool prefetch_eval_ = std::getenv("VLLM_V41_PREFETCH_EVAL") != nullptr;
  static double env_number(const char* name, double fallback) {
    const char* v = std::getenv(name);
    return v && *v ? std::atof(v) : fallback;
  }
  const bool prefetch_ = env_number("VLLM_V41_PREFETCH", 0) > 0;
  const bool dma_demand_ = env_number("VLLM_V41_DMA_DEMAND", 0) > 0;
  const uint64_t dma_issue_ns_ =
      uint64_t(env_number("VLLM_V41_DMA_ISSUE_US", 300) * 1000);
  const uint64_t dma_done_ns_ =
      uint64_t(env_number("VLLM_V41_DMA_DONE_US", 4000) * 1000);
  uint64_t stat_dma_ = 0, stat_dma_fallback_ = 0, stat_dma_ns_ = 0,
           stat_background_ = 0;
  const size_t bg_inflight_ = size_t(env_number("VLLM_V41_BG_INFLIGHT", 8));
  const size_t prefetch_topk_ = size_t(env_number("VLLM_V41_PREFETCH_TOPK", 8));
  const size_t prefetch_budget_ =
      size_t(env_number("VLLM_V41_PREFETCH_BUDGET", 3));
  const size_t prefetch_inflight_ =
      size_t(env_number("VLLM_V41_PREFETCH_INFLIGHT", 6));
  const float prefetch_min_weight_ =
      float(env_number("VLLM_V41_PREFETCH_MIN_WEIGHT", 1.5));
  const uint64_t prefetch_wait_ns_ =
      uint64_t(env_number("VLLM_V41_PREFETCH_WAIT_US", 3000) * 1000);
  const unsigned ranks_;
  const uint64_t timeout_ns_;
  SharedExpertPlanner planner_;
  std::atomic<bool> stopped_{false};
  const bool coord_debug_ = std::getenv("VLLM_V41_COORD_DEBUG") != nullptr;
  const bool plan_stats_ = std::getenv("VLLM_V41_PLAN_STATS") != nullptr;
  FILE* trace_ = nullptr;
  uint64_t trace_count_ = 0;
  std::thread thread_;
  std::mutex error_mutex_;
  std::string error_;
};

}  // namespace vllm::gguf
