// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// DeepSeek-V4.1 Engram row reader (lane L-ENGRAM, PORT_DESIGN §3.6 + DECISIONS D4/D5).
//
// Reads 256-byte FP8 Engram rows IN PLACE from the official safetensors shards (one random read per row; a row
// that straddles a 4 KiB page boundary costs two pages), appends each row's 8 UE8M0 scale bytes from a pinned
// table registered by Python, and writes the 264-byte result into caller-owned (pinned) staging memory.
//
// Built at runtime by torch.utils.cpp_extension.load (no CMakeLists entry; see common/engram_host.py).
//
// I/O strategy (buffered mode, the default): a job's rows are split into chunks claimed by the pool threads.
// Per chunk: (1) preadv2(RWF_NOWAIT) serves page-cache hits without blocking; (2) POSIX_FADV_WILLNEED is issued
// for every miss, which submits the page reads asynchronously (deep device queue from few threads); (3) blocking
// pread collects the misses. Low-priority prefetch jobs only issue WILLNEED (admission warm-up of the page
// cache). O_DIRECT mode bypasses the page cache (cold-path measurements): aligned bounce-buffer reads, QD =
// threads. Every I/O failure is reported with file, offset and errno; nothing is retried silently.

#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#ifndef RWF_NOWAIT
  #define RWF_NOWAIT 0x00000008
#endif

namespace {

constexpr int64_t kPage = 4096;
constexpr int64_t kDirectAlign = 4096;
constexpr int64_t kGatherChunk = 32;
constexpr int64_t kPrefetchChunk = 256;

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct Job {
  int64_t id = 0;
  bool prefetch_only = false;
  int64_t n = 0;
  int64_t row_bytes = 0;
  int64_t extra_bytes = 0;
  int64_t dst_stride = 0;
  // Index inputs are copied and dst is a raw pointer: a worker thread never owns a Python-visible tensor (whose
  // release could need the GIL). The caller (EngramHostService) owns dst and joins the pool (shutdown) first.
  std::vector<int32_t> file_v, xtab_v;
  std::vector<int64_t> off_v, xidx_v;
  const int32_t* file_idx = nullptr;
  const int64_t* offsets = nullptr;
  const int32_t* xtab = nullptr;
  const int64_t* xidx = nullptr;
  uint8_t* dst = nullptr;

  std::atomic<int64_t> next{0};
  std::atomic<int64_t> finished{0};
  std::atomic<int64_t> hits{0};
  std::atomic<int64_t> misses{0};
  std::atomic<int64_t> pages{0};
  int64_t t_submit_ns = 0;
  std::atomic<int64_t> t_done_ns{0};

  std::mutex err_mu;
  std::string error;
  std::mutex done_mu;
  std::condition_variable done_cv;
  bool done = false;
};

struct Table {
  const uint8_t* base;
  int64_t entries;
  torch::Tensor keep;
};

class RowReader {
 public:
  RowReader(const std::vector<std::string>& paths, int64_t num_threads, bool o_direct)
      : o_direct_(o_direct) {
    TORCH_CHECK(!paths.empty(), "Engram RowReader needs at least one file");
    TORCH_CHECK(num_threads >= 1 && num_threads <= 64, "Engram RowReader: num_threads must be in [1, 64], got ",
                num_threads);
    for (const auto& p : paths) {
      int flags = O_RDONLY | O_CLOEXEC;
      if (o_direct_) flags |= O_DIRECT;
      int fd = ::open(p.c_str(), flags);
      TORCH_CHECK(fd >= 0, "Engram RowReader: open(", p, ") failed: errno ", errno, " (", std::strerror(errno), ")");
      if (!o_direct_) {
        int rc = ::posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
        TORCH_CHECK(rc == 0, "Engram RowReader: posix_fadvise(RANDOM) on ", p, " failed: ", std::strerror(rc));
      }
      struct stat st;
      TORCH_CHECK(::fstat(fd, &st) == 0, "Engram RowReader: fstat(", p, ") failed: ", std::strerror(errno));
      fds_.push_back(fd);
      sizes_.push_back(static_cast<int64_t>(st.st_size));
      paths_.push_back(p);
    }
    for (int64_t i = 0; i < num_threads; ++i) {
      threads_.emplace_back([this] { worker(); });
    }
  }

  ~RowReader() { shutdown(); }

  void shutdown() {
    std::vector<std::shared_ptr<Job>> pending;
    {
      std::lock_guard<std::mutex> g(mu_);
      if (stop_) return;
      stop_ = true;
      for (auto& kv : jobs_) pending.push_back(kv.second);
    }
    cv_.notify_all();
    for (auto& t : threads_) {
      if (t.joinable()) t.join();
    }
    threads_.clear();
    for (auto& job : pending) {  // unfinished jobs fail loudly instead of leaving a wait() hanging
      std::lock_guard<std::mutex> g(job->done_mu);
      if (!job->done) {
        {
          std::lock_guard<std::mutex> e(job->err_mu);
          if (job->error.empty()) job->error = "Engram RowReader shut down before the job finished";
        }
        job->done = true;
      }
      job->done_cv.notify_all();
    }
    for (int fd : fds_) ::close(fd);
    fds_.clear();
  }

  int64_t add_table(torch::Tensor t, int64_t entry_bytes) {
    TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == at::kByte && t.is_contiguous(),
                "Engram RowReader: extra table must be a contiguous uint8 CPU tensor");
    TORCH_CHECK(entry_bytes > 0 && t.numel() % entry_bytes == 0, "Engram RowReader: table size ", t.numel(),
                " is not a multiple of entry_bytes ", entry_bytes);
    std::lock_guard<std::mutex> g(mu_);
    TORCH_CHECK(next_id_ == 1, "Engram RowReader: add_table must precede the first submit (workers read tables_ unlocked)");
    TORCH_CHECK(extra_bytes_ == 0 || extra_bytes_ == entry_bytes,
                "Engram RowReader: every extra table must use the same entry size");
    extra_bytes_ = entry_bytes;
    tables_.push_back(Table{t.data_ptr<uint8_t>(), t.numel() / entry_bytes, t});
    return static_cast<int64_t>(tables_.size()) - 1;
  }

  // Gather rows: for i in [0, n): row_bytes from files[file_idx[i]] at offsets[i] -> dst + i * dst_stride, then
  // extra_bytes from tables[xtab[i]] entry xidx[i] -> dst + i * dst_stride + row_bytes (xtab[i] < 0: no extra).
  int64_t submit(torch::Tensor file_idx, torch::Tensor offsets, torch::Tensor xtab, torch::Tensor xidx,
                 torch::Tensor dst, int64_t dst_stride, int64_t row_bytes) {
    const int64_t n = offsets.numel();
    check_index_inputs(file_idx, offsets, n, row_bytes);
    TORCH_CHECK(xtab.device().is_cpu() && xtab.scalar_type() == at::kInt && xtab.is_contiguous() &&
                    xtab.numel() == n,
                "Engram RowReader: xtab must be int32 [n] contiguous CPU");
    TORCH_CHECK(xidx.device().is_cpu() && xidx.scalar_type() == at::kLong && xidx.is_contiguous() &&
                    xidx.numel() == n,
                "Engram RowReader: xidx must be int64 [n] contiguous CPU");
    TORCH_CHECK(dst.device().is_cpu() && dst.scalar_type() == at::kByte && dst.is_contiguous(),
                "Engram RowReader: dst must be contiguous uint8 CPU memory");
    TORCH_CHECK(row_bytes > 0 && dst_stride >= row_bytes + (n ? extra_bytes_ : 0),
                "Engram RowReader: dst_stride ", dst_stride, " too small for row ", row_bytes, " + extra ",
                extra_bytes_);
    TORCH_CHECK(dst.numel() >= n * dst_stride, "Engram RowReader: dst holds ", dst.numel(), " bytes, need ",
                n * dst_stride);
    const int32_t* xt = xtab.data_ptr<int32_t>();
    const int64_t* xi = xidx.data_ptr<int64_t>();
    {
      std::lock_guard<std::mutex> g(mu_);
      for (int64_t i = 0; i < n; ++i) {
        if (xt[i] < 0) continue;
        TORCH_CHECK(xt[i] < static_cast<int32_t>(tables_.size()), "Engram RowReader: table id ", xt[i],
                    " not registered");
        TORCH_CHECK(xi[i] >= 0 && xi[i] < tables_[xt[i]].entries, "Engram RowReader: extra index ", xi[i],
                    " outside table ", xt[i], " (", tables_[xt[i]].entries, " entries)");
      }
    }
    auto job = std::make_shared<Job>();
    job->n = n;
    job->row_bytes = row_bytes;
    job->extra_bytes = extra_bytes_;
    job->dst_stride = dst_stride;
    job->file_v.assign(file_idx.data_ptr<int32_t>(), file_idx.data_ptr<int32_t>() + n);
    job->off_v.assign(offsets.data_ptr<int64_t>(), offsets.data_ptr<int64_t>() + n);
    job->xtab_v.assign(xt, xt + n);
    job->xidx_v.assign(xi, xi + n);
    job->file_idx = job->file_v.data();
    job->offsets = job->off_v.data();
    job->xtab = job->xtab_v.data();
    job->xidx = job->xidx_v.data();
    job->dst = dst.data_ptr<uint8_t>();
    return enqueue(job, /*high=*/true);
  }

  // Low priority page-cache warm-up: WILLNEED for each row's page span (no data copy).
  int64_t prefetch(torch::Tensor file_idx, torch::Tensor offsets, int64_t row_bytes) {
    const int64_t n = offsets.numel();
    check_index_inputs(file_idx, offsets, n, row_bytes);
    TORCH_CHECK(!o_direct_, "Engram RowReader: prefetch needs the page cache (buffered mode)");
    auto job = std::make_shared<Job>();
    job->prefetch_only = true;
    job->n = n;
    job->row_bytes = row_bytes;
    job->file_v.assign(file_idx.data_ptr<int32_t>(), file_idx.data_ptr<int32_t>() + n);
    job->off_v.assign(offsets.data_ptr<int64_t>(), offsets.data_ptr<int64_t>() + n);
    job->file_idx = job->file_v.data();
    job->offsets = job->off_v.data();
    return enqueue(job, /*high=*/false);
  }

  bool done(int64_t ticket) {
    auto job = find(ticket);
    std::lock_guard<std::mutex> g(job->done_mu);
    return job->done;
  }

  // Blocks (GIL released by the binding) until the job completes; returns its stats and forgets it.
  std::vector<int64_t> wait(int64_t ticket) {
    auto job = find(ticket);
    {
      std::unique_lock<std::mutex> lk(job->done_mu);
      job->done_cv.wait(lk, [&] { return job->done; });
    }
    {
      std::lock_guard<std::mutex> g(mu_);
      jobs_.erase(ticket);  // `job` still holds a reference: destruction happens outside mu_
    }
    {
      std::lock_guard<std::mutex> g(job->err_mu);
      TORCH_CHECK(job->error.empty(), "Engram row gather failed: ", job->error);
    }
    return {job->n, job->hits.load(), job->misses.load(), job->pages.load(),
            job->t_done_ns.load() - job->t_submit_ns};
  }

  int64_t pending_prefetch_rows() {
    std::lock_guard<std::mutex> g(mu_);
    int64_t rows = 0;
    for (auto& j : low_) rows += std::max<int64_t>(0, j->n - j->next.load());
    return rows;
  }

  void drop_cache(int64_t file) {
    TORCH_CHECK(file >= 0 && file < static_cast<int64_t>(fds_.size()), "Engram RowReader: bad file index");
    int rc = ::posix_fadvise(fds_[file], 0, 0, POSIX_FADV_DONTNEED);
    TORCH_CHECK(rc == 0, "posix_fadvise(DONTNEED) failed: ", std::strerror(rc));
  }

  std::vector<int64_t> file_sizes() const { return sizes_; }
  bool o_direct() const { return o_direct_; }
  int64_t num_threads() const { return static_cast<int64_t>(threads_.size()); }

 private:
  void check_index_inputs(const torch::Tensor& file_idx, const torch::Tensor& offsets, int64_t n,
                          int64_t row_bytes) {
    TORCH_CHECK(row_bytes > 0 && row_bytes <= kDirectAlign, "Engram RowReader: row_bytes must be in (0, 4096]");
    TORCH_CHECK(file_idx.device().is_cpu() && file_idx.scalar_type() == at::kInt && file_idx.is_contiguous() &&
                    file_idx.numel() == n,
                "Engram RowReader: file_idx must be int32 [n] contiguous CPU");
    TORCH_CHECK(offsets.device().is_cpu() && offsets.scalar_type() == at::kLong && offsets.is_contiguous(),
                "Engram RowReader: offsets must be int64 contiguous CPU");
    const int32_t* f = file_idx.data_ptr<int32_t>();
    const int64_t* o = offsets.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
      TORCH_CHECK(f[i] >= 0 && f[i] < static_cast<int32_t>(fds_.size()), "Engram RowReader: file index ", f[i],
                  " out of range");
      TORCH_CHECK(o[i] >= 0 && o[i] + row_bytes <= sizes_[f[i]], "Engram RowReader: row at ", o[i], " ends beyond ",
                  paths_[f[i]], " (", sizes_[f[i]], " bytes)");
    }
  }

  int64_t enqueue(std::shared_ptr<Job> job, bool high) {
    std::lock_guard<std::mutex> g(mu_);
    TORCH_CHECK(!stop_, "Engram RowReader is shut down");
    job->id = next_id_++;
    job->t_submit_ns = now_ns();
    jobs_[job->id] = job;
    if (job->n == 0) {
      job->t_done_ns = job->t_submit_ns;
      job->done = true;
      if (job->prefetch_only) jobs_.erase(job->id);
      return job->id;
    }
    (high ? high_ : low_).push_back(job);
    cv_.notify_all();
    return job->id;
  }

  std::shared_ptr<Job> find(int64_t ticket) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = jobs_.find(ticket);
    TORCH_CHECK(it != jobs_.end(), "Engram RowReader: unknown or already-waited ticket ", ticket);
    return it->second;
  }

  // Under mu_: first job with unclaimed rows, high queue first. Fully-claimed jobs leave the queues into
  // `grave`, which the caller drops AFTER unlocking (a tensor deleter may need the GIL: never under mu_).
  std::shared_ptr<Job> pick_locked(std::vector<std::shared_ptr<Job>>& grave) {
    for (auto* q : {&high_, &low_}) {
      while (!q->empty()) {
        auto& j = q->front();
        if (j->next.load() < j->n) return j;
        grave.push_back(std::move(j));
        q->pop_front();
      }
    }
    return nullptr;
  }

  void worker() {
    std::vector<int64_t> miss;
    miss.reserve(kPrefetchChunk);
    void* bounce = nullptr;
    if (o_direct_) {
      TORCH_CHECK(::posix_memalign(&bounce, kDirectAlign, 2 * kDirectAlign) == 0,
                  "Engram RowReader: posix_memalign failed");
    }
    std::vector<std::shared_ptr<Job>> grave;
    while (true) {
      std::shared_ptr<Job> job;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return stop_ || (job = pick_locked(grave)) != nullptr; });
        if (stop_) break;
      }
      grave.clear();
      const int64_t chunk = job->prefetch_only ? kPrefetchChunk : kGatherChunk;
      const int64_t start = job->next.fetch_add(chunk);
      if (start >= job->n) continue;
      const int64_t end = std::min(job->n, start + chunk);
      try {
        if (job->prefetch_only) {
          do_prefetch(*job, start, end);
        } else if (o_direct_) {
          do_direct(*job, start, end, static_cast<uint8_t*>(bounce));
        } else {
          do_buffered(*job, start, end, miss);
        }
      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> g(job->err_mu);
        if (job->error.empty()) job->error = e.what();
      }
      const int64_t fin = job->finished.fetch_add(end - start) + (end - start);
      if (fin == job->n) {
        job->t_done_ns = now_ns();
        {
          std::lock_guard<std::mutex> g(job->done_mu);
          job->done = true;
        }
        job->done_cv.notify_all();
        if (job->prefetch_only) {  // nobody waits on prefetch tickets: forget them here (job still held locally)
          std::lock_guard<std::mutex> g(mu_);
          jobs_.erase(job->id);
        }
      }
    }
    if (bounce) std::free(bounce);
  }

  static int64_t page_span(int64_t off, int64_t len) {
    return (off + len - 1) / kPage - off / kPage + 1;
  }

  void copy_extra(Job& job, int64_t i) {
    if (job.extra_bytes == 0 || job.xtab[i] < 0) return;
    const Table& t = tables_[job.xtab[i]];
    std::memcpy(job.dst + i * job.dst_stride + job.row_bytes, t.base + job.xidx[i] * job.extra_bytes,
                job.extra_bytes);
  }

  [[noreturn]] void io_fail(const Job& job, int64_t i, const char* what, ssize_t r, int err) {
    std::ostringstream os;
    os << what << " " << paths_[job.file_idx[i]] << " offset " << job.offsets[i] << " len " << job.row_bytes
       << " returned " << r << " errno " << err << " (" << std::strerror(err) << ")";
    throw std::runtime_error(os.str());
  }

  void read_full(Job& job, int64_t i) {
    const int fd = fds_[job.file_idx[i]];
    uint8_t* out = job.dst + i * job.dst_stride;
    int64_t got = 0;
    while (got < job.row_bytes) {
      ssize_t r = ::pread(fd, out + got, job.row_bytes - got, job.offsets[i] + got);
      if (r < 0 && errno == EINTR) continue;
      if (r <= 0) io_fail(job, i, "pread", r, r < 0 ? errno : EIO);
      got += r;
    }
  }

  void do_buffered(Job& job, int64_t start, int64_t end, std::vector<int64_t>& miss) {
    miss.clear();
    int64_t pages = 0;
    for (int64_t i = start; i < end; ++i) {
      pages += page_span(job.offsets[i], job.row_bytes);
      bool hit = false;
      if (nowait_ok_.load(std::memory_order_relaxed)) {
        struct iovec iov{job.dst + i * job.dst_stride, static_cast<size_t>(job.row_bytes)};
        ssize_t r = ::preadv2(fds_[job.file_idx[i]], &iov, 1, job.offsets[i], RWF_NOWAIT);
        if (r == job.row_bytes) {
          hit = true;
        } else if (r < 0 && (errno == EOPNOTSUPP || errno == EINVAL || errno == ENOSYS)) {
          nowait_ok_.store(false);  // kernel/fs lacks RWF_NOWAIT: every row takes the blocking path (counted)
        } else if (r < 0 && errno != EAGAIN && errno != EINTR) {
          io_fail(job, i, "preadv2(RWF_NOWAIT)", r, errno);
        }
      }
      if (hit) {
        copy_extra(job, i);
      } else {
        miss.push_back(i);
      }
    }
    for (int64_t i : miss) {  // submit every miss to the device before blocking on any of them
      const int64_t off = job.offsets[i];
      const int64_t a = off / kPage * kPage;
      const int64_t b = (off + job.row_bytes + kPage - 1) / kPage * kPage;
      ::posix_fadvise(fds_[job.file_idx[i]], a, b - a, POSIX_FADV_WILLNEED);  // advisory; pread below is authoritative
    }
    for (int64_t i : miss) {
      read_full(job, i);
      copy_extra(job, i);
    }
    job.hits.fetch_add(static_cast<int64_t>((end - start) - miss.size()));
    job.misses.fetch_add(static_cast<int64_t>(miss.size()));
    job.pages.fetch_add(pages);
  }

  void do_direct(Job& job, int64_t start, int64_t end, uint8_t* bounce) {
    // 4 KiB alignment is valid for 512 B and 4 KiB logical-block devices alike (measurement mode only).
    int64_t pages = 0;
    for (int64_t i = start; i < end; ++i) {
      const int64_t off = job.offsets[i];
      const int64_t a = off / kDirectAlign * kDirectAlign;
      const int64_t b = (off + job.row_bytes + kDirectAlign - 1) / kDirectAlign * kDirectAlign;
      const int64_t need = off + job.row_bytes - a;   // a row at the end of the file: the read may stop at EOF
      TORCH_CHECK(b - a <= 2 * kDirectAlign, "Engram RowReader: O_DIRECT span too large");
      int64_t got = 0;
      while (got < need) {
        ssize_t r = ::pread(fds_[job.file_idx[i]], bounce + got, b - a - got, a + got);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) io_fail(job, i, "pread(O_DIRECT)", r, r < 0 ? errno : EIO);
        got += r;
      }
      std::memcpy(job.dst + i * job.dst_stride, bounce + (off - a), job.row_bytes);
      copy_extra(job, i);
      pages += page_span(off, job.row_bytes);
    }
    job.misses.fetch_add(end - start);
    job.pages.fetch_add(pages);
  }

  void do_prefetch(Job& job, int64_t start, int64_t end) {
    int64_t pages = 0;
    for (int64_t i = start; i < end; ++i) {
      const int64_t off = job.offsets[i];
      const int64_t a = off / kPage * kPage;
      const int64_t b = (off + job.row_bytes + kPage - 1) / kPage * kPage;
      if (::posix_fadvise(fds_[job.file_idx[i]], a, b - a, POSIX_FADV_WILLNEED) == 0) {
        pages += (b - a) / kPage;   // only advice the kernel accepted counts as warm-up
      } else {
        job.misses.fetch_add(1);    // reported as failed prefetch rows (stats only; the step gather is authoritative)
      }
    }
    job.pages.fetch_add(pages);
  }

  bool o_direct_;
  std::vector<int> fds_;
  std::vector<int64_t> sizes_;
  std::vector<std::string> paths_;
  std::vector<Table> tables_;
  int64_t extra_bytes_ = 0;
  std::vector<std::thread> threads_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Job>> high_, low_;
  std::unordered_map<int64_t, std::shared_ptr<Job>> jobs_;
  int64_t next_id_ = 1;
  bool stop_ = false;
  std::atomic<bool> nowait_ok_{true};
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  py::class_<RowReader>(m, "RowReader")
      .def(py::init<const std::vector<std::string>&, int64_t, bool>(), py::arg("paths"), py::arg("num_threads"),
           py::arg("o_direct"))
      .def("add_table", &RowReader::add_table, py::arg("table"), py::arg("entry_bytes"))
      .def("submit", &RowReader::submit, py::arg("file_idx"), py::arg("offsets"), py::arg("xtab"),
           py::arg("xidx"), py::arg("dst"), py::arg("dst_stride"), py::arg("row_bytes"),
           py::call_guard<py::gil_scoped_release>())
      .def("prefetch", &RowReader::prefetch, py::arg("file_idx"), py::arg("offsets"), py::arg("row_bytes"),
           py::call_guard<py::gil_scoped_release>())
      .def("done", &RowReader::done, py::arg("ticket"))
      .def("wait", &RowReader::wait, py::arg("ticket"), py::call_guard<py::gil_scoped_release>())
      .def("pending_prefetch_rows", &RowReader::pending_prefetch_rows)
      .def("drop_cache", &RowReader::drop_cache, py::arg("file"))
      .def("file_sizes", &RowReader::file_sizes)
      .def("o_direct", &RowReader::o_direct)
      .def("num_threads", &RowReader::num_threads)
      .def("shutdown", &RowReader::shutdown, py::call_guard<py::gil_scoped_release>());
}
