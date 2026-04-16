#pragma once

// Portable memory and cache-pressure measurement utilities.
//
// Peak RSS: we poll /proc/self/status at build start and end. Linux updates
// VmRSS continuously so this gives a good approximation of the net allocation.
//
// LLC miss proxy: since hardware performance counters aren't available in
// container environments, we estimate cache pressure from query latency using
// a two-point model. Random pointer-chasing through a buffer larger than L3
// gives the full-miss latency (DRAM_NS). An L1-resident access costs ~4 ns.
// For a measured query latency Q_NS, the estimated miss fraction is:
//   miss_frac = (Q_NS - L1_NS) / (DRAM_NS - L1_NS)  clamped to [0, 1]
// LLC misses per 1k queries = miss_frac * 1000.
//
// This is a rough model but correctly ranks structures: chaining-based tables
// with pointer-scattered nodes will show higher miss fractions than open-
// addressing tables where the probed slots are physically adjacent.

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace tpoptoa {

inline std::size_t current_rss_kb() {
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    std::size_t rss = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "VmRSS:", 6) == 0) {
            std::sscanf(line + 6, "%zu", &rss);
            break;
        }
    }
    std::fclose(f);
    return rss;
}

// RAII helper: record RSS before and after a build, report the delta.
class RssHighWaterMark {
public:
    RssHighWaterMark() : before_kb_(current_rss_kb()), peak_kb_(before_kb_) {}

    void sample() {
        std::size_t now = current_rss_kb();
        if (now > peak_kb_) peak_kb_ = now;
    }

    std::size_t peak_bytes() const noexcept { return peak_kb_ * 1024; }
    std::size_t net_bytes()  const noexcept {
        return (peak_kb_ > before_kb_) ? (peak_kb_ - before_kb_) * 1024 : 0;
    }

private:
    std::size_t before_kb_;
    std::size_t peak_kb_;
};

// Measure the average pointer-chase latency through a buffer that doesn't fit
// in L3. Returns nanoseconds per access. Used as the DRAM_NS baseline for the
// LLC miss proxy calculation.
inline double pointer_chase_latency_ns(std::size_t buffer_bytes = 64 * 1024 * 1024) {
    struct alignas(64) Node { std::size_t next; char pad[56]; };

    std::size_t n = buffer_bytes / sizeof(Node);
    if (n < 1024) n = 1024;

    std::vector<Node> nodes(n);
    std::mt19937 rng(42);
    for (std::size_t i = 0; i < n; ++i) nodes[i].next = i;
    for (std::size_t i = n - 1; i > 0; --i) {
        std::size_t j = rng() % (i + 1);
        std::swap(nodes[i].next, nodes[j].next);
    }

    // One warm-up pass to map all pages.
    volatile std::size_t idx = 0;
    for (std::size_t i = 0; i < n; ++i) idx = nodes[idx].next;

    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t r = 0; r < 4; ++r)
        for (std::size_t i = 0; i < n; ++i)
            idx = nodes[idx].next;
    auto t1 = std::chrono::steady_clock::now();
    (void)idx;

    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return ns / static_cast<double>(n * 4);
}

// Estimate LLC misses per 1000 queries from measured query latency.
// Returns a value in [0, 1000]; never negative.
inline double estimate_llc_misses_per_1k(double query_ns, double dram_latency_ns) {
    constexpr double L1_NS = 4.0;
    if (dram_latency_ns <= L1_NS) return 0.0;
    double frac = (query_ns - L1_NS) / (dram_latency_ns - L1_NS);
    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;
    return frac * 1000.0;
}

} // namespace tpoptoa
