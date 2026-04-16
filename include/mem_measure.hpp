#pragma once

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace tpoptoa {

// Reads VmRSS from /proc/self/status (Linux). Returns kB.
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

// RAII: records RSS before and after an operation, reports peak delta.
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

// Measures DRAM latency via random pointer chasing in a large buffer.
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

// Approximates LLC misses per 1000 queries from measured query latency.
// Assumes L1 hit = 4 ns, DRAM = measured, linear interpolation.
inline double estimate_llc_misses_per_1k(double query_ns, double dram_latency_ns) {
    constexpr double L1_NS = 4.0;
    if (dram_latency_ns <= L1_NS) return 0.0;
    double frac = (query_ns - L1_NS) / (dram_latency_ns - L1_NS);
    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;
    return frac * 1000.0;
}

} // namespace tpoptoa
