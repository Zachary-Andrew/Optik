#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/ioctl.h>
#include <unistd.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <errno.h>

#include "../include/kmer.hpp"
#include "../include/kmer_extractor.hpp"
#include "../include/fasta_reader.hpp"
#include "../include/tiny_pointer.hpp"
#include "../include/elastic_hash.hpp"
#include "../include/tp_index.hpp"
#include "../include/stats.hpp"
#include "../include/mem_measure.hpp"

using Clock = std::chrono::steady_clock;

static inline double elapsed_sec(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

static inline double elapsed_ns(Clock::time_point t0) {
    return std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
}

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
    return syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

struct PerfCounters {
    int fd_l1_misses;    // L1 data cache misses
    int fd_llc_misses;   // LLC/L3 cache misses (generic cache-misses)
};

static PerfCounters init_perf_counters() {
    PerfCounters pc = {-1, -1};
    struct perf_event_attr pe;
    
    // L1 data cache read misses (works everywhere)
    memset(&pe, 0, sizeof(pe));
    pe.type = PERF_TYPE_HW_CACHE;
    pe.size = sizeof(pe);
    pe.config = (PERF_COUNT_HW_CACHE_L1D) |
                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    pc.fd_l1_misses = perf_event_open(&pe, 0, -1, -1, 0);
    
    // Generic cache misses (LLC/L3 on most CPUs, works on Intel and AMD)
    memset(&pe, 0, sizeof(pe));
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(pe);
    pe.config = PERF_COUNT_HW_CACHE_MISSES;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    pc.fd_llc_misses = perf_event_open(&pe, 0, -1, -1, 0);
    
    return pc;
}

static void start_counters(PerfCounters& pc) {
    if (pc.fd_l1_misses >= 0) {
        ioctl(pc.fd_l1_misses, PERF_EVENT_IOC_RESET, 0);
        ioctl(pc.fd_l1_misses, PERF_EVENT_IOC_ENABLE, 0);
    }
    if (pc.fd_llc_misses >= 0) {
        ioctl(pc.fd_llc_misses, PERF_EVENT_IOC_RESET, 0);
        ioctl(pc.fd_llc_misses, PERF_EVENT_IOC_ENABLE, 0);
    }
}

static void stop_counters(PerfCounters& pc) {
    if (pc.fd_l1_misses >= 0) ioctl(pc.fd_l1_misses, PERF_EVENT_IOC_DISABLE, 0);
    if (pc.fd_llc_misses >= 0) ioctl(pc.fd_llc_misses, PERF_EVENT_IOC_DISABLE, 0);
}

static void read_counters(PerfCounters& pc, long long& l1_misses, long long& llc_misses) {
    l1_misses = llc_misses = -1;
    if (pc.fd_l1_misses >= 0) read(pc.fd_l1_misses, &l1_misses, sizeof(long long));
    if (pc.fd_llc_misses >= 0) read(pc.fd_llc_misses, &llc_misses, sizeof(long long));
}

static void close_counters(PerfCounters& pc) {
    if (pc.fd_l1_misses >= 0) close(pc.fd_l1_misses);
    if (pc.fd_llc_misses >= 0) close(pc.fd_llc_misses);
}

static bool check_perf_counters() {
    PerfCounters pc = init_perf_counters();
    bool ok = (pc.fd_llc_misses >= 0);
    close_counters(pc);
    return ok;
}

// Load k-mers exactly as build.cpp does: one pass counting, then extract keys.
// Returns the distinct keys and all genomic-order queries (for timing).
static void load_dataset(const char* path, std::size_t k,
                          std::vector<tpoptoa::KmerWord>& keys,
                          std::vector<tpoptoa::KmerWord>& queries) {
    std::unordered_map<tpoptoa::KmerWord, uint32_t> counts;
    queries.clear();

    tpoptoa::iterate_sequences(path, [&](const tpoptoa::SeqRecord& rec) {
        tpoptoa::extract_kmers(rec.seq, k, [&](tpoptoa::KmerWord kmer) {
            // Same sentinel filtering as build.cpp
            if (kmer == tpoptoa::EMPTY_KEY || kmer == tpoptoa::TOMBSTONE) return;
            ++counts[kmer];
            queries.push_back(kmer);
        });
    });

    keys.clear();
    keys.reserve(counts.size());
    for (const auto& [kw, _] : counts) keys.push_back(kw);
}

// Build exactly as build.cpp does: stage all pairs, then build.
static double timed_build_tpoptoa(const std::vector<tpoptoa::KmerWord>& keys,
                                   std::size_t k,
                                   tpoptoa::RssHighWaterMark& hwm,
                                   tpoptoa::TinyPointerIndex<uint32_t>& idx_out) {
    const std::size_t n = keys.size();
    
    auto t0 = Clock::now();
    tpoptoa::TinyPointerIndex<uint32_t> idx(k, n);
    for (std::size_t i = 0; i < n; ++i)
        idx.stage(keys[i], static_cast<uint32_t>(i + 1));  // count = i+1 as placeholder
    idx.build();
    double bs = elapsed_sec(t0);
    hwm.sample();
    
    idx_out = std::move(idx);
    return bs;
}

static double timed_build_stdopen(const std::vector<tpoptoa::KmerWord>& keys,
                                   tpoptoa::RssHighWaterMark& hwm,
                                   tpoptoa::ElasticHashTable<uint32_t>& ht_out) {
    const std::size_t n = keys.size();
    
    auto t0 = Clock::now();
    tpoptoa::ElasticHashTable<uint32_t> ht(n * 3 / 2 + 64);
    for (std::size_t i = 0; i < n; ++i)
        ht.insert(keys[i], static_cast<uint32_t>(i + 1));
    double bs = elapsed_sec(t0);
    hwm.sample();
    
    ht_out = std::move(ht);
    return bs;
}

static double timed_build_stdunordered(const std::vector<tpoptoa::KmerWord>& keys,
                                        tpoptoa::RssHighWaterMark& hwm,
                                        std::unordered_map<tpoptoa::KmerWord, uint32_t>& umap_out) {
    const std::size_t n = keys.size();
    
    auto t0 = Clock::now();
    std::unordered_map<tpoptoa::KmerWord, uint32_t> umap;
    umap.reserve(n * 3 / 2 + 64);
    for (std::size_t i = 0; i < n; ++i)
        umap[keys[i]] = static_cast<uint32_t>(i + 1);
    double bs = elapsed_sec(t0);
    hwm.sample();
    
    umap_out = std::move(umap);
    return bs;
}

// Query exactly as query.cpp does: prefetch pipeline with lookahead.
template <typename Index>
static double timed_query_pipelined(const Index& idx,
                                     const std::vector<tpoptoa::KmerWord>& qs,
                                     long long& l1_out, long long& llc_out) {
    const std::size_t Q = qs.size();
    const std::size_t L = tpoptoa::PREFETCH_LOOKAHEAD;
    volatile uint64_t sink = 0;
    
    // Warmup
    for (std::size_t i = 0; i < Q; ++i) {
        if (i + L < Q) idx.prefetch_hint(qs[i + L]);
        const auto* v = idx.find(qs[i]);
        if (v) sink ^= static_cast<uint64_t>(*v);
    }
    
    PerfCounters pc = init_perf_counters();
    start_counters(pc);
    
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < Q; ++i) {
        if (i + L < Q) idx.prefetch_hint(qs[i + L]);
        const auto* v = idx.find(qs[i]);
        if (v) sink ^= static_cast<uint64_t>(*v);
    }
    double elapsed = elapsed_ns(t0) / static_cast<double>(Q);
    
    stop_counters(pc);
    read_counters(pc, l1_out, llc_out);
    close_counters(pc);
    
    (void)sink;
    return elapsed;
}

// std::unordered_map doesn't have prefetch, so query loop is simpler.
static double timed_query_umap(const std::unordered_map<tpoptoa::KmerWord, uint32_t>& umap,
                                const std::vector<tpoptoa::KmerWord>& qs,
                                long long& l1_out, long long& llc_out) {
    const std::size_t Q = qs.size();
    volatile uint64_t sink = 0;
    
    // Warmup
    for (const auto& q : qs) {
        auto it = umap.find(q);
        if (it != umap.end()) sink ^= it->second;
    }
    
    PerfCounters pc = init_perf_counters();
    start_counters(pc);
    
    auto t0 = Clock::now();
    for (const auto& q : qs) {
        auto it = umap.find(q);
        if (it != umap.end()) sink ^= it->second;
    }
    double elapsed = elapsed_ns(t0) / static_cast<double>(Q);
    
    stop_counters(pc);
    read_counters(pc, l1_out, llc_out);
    close_counters(pc);
    
    (void)sink;
    return elapsed;
}

struct TrialOut { 
    double build_s;
    double rss_mb;
    double idx_mb;
    double qns;
    long long l1_misses;
    long long llc_misses;
};

struct CellStats {
    std::string method;
    std::size_t n_keys = 0;
    std::vector<double> build_s;
    std::vector<double> rss_mb;
    std::vector<double> idx_mb;
    std::vector<double> qns;
    std::vector<long long> l1_misses;
    std::vector<long long> llc_misses;
};

static void record(CellStats& c, const TrialOut& r) {
    c.build_s.push_back(r.build_s);
    c.rss_mb.push_back(r.rss_mb);
    c.idx_mb.push_back(r.idx_mb);
    c.qns.push_back(r.qns);
    c.l1_misses.push_back(r.l1_misses);
    c.llc_misses.push_back(r.llc_misses);
}

static TrialOut run_tpoptoa(const std::vector<tpoptoa::KmerWord>& keys,
                             const std::vector<tpoptoa::KmerWord>& queries,
                             std::size_t k) {
    const std::size_t n = keys.size();
    tpoptoa::RssHighWaterMark hwm;

    tpoptoa::TinyPointerIndex<uint32_t> idx(0, 0);
    double bs = timed_build_tpoptoa(keys, k, hwm, idx);

    long long l1_misses, llc_misses;
    double qns = timed_query_pipelined(idx, queries, l1_misses, llc_misses);

    TrialOut r;
    r.build_s = bs;
    r.rss_mb = static_cast<double>(hwm.net_bytes()) / (1024.0 * 1024.0);
    r.idx_mb = static_cast<double>(idx.bytes()) / (1024.0 * 1024.0);
    r.qns = qns;
    r.l1_misses = l1_misses;
    r.llc_misses = llc_misses;
    return r;
}

static TrialOut run_stdopen(const std::vector<tpoptoa::KmerWord>& keys,
                             const std::vector<tpoptoa::KmerWord>& queries) {
    const std::size_t n = keys.size();
    tpoptoa::RssHighWaterMark hwm;

    tpoptoa::ElasticHashTable<uint32_t> ht(0);
    double bs = timed_build_stdopen(keys, hwm, ht);

    long long l1_misses, llc_misses;
    double qns = timed_query_pipelined(ht, queries, l1_misses, llc_misses);

    TrialOut r;
    r.build_s = bs;
    r.rss_mb = static_cast<double>(hwm.net_bytes()) / (1024.0 * 1024.0);
    r.idx_mb = static_cast<double>(ht.bytes()) / (1024.0 * 1024.0);
    r.qns = qns;
    r.l1_misses = l1_misses;
    r.llc_misses = llc_misses;
    return r;
}

static TrialOut run_stdunordered(const std::vector<tpoptoa::KmerWord>& keys,
                                  const std::vector<tpoptoa::KmerWord>& queries) {
    const std::size_t n = keys.size();
    tpoptoa::RssHighWaterMark hwm;

    std::unordered_map<tpoptoa::KmerWord, uint32_t> umap;
    double bs = timed_build_stdunordered(keys, hwm, umap);

    long long l1_misses, llc_misses;
    double qns = timed_query_umap(umap, queries, l1_misses, llc_misses);

    // Estimate memory for unordered_map
    double idx_mb = static_cast<double>(
        umap.bucket_count() * sizeof(void*) +
        umap.size() * (sizeof(tpoptoa::KmerWord) + sizeof(uint32_t) + sizeof(void*))
    ) / (1024.0 * 1024.0);

    TrialOut r;
    r.build_s = bs;
    r.rss_mb = static_cast<double>(hwm.net_bytes()) / (1024.0 * 1024.0);
    r.idx_mb = idx_mb;
    r.qns = qns;
    r.l1_misses = l1_misses;
    r.llc_misses = llc_misses;
    return r;
}

static void print_table(FILE* out, const CellStats& tp, const CellStats& so,
                         const CellStats& su, std::size_t n_trials) {
    using namespace tpoptoa::stats;
    
    auto row = [&](const char* label,
                   const std::vector<double>& a,
                   const std::vector<double>& b,
                   const std::vector<double>& c,
                   const char* unit) {
        std::fprintf(out,
            "  %-20s  %8.3f (%5.3f)  %8.3f (%5.3f)  %8.3f (%5.3f)  %s\n",
            label,
            mean(a), stddev(a), mean(b), stddev(b), mean(c), stddev(c), unit);
    };
    
    std::fprintf(out, "\n");
    std::fprintf(out,
        "  %-20s  %14s  %14s  %14s\n",
        "", "TP+OptOA", "StdOpen", "StdUnordered");
    std::fprintf(out,
        "  %-20s  %14s  %14s  %14s\n",
        "", "avg (sd)", "avg (sd)", "avg (sd)");
    std::fprintf(out, "  %s\n", std::string(68, '-').c_str());

    row("build time",  tp.build_s,  so.build_s,  su.build_s,  "s");
    row("build RSS",   tp.rss_mb,   so.rss_mb,   su.rss_mb,   "MB");
    row("index size",  tp.idx_mb,   so.idx_mb,   su.idx_mb,   "MB");
    row("query time",  tp.qns,      so.qns,      su.qns,      "ns/query");
    std::fprintf(out,
        "\n  %zu trials, genomic query order, prefetch lookahead=%zu\n",
        n_trials, tpoptoa::PREFETCH_LOOKAHEAD);
}

static void print_cache_table(FILE* out, const CellStats& tp, const CellStats& so,
                               const CellStats& su) {
    using namespace tpoptoa::stats;
    
    auto to_double = [](const std::vector<long long>& v) {
        std::vector<double> d(v.size());
        for (size_t i = 0; i < v.size(); ++i) d[i] = static_cast<double>(v[i]);
        return d;
    };
    
    std::vector<double> tp_l1 = to_double(tp.l1_misses);
    std::vector<double> so_l1 = to_double(so.l1_misses);
    std::vector<double> su_l1 = to_double(su.l1_misses);
    
    std::vector<double> tp_llc = to_double(tp.llc_misses);
    std::vector<double> so_llc = to_double(so.llc_misses);
    std::vector<double> su_llc = to_double(su.llc_misses);
    
    std::fprintf(out, "\n  -- Cache Misses (per query phase) --\n");
    std::fprintf(out,
        "  %-20s  %14s  %14s  %14s\n",
        "", "TP+OptOA", "StdOpen", "StdUnordered");
    std::fprintf(out,
        "  %-20s  %14s  %14s  %14s\n",
        "", "avg (sd)", "avg (sd)", "avg (sd)");
    std::fprintf(out, "  %s\n", std::string(68, '-').c_str());
    
    std::fprintf(out,
        "  %-20s  %8.0f (%5.0f)  %8.0f (%5.0f)  %8.0f (%5.0f)  %s\n",
        "L1 misses",
        mean(tp_l1), stddev(tp_l1), mean(so_l1), stddev(so_l1),
        mean(su_l1), stddev(su_l1), "count");
    
    std::fprintf(out,
        "  %-20s  %8.0f (%5.0f)  %8.0f (%5.0f)  %8.0f (%5.0f)  %s\n",
        "LLC misses",
        mean(tp_llc), stddev(tp_llc), mean(so_llc), stddev(so_llc),
        mean(su_llc), stddev(su_llc), "count");
}

static void write_tsv(const char* path,
                       const CellStats& tp, const CellStats& so, const CellStats& su) {
    FILE* f = std::fopen(path, "w");
    if (!f) { std::fprintf(stderr, "Cannot open %s\n", path); return; }
    std::fprintf(f, "method\tn_keys\ttrial\tbuild_sec\trss_mb\tidx_mb\tquery_ns\tl1_misses\tllc_misses\n");
    for (const auto* c : {&tp, &so, &su}) {
        for (std::size_t t = 0; t < c->build_s.size(); ++t) {
            std::fprintf(f, "%s\t%zu\t%zu\t%.6f\t%.3f\t%.3f\t%.4f\t%lld\t%lld\n",
                c->method.c_str(), c->n_keys, t+1,
                c->build_s[t], c->rss_mb[t], c->idx_mb[t],
                c->qns[t],
                c->l1_misses[t], c->llc_misses[t]);
        }
    }
    std::fclose(f);
    std::fprintf(stderr, "[bench] raw data written to %s\n", path);
}

static void usage(const char* p) {
    std::fprintf(stderr,
        "Usage: %s -i <file|-> [options]\n\n"
        "  -i <path>  FASTA/FASTQ input (required; use - to read from stdin)\n"
        "  -n <int>   independent trials per method  (default: 30)\n"
        "  -k <int>   k-mer length 1..%zu            (default: 31)\n"
        "  -o <path>  TSV output path               (default: benchmark.tsv)\n"
        "  -h         this help\n\n"
        "Note: For hardware cache counters, run:\n"
        "  sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'\n",
        p, tpoptoa::MAX_K);
}

int main(int argc, char** argv) {
    const char* input = nullptr;
    std::size_t n_trials = 30;
    std::size_t k = 31;
    const char* tsv_path = "benchmark.tsv";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-i") == 0 && i+1 < argc) input = argv[++i];
        else if (std::strcmp(argv[i], "-n") == 0 && i+1 < argc) n_trials = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-k") == 0 && i+1 < argc) k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-o") == 0 && i+1 < argc) tsv_path = argv[++i];
        else if (std::strcmp(argv[i], "-h") == 0) { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (!input) {
        std::fprintf(stderr, "Error: -i <file|-> is required.\n");
        usage(argv[0]); return 1;
    }

    // Validate k before touching any k-mer machinery.
    if (k < 1 || k > tpoptoa::MAX_K) {
        std::fprintf(stderr,
            "Error: k must be in [1, %zu].\n",
            tpoptoa::MAX_K);
        return 1;
    }

    // Check if perf counters are available
    if (!check_perf_counters()) {
        std::fprintf(stderr, "Warning: cache-misses counter not available.\n");
        std::fprintf(stderr, "Run: sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'\n\n");
    }

    std::fprintf(stderr,
        "[bench] input=%s  n_trials=%zu  k=%zu  lookahead=%zu  out=%s\n",
        input, n_trials, k, tpoptoa::PREFETCH_LOOKAHEAD, tsv_path);

    std::fprintf(stderr, "[bench] loading k-mers...\n");
    std::vector<tpoptoa::KmerWord> keys, queries;
    try { load_dataset(input, k, keys, queries); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what()); return 1;
    }
    if (keys.empty()) {
        std::fprintf(stderr, "Error: no valid k-mers extracted.\n"); return 1;
    }
    std::fprintf(stderr, "[bench] %zu distinct k-mers, %zu queries\n",
                 keys.size(), queries.size());

    enum { TP = 0, SO = 1, SU = 2 };
    CellStats cells[3];
    cells[TP].method = "TP+OptOA";
    cells[TP].n_keys = keys.size();
    cells[SO].method = "StdOpen";
    cells[SO].n_keys = keys.size();
    cells[SU].method = "StdUnordered";
    cells[SU].n_keys = keys.size();

    for (std::size_t t = 0; t < n_trials; ++t) {
        TrialOut ra = run_tpoptoa(keys, queries, k);
        TrialOut rb = run_stdopen(keys, queries);
        TrialOut rc = run_stdunordered(keys, queries);
        record(cells[TP], ra);
        record(cells[SO], rb);
        record(cells[SU], rc);
        std::fprintf(stderr,
            "[bench]   trial %2zu/%zu  TP=%5.1fns  SO=%5.1fns  SU=%5.1fns\n",
            t+1, n_trials, ra.qns, rb.qns, rc.qns);
    }

    const std::size_t n = keys.size();
    std::fprintf(stdout,
        "\n"
        "======================================================================\n"
        "  RESULTS  %s  (%zu k-mers, %zu trials)\n"
        "======================================================================\n\n",
        input, n, n_trials);

    print_table(stdout, cells[TP], cells[SO], cells[SU], n_trials);
    print_cache_table(stdout, cells[TP], cells[SO], cells[SU]);

    write_tsv(tsv_path, cells[TP], cells[SO], cells[SU]);
    return 0;
}
