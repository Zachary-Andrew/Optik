#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

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

static void load_dataset(const char* path, std::size_t k,
                          std::vector<tpoptoa::KmerWord>& keys,
                          std::vector<tpoptoa::KmerWord>& queries) {
    std::unordered_map<tpoptoa::KmerWord, bool> seen;
    queries.clear();

    tpoptoa::iterate_sequences(path, [&](const tpoptoa::SeqRecord& rec) {
        tpoptoa::extract_kmers(rec.seq, k, [&](tpoptoa::KmerWord kmer) {
            if (kmer == tpoptoa::EMPTY_KEY || kmer == tpoptoa::TOMBSTONE) return;
            seen.emplace(kmer, true);
            queries.push_back(kmer);
        });
    });

    keys.clear();
    keys.reserve(seen.size());
    for (const auto& [kw, _] : seen) keys.push_back(kw);
}

// Pipelined query: prefetch i+L while processing i.
template <typename Index>
static double timed_query_pipelined(const Index& idx,
                                     const std::vector<tpoptoa::KmerWord>& qs) {
    const std::size_t Q = qs.size();
    const std::size_t L = tpoptoa::PREFETCH_LOOKAHEAD;
    volatile uint64_t sink = 0;

    for (std::size_t i = 0; i < Q; ++i) {
        if (i + L < Q) idx.prefetch_hint(qs[i + L]);
        const auto* v = idx.find(qs[i]);
        if (v) sink ^= static_cast<uint64_t>(*v);
    }

    auto t0 = Clock::now();
    for (std::size_t i = 0; i < Q; ++i) {
        if (i + L < Q) idx.prefetch_hint(qs[i + L]);
        const auto* v = idx.find(qs[i]);
        if (v) sink ^= static_cast<uint64_t>(*v);
    }
    (void)sink;
    return elapsed_ns(t0) / static_cast<double>(Q);
}

struct TrialOut { double build_s, rss_mb, idx_mb, qns, llc, slot_bits; };
struct CellStats {
    std::string method;
    std::size_t n_keys = 0;
    std::vector<double> build_s, rss_mb, idx_mb, qns, llc, slot_bits;
    double tp_bits_actual = 0.0, tp_bits_bound = 0.0;
};

static void record(CellStats& c, const TrialOut& r) {
    c.build_s  .push_back(r.build_s);  c.rss_mb   .push_back(r.rss_mb);
    c.idx_mb   .push_back(r.idx_mb);   c.qns      .push_back(r.qns);
    c.llc      .push_back(r.llc);      c.slot_bits.push_back(r.slot_bits);
}

static TrialOut run_tpoptoa(const std::vector<tpoptoa::KmerWord>& keys,
                             const std::vector<tpoptoa::KmerWord>& queries,
                             std::size_t k, double dram_ns, CellStats& cell) {
    const std::size_t n = keys.size();
    tpoptoa::RssHighWaterMark hwm;

    auto t0 = Clock::now();
    tpoptoa::TinyPointerIndex<uint32_t> idx(k, n);
    for (std::size_t i = 0; i < n; ++i)
        idx.stage(keys[i], static_cast<uint32_t>(i));
    idx.build();
    double bs = elapsed_sec(t0);
    hwm.sample();

    if (cell.tp_bits_actual == 0.0 && n > 0) {
        cell.tp_bits_actual = static_cast<double>(tpoptoa::TINY_BITS);
        double nd = static_cast<double>(n);
        double log3n = (nd > 8.0) ? std::log2(std::log2(std::log2(nd))) : 1.0;
        cell.tp_bits_bound = std::ceil(log3n + std::log2(4.0));
    }

    TrialOut r;
    r.build_s   = bs;
    r.rss_mb    = static_cast<double>(hwm.net_bytes()) / (1024.0 * 1024.0);
    r.idx_mb    = static_cast<double>(idx.bytes())     / (1024.0 * 1024.0);
    r.slot_bits = idx.bits_per_entry();
    r.qns       = timed_query_pipelined(idx, queries);
    r.llc       = tpoptoa::estimate_llc_misses_per_1k(r.qns, dram_ns);
    return r;
}

static TrialOut run_stdopen(const std::vector<tpoptoa::KmerWord>& keys,
                             const std::vector<tpoptoa::KmerWord>& queries,
                             double dram_ns) {
    const std::size_t n = keys.size();
    tpoptoa::RssHighWaterMark hwm;

    auto t0 = Clock::now();
    tpoptoa::ElasticHashTable<uint32_t> ht(static_cast<std::size_t>(n * 1.5 + 64));
    for (std::size_t i = 0; i < n; ++i) {
        bool ok = ht.insert(keys[i], static_cast<uint32_t>(i));
        (void)ok;
    }
    double bs = elapsed_sec(t0);
    hwm.sample();

    TrialOut r;
    r.build_s   = bs;
    r.rss_mb    = static_cast<double>(hwm.net_bytes()) / (1024.0 * 1024.0);
    r.idx_mb    = static_cast<double>(ht.bytes())      / (1024.0 * 1024.0);
    r.slot_bits = ht.bits_per_entry();
    r.qns       = timed_query_pipelined(ht, queries);
    r.llc       = tpoptoa::estimate_llc_misses_per_1k(r.qns, dram_ns);
    return r;
}

static TrialOut run_stdunordered(const std::vector<tpoptoa::KmerWord>& keys,
                                  const std::vector<tpoptoa::KmerWord>& queries,
                                  double dram_ns) {
    const std::size_t n = keys.size();
    tpoptoa::RssHighWaterMark hwm;

    auto t0 = Clock::now();
    std::unordered_map<tpoptoa::KmerWord, uint32_t> umap;
    umap.reserve(static_cast<std::size_t>(n * 1.5));
    for (std::size_t i = 0; i < n; ++i)
        umap.emplace(keys[i], static_cast<uint32_t>(i));
    double bs = elapsed_sec(t0);
    hwm.sample();

    double idx_mb = static_cast<double>(
        umap.bucket_count() * sizeof(void*) +
        umap.size() * (sizeof(tpoptoa::KmerWord) + sizeof(uint32_t) + sizeof(void*))
    ) / (1024.0 * 1024.0);

    volatile uint64_t sink = 0;
    for (tpoptoa::KmerWord q : queries) {
        auto it = umap.find(q); if (it != umap.end()) sink ^= it->second;
    }
    auto tq = Clock::now();
    for (tpoptoa::KmerWord q : queries) {
        auto it = umap.find(q); if (it != umap.end()) sink ^= it->second;
    }
    (void)sink;

    TrialOut r;
    r.build_s   = bs;
    r.rss_mb    = static_cast<double>(hwm.net_bytes()) / (1024.0 * 1024.0);
    r.idx_mb    = idx_mb;
    r.slot_bits = (umap.size() > 0)
                  ? static_cast<double>(
                        (umap.bucket_count() * sizeof(void*) +
                         umap.size() * (sizeof(tpoptoa::KmerWord) + sizeof(uint32_t) + sizeof(void*))) * 8)
                    / static_cast<double>(umap.size())
                  : 0.0;
    r.qns       = elapsed_ns(tq) / static_cast<double>(queries.size());
    r.llc       = tpoptoa::estimate_llc_misses_per_1k(r.qns, dram_ns);
    return r;
}

static void print_table(FILE* out, const CellStats& tp, const CellStats& so,
                         const CellStats& su, std::size_t n_trials) {
    using namespace tpoptoa::stats;
    std::fprintf(out,
        "  %-20s  %14s  %14s  %14s\n",
        "", "TP+OptOA", "StdOpen", "StdUnordered");
    std::fprintf(out,
        "  %-20s  %14s  %14s  %14s\n",
        "", "avg (sd)", "avg (sd)", "avg (sd)");
    std::fprintf(out, "  %s\n", std::string(68, '-').c_str());

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
    row("build time",  tp.build_s,  so.build_s,  su.build_s,  "s");
    row("build RSS",   tp.rss_mb,   so.rss_mb,   su.rss_mb,   "MB");
    row("index size",  tp.idx_mb,   so.idx_mb,   su.idx_mb,   "MB");
    row("slot bits",   tp.slot_bits,so.slot_bits,su.slot_bits,"bits/entry");
    row("query time",  tp.qns,      so.qns,      su.qns,      "ns/query");
    row("LLC miss/1k", tp.llc,      so.llc,      su.llc,      "est.");
    std::fprintf(out,
        "\n  %zu trials, genomic query order, prefetch lookahead=%zu\n",
        n_trials, tpoptoa::PREFETCH_LOOKAHEAD);
}

static void print_tp_report(FILE* out, const CellStats& tp, std::size_t n) {
    double saving = tp.tp_bits_actual > 0.0 ? 32.0 / tp.tp_bits_actual : 0.0;
    std::fprintf(out,
        "  Tiny-pointer space (%zu k-mers)\n"
        "    bits per pointer  actual=%.0f  bound=%.0f  within_bound=%s\n"
        "    TinyPointerArray  %.3f MB  vs naive 32-bit: %.3f MB  (%.1fx saving)\n",
        n,
        tp.tp_bits_actual, tp.tp_bits_bound,
        (tp.tp_bits_actual <= tp.tp_bits_bound + 2.0) ? "yes" : "NO",
        (tp.tp_bits_actual * static_cast<double>(n)) / (8.0 * 1024.0 * 1024.0),
        (32.0 * static_cast<double>(n)) / (8.0 * 1024.0 * 1024.0),
        saving);
}

static void print_sig(FILE* out, const CellStats& tp, const CellStats& cmp) {
    using namespace tpoptoa::stats;
    std::fprintf(out, "  TP+OptOA vs %s\n", cmp.method.c_str());
    std::fprintf(out,
        "    %-20s  %8s  %6s  %8s  %8s  %10s  verdict\n",
        "metric", "t-stat", "df", "p", "d", "|d|");
    std::fprintf(out, "    %s\n", std::string(76, '-').c_str());

    struct { const char* n; const std::vector<double>* a; const std::vector<double>* b; }
    ms[] = {
        { "build time (s)",  &tp.build_s,   &cmp.build_s  },
        { "build RSS (MB)",  &tp.rss_mb,    &cmp.rss_mb   },
        { "index size (MB)", &tp.idx_mb,    &cmp.idx_mb   },
        { "slot bits",       &tp.slot_bits, &cmp.slot_bits},
        { "query time (ns)", &tp.qns,       &cmp.qns      },
        { "LLC miss/1k",     &tp.llc,       &cmp.llc      },
    };
    for (const auto& m : ms) {
        auto r = welch_t_test(*m.a, *m.b);
        bool ok = (r.p_value < 0.05) && (std::fabs(r.cohens_d_val) >= 0.5);
        std::fprintf(out,
            "    %-20s  %8.3f  %6.1f  %8.4f  %8.3f  %10s  %s%s\n",
            m.n, r.t_stat, r.df, r.p_value, r.cohens_d_val,
            cohens_label(r.cohens_d_val), sig_stars(r.p_value),
            ok ? "  MEANINGFUL" : "");
    }
    std::fprintf(out, "\n");
}

static void write_tsv(const char* path,
                       const CellStats& tp, const CellStats& so, const CellStats& su) {
    FILE* f = std::fopen(path, "w");
    if (!f) { std::fprintf(stderr, "Cannot open %s\n", path); return; }
    std::fprintf(f, "method\tn_keys\ttrial\tbuild_sec\trss_mb\tidx_mb\tslot_bits\tquery_ns\tllc\n");
    for (const auto* c : {&tp, &so, &su}) {
        for (std::size_t t = 0; t < c->build_s.size(); ++t) {
            std::fprintf(f, "%s\t%zu\t%zu\t%.6f\t%.3f\t%.3f\t%.2f\t%.4f\t%.4f\n",
                c->method.c_str(), c->n_keys, t+1,
                c->build_s[t], c->rss_mb[t], c->idx_mb[t],
                c->slot_bits[t], c->qns[t], c->llc[t]);
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
        "  -k <int>   k-mer length                  (default: 31)\n"
        "  -o <path>  TSV output path               (default: benchmark.tsv)\n"
        "  -h         this help\n\n"
        "Examples:\n"
        "  %s -i ecoli.fa -n 30\n"
        "  cat chr1.fa | %s -i - -k 31 -n 20\n",
        p, p, p);
}

int main(int argc, char** argv) {
    const char* input   = nullptr;
    std::size_t n_trials = 30;
    std::size_t k        = 31;
    const char* tsv_path = "benchmark.tsv";

    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "-i") == 0 && i+1<argc) input    = argv[++i];
        else if (std::strcmp(argv[i], "-n") == 0 && i+1<argc) n_trials = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-k") == 0 && i+1<argc) k        = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-o") == 0 && i+1<argc) tsv_path = argv[++i];
        else if (std::strcmp(argv[i], "-h") == 0) { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (!input) {
        std::fprintf(stderr, "Error: -i <file|-> is required.\n");
        usage(argv[0]); return 1;
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

    std::fprintf(stderr, "[bench] measuring DRAM latency...\n");
    double dram_ns = tpoptoa::pointer_chase_latency_ns(32 * 1024 * 1024);
    std::fprintf(stderr, "[bench] DRAM baseline: %.1f ns\n\n", dram_ns);

    enum { TP=0, SO=1, SU=2 };
    CellStats cells[3];
    cells[TP].method = "TP+OptOA";     cells[TP].n_keys = keys.size();
    cells[SO].method = "StdOpen";      cells[SO].n_keys = keys.size();
    cells[SU].method = "StdUnordered"; cells[SU].n_keys = keys.size();

    for (std::size_t t = 0; t < n_trials; ++t) {
        TrialOut ra = run_tpoptoa    (keys, queries, k, dram_ns, cells[TP]);
        TrialOut rb = run_stdopen    (keys, queries,    dram_ns);
        TrialOut rc = run_stdunordered(keys, queries,   dram_ns);
        record(cells[TP], ra); record(cells[SO], rb); record(cells[SU], rc);
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

    std::fprintf(stdout, "\n  -- Tiny-Pointer Space --\n");
    print_tp_report(stdout, cells[TP], n);

    std::fprintf(stdout, "\n  -- Significance (MEANINGFUL = p<0.05 AND |d|>=0.5) --\n\n");
    print_sig(stdout, cells[TP], cells[SO]);
    print_sig(stdout, cells[TP], cells[SU]);

    write_tsv(tsv_path, cells[TP], cells[SO], cells[SU]);
    return 0;
}
