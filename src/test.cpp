#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../include/kmer.hpp"
#include "../include/kmer_extractor.hpp"
#include "../include/fasta_reader.hpp"
#include "../include/tiny_pointer.hpp"
#include "../include/elastic_hash.hpp"
#include "../include/tp_index.hpp"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond) \
    do { \
        ++g_pass; \
        if (!(cond)) { --g_pass; ++g_fail; \
            std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        } \
    } while (0)

static void test_kmer_encoding() {
    using namespace tpoptoa;

    KmerWord w = 0;
    CHECK(encode_kmer("ACGT", 4, w));
    CHECK(w == 0b00011011);
    CHECK(decode_kmer(w, 4) == "ACGT");
    CHECK(!encode_kmer("ACNT", 4, w));

    CHECK(reverse_complement(0b00011011, 4) == 0b00011011);

    KmerWord fwd = 0; encode_kmer("AAAA", 4, fwd);
    KmerWord rc  = reverse_complement(fwd, 4);
    CHECK(canonical(fwd, 4) == std::min(fwd, rc));

    KmerWord rolled = roll_kmer(0b00011011, base_encode('A'), kmer_mask(4));
    CHECK(decode_kmer(rolled, 4) == "CGTA");
}

static void test_kmer_extraction() {
    using namespace tpoptoa;

    std::vector<KmerWord> v;
    extract_kmers("ACGTACGTACGT", 4, [&](KmerWord km){ v.push_back(km); });
    CHECK(v.size() == 9);

    v.clear();
    extract_kmers("ACG", 4, [&](KmerWord km){ v.push_back(km); });
    CHECK(v.empty());

    v.clear();
    extract_kmers("ACGTNACGT", 4, [&](KmerWord km){ v.push_back(km); });
    CHECK(v.size() == 2);
}

static void test_tiny_pointer_array() {
    using namespace tpoptoa;

    const std::size_t N = 2000;
    TinyPointerArray arr(N);
    std::mt19937_64 rng(42);
    std::vector<uint64_t> vals(N);

    for (std::size_t i = 0; i < N; ++i) {
        vals[i] = rng() % TINY_BLOCK;
        arr.set(i, vals[i]);
    }
    for (std::size_t i = 0; i < N; ++i)
        CHECK(arr.get(i) == vals[i]);

    CHECK(TinyPointerArray::block_base(0)  == 0);
    CHECK(TinyPointerArray::block_base(63) == 0);
    CHECK(TinyPointerArray::block_base(64) == 64);

    for (std::size_t i = 0; i < N; ++i)
        CHECK(arr.full_index(i) == TinyPointerArray::block_base(i) + vals[i]);

    std::size_t expected_bytes = (N * TINY_BITS + 63) / 64 * 8;
    CHECK(arr.bytes() == expected_bytes);

    std::fprintf(stderr,
        "  tiny_pointer: %u bits/entry  %.4f MB for %zu entries"
        "  (vs 32-bit: %.4f MB  —  %.1fx saving)\n",
        TINY_BITS,
        static_cast<double>(arr.bytes()) / (1024.0 * 1024.0), N,
        static_cast<double>(N * 4) / (1024.0 * 1024.0),
        32.0 / static_cast<double>(TINY_BITS));
}

static void test_elastic_hash() {
    using namespace tpoptoa;

    const std::size_t N = 50000;
    ElasticHashTable<uint32_t> ht(static_cast<std::size_t>(N * 1.5));

    std::unordered_map<uint64_t, uint32_t> ref;
    ref.reserve(N);
    std::mt19937_64 rng(123);

    for (std::size_t i = 0; i < N; ++i) {
        uint64_t key = rng();
        while (key == EMPTY_KEY || key == TOMBSTONE) key = rng();
        if (ref.count(key)) continue;
        ref[key] = static_cast<uint32_t>(i);
        CHECK(ht.insert(key, static_cast<uint32_t>(i)));
    }
    for (const auto& [key, val] : ref) {
        const uint32_t* p = ht.find(key);
        CHECK(p != nullptr);
        if (p) CHECK(*p == val);
    }
    if (ref.count(0) == 0) CHECK(ht.find(0) == nullptr);
}

static void test_tp_index_synthetic() {
    using namespace tpoptoa;

    const std::size_t K = 31;
    const std::size_t N = 10000;

    KmerWord mask = kmer_mask(K);
    std::mt19937_64 rng(999);
    std::unordered_map<KmerWord, uint32_t> ref;
    ref.reserve(N);

    while (ref.size() < N) {
        KmerWord can = canonical(rng() & mask, K);
        if (can == EMPTY_KEY || can == TOMBSTONE) continue;
        ref.emplace(can, static_cast<uint32_t>(ref.size() + 1));
    }

    TinyPointerIndex<uint32_t> idx(K, ref.size());
    for (const auto& [kmer, cnt] : ref) idx.stage(kmer, cnt);
    idx.build();
    CHECK(idx.size() == ref.size());

    std::size_t hits = 0;
    for (const auto& [kmer, cnt] : ref) {
        const uint32_t* v = idx.find(kmer);
        if (v && *v == cnt) ++hits;
    }
    CHECK(hits == ref.size());

    double nd     = static_cast<double>(N);
    double log3n  = (nd > 8.0) ? std::log2(std::log2(std::log2(nd))) : 1.0;
    double bound  = std::ceil(log3n + std::log2(4.0));
    bool   within = (static_cast<double>(TINY_BITS) <= bound + 2.0);
    std::fprintf(stderr,
        "  tp_index synthetic: %zu k-mers  %.2f MB  tiny-ptr %u bits  bound %.0f  %s\n",
        idx.size(), static_cast<double>(idx.bytes()) / (1024.0 * 1024.0),
        TINY_BITS, bound, within ? "OK" : "EXCEEDS BOUND");
}

static void test_real_file(const char* path, std::size_t k) {
    using namespace tpoptoa;
    std::fprintf(stderr, "=== real-file round-trip: %s (k=%zu) ===\n", path, k);

    std::unordered_map<KmerWord, uint32_t> counts;
    std::vector<KmerWord> queries;

    try {
        iterate_sequences(path, [&](const SeqRecord& rec) {
            extract_kmers(rec.seq, k, [&](KmerWord kmer) {
                ++counts[kmer];
                queries.push_back(kmer);
            });
        });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  Error: %s\n", e.what()); CHECK(false); return;
    }
    if (counts.empty()) { std::fprintf(stderr, "  No k-mers.\n"); CHECK(false); return; }
    std::fprintf(stderr, "  %zu distinct k-mers, %zu queries\n",
                 counts.size(), queries.size());

    TinyPointerIndex<uint32_t> idx(k, counts.size());
    for (const auto& [kmer, cnt] : counts) idx.stage(kmer, cnt);
    idx.build();
    CHECK(idx.size() == counts.size());

    std::size_t correct = 0;
    for (const auto& [kmer, cnt] : counts) {
        const uint32_t* v = idx.find(kmer);
        if (v && *v == cnt) ++correct;
    }
    CHECK(correct == counts.size());

    if (counts.size() < 50000000) {
        std::mt19937_64 rng(0xBEEF);
        KmerWord mask = kmer_mask(k);
        std::size_t absent_ok = 0, absent_total = 0;
        for (int a = 0; a < 5000 && absent_total < 20; ++a) {
            KmerWord probe = canonical(rng() & mask, k);
            if (counts.count(probe) == 0) {
                ++absent_total;
                if (idx.find(probe) == nullptr) ++absent_ok;
            }
        }
        if (absent_total > 0) CHECK(absent_ok == absent_total);
    }

    double nd    = static_cast<double>(counts.size());
    double log3n = (nd > 8.0) ? std::log2(std::log2(std::log2(nd))) : 1.0;
    double bound = std::ceil(log3n + std::log2(4.0));
    std::fprintf(stderr,
        "  tp_index: %.2f MB total  tiny-ptr %u bits/entry  bound %.0f  %s\n",
        static_cast<double>(idx.bytes()) / (1024.0 * 1024.0),
        TINY_BITS, bound,
        (static_cast<double>(TINY_BITS) <= bound + 2.0) ? "OK" : "EXCEEDS BOUND");

    const std::size_t L = tpoptoa::PREFETCH_LOOKAHEAD;
    const std::size_t Q = queries.size();
    volatile uint64_t sink = 0;
    for (std::size_t i = 0; i < Q; ++i) {
        if (i + L < Q) idx.prefetch_hint(queries[i + L]);
        const uint32_t* v = idx.find(queries[i]); if (v) sink ^= *v;
    }
    auto tq = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < Q; ++i) {
        if (i + L < Q) idx.prefetch_hint(queries[i + L]);
        const uint32_t* v = idx.find(queries[i]); if (v) sink ^= *v;
    }
    double q_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - tq).count();
    (void)sink;
    std::fprintf(stderr,
        "  throughput: %.1fns/query  (%zu queries, genomic order)\n",
        q_s * 1e9 / static_cast<double>(Q), Q);
}

int main(int argc, char** argv) {
    const char* input_file = nullptr;
    std::size_t k = 31;

    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "-i") == 0 && i+1<argc) input_file = argv[++i];
        else if (std::strcmp(argv[i], "-k") == 0 && i+1<argc) k          = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-h") == 0) {
            std::fprintf(stderr,
                "Usage: %s -i <file|->  [-k klen]\n"
                "  -i  FASTA/FASTQ file (required; - for stdin)\n"
                "  -k  k-mer length (default: 31)\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]); return 1;
        }
    }
    if (!input_file) {
        std::fprintf(stderr, "Error: -i <file|-> is required.\n"
            "Usage: %s -i genome.fa [-k 31]\n"
            "       cat genome.fa | %s -i -\n", argv[0], argv[0]);
        return 1;
    }

    std::fprintf(stderr, "=== kmer encoding ===\n");      test_kmer_encoding();
    std::fprintf(stderr, "=== kmer extraction ===\n");    test_kmer_extraction();
    std::fprintf(stderr, "=== TinyPointerArray ===\n");   test_tiny_pointer_array();
    std::fprintf(stderr, "=== ElasticHashTable ===\n");   test_elastic_hash();
    std::fprintf(stderr, "=== TinyPointerIndex ===\n");   test_tp_index_synthetic();
    test_real_file(input_file, k);

    if (g_fail == 0)
        std::fprintf(stderr, "\nAll %d tests passed.\n", g_pass);
    else
        std::fprintf(stderr, "\n%d / %d tests FAILED.\n", g_fail, g_pass + g_fail);

    return g_fail == 0 ? 0 : 1;
}
