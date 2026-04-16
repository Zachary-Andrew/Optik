#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../include/kmer.hpp"
#include "../include/kmer_extractor.hpp"
#include "../include/fasta_reader.hpp"
#include "../include/tp_index.hpp"

static const uint64_t INDEX_MAGIC = UINT64_C(0x54504F50544F4100);

static double wall_sec() {
    using C = std::chrono::steady_clock;
    static auto t0 = C::now();
    return std::chrono::duration<double>(C::now() - t0).count();
}

static void usage(const char* p) {
    std::fprintf(stderr,
        "Usage: %s -i <fasta/fastq> [-k <len>] [-o <index.bin>]\n"
        "  -i  input genome file (FASTA or FASTQ, required)\n"
        "  -k  k-mer length 1..32 (default: 31)\n"
        "  -o  output index path (default: index.bin)\n",
        p);
}

int main(int argc, char** argv) {
    const char* input  = nullptr;
    const char* output = "index.bin";
    std::size_t k      = 31;

    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "-i") == 0 && i+1<argc) input  = argv[++i];
        else if (std::strcmp(argv[i], "-k") == 0 && i+1<argc) k      = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-o") == 0 && i+1<argc) output = argv[++i];
        else if (std::strcmp(argv[i], "-h") == 0) { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (!input) { std::fprintf(stderr, "Error: -i is required.\n"); usage(argv[0]); return 1; }
    if (k < 1 || k > tpoptoa::MAX_K) {
        std::fprintf(stderr, "Error: k must be in [1, %zu].\n", tpoptoa::MAX_K);
        return 1;
    }

    // First pass: count occurrences of each k‑mer.
    std::fprintf(stderr, "[build] pass 1: counting k-mers (k=%zu) from %s\n", k, input);
    std::unordered_map<tpoptoa::KmerWord, uint32_t> counts;
    counts.reserve(1 << 22);

    try {
        tpoptoa::iterate_sequences(input, [&](const tpoptoa::SeqRecord& rec) {
            tpoptoa::extract_kmers(rec.seq, k, [&](tpoptoa::KmerWord kmer) {
                ++counts[kmer];
            });
        });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what()); return 1;
    }

    std::size_t n = counts.size();
    std::fprintf(stderr, "[build] pass 1 done in %.2fs — %zu distinct k-mers\n",
                 wall_sec(), n);
    if (n == 0) { std::fprintf(stderr, "No k-mers found.\n"); return 0; }

    // Second pass: build the TP+OptOA index.
    std::fprintf(stderr, "[build] pass 2: building index...\n");
    tpoptoa::TinyPointerIndex<uint32_t> idx(k, n);
    for (const auto& [kmer, cnt] : counts)
        idx.stage(kmer, cnt);
    counts.clear();
    { decltype(counts) tmp; tmp.swap(counts); }

    idx.build();
    std::fprintf(stderr, "[build] pass 2 done in %.2fs — index %.2f MB\n",
                 wall_sec(), static_cast<double>(idx.bytes()) / (1024.0 * 1024.0));

    // Write binary index: magic, k, n, then (kmer, count) pairs.
    FILE* out = std::fopen(output, "wb");
    if (!out) { std::fprintf(stderr, "Cannot open %s\n", output); return 1; }

    uint64_t k64 = k, n64 = n;
    std::fwrite(&INDEX_MAGIC, 8, 1, out);
    std::fwrite(&k64, 8, 1, out);
    std::fwrite(&n64, 8, 1, out);

    try {
        tpoptoa::iterate_sequences(input, [&](const tpoptoa::SeqRecord& rec) {
            tpoptoa::extract_kmers(rec.seq, k, [&](tpoptoa::KmerWord kmer) {
                const uint32_t* v = idx.find(kmer);
                if (!v) return;
                uint64_t kw = kmer;
                std::fwrite(&kw, 8, 1, out);
                std::fwrite(v, 4, 1, out);
            });
        });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what()); std::fclose(out); return 1;
    }

    std::fclose(out);
    std::fprintf(stderr, "[build] wrote %s in %.2fs\n", output, wall_sec());
    return 0;
}
