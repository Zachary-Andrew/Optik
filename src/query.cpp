#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "../include/kmer.hpp"
#include "../include/kmer_extractor.hpp"
#include "../include/fasta_reader.hpp"

static const uint64_t INDEX_MAGIC = UINT64_C(0x54504F50544F4100);

static std::unordered_map<tpoptoa::KmerWord, uint32_t>
load_index(const char* path, std::size_t& k_out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("cannot open: ") + path);

    uint64_t magic = 0, k64 = 0, n64 = 0;
    if (std::fread(&magic, 8, 1, f) != 1 ||
        std::fread(&k64,   8, 1, f) != 1 ||
        std::fread(&n64,   8, 1, f) != 1)
        throw std::runtime_error("truncated index header");
    if (magic != INDEX_MAGIC)
        throw std::runtime_error("bad magic — rebuild index with tpoptoa_build");

    k_out = static_cast<std::size_t>(k64);
    std::unordered_map<tpoptoa::KmerWord, uint32_t> map;
    map.reserve(static_cast<std::size_t>(n64 * 1.3));

    uint64_t kw = 0; uint32_t cnt = 0;
    while (std::fread(&kw, 8, 1, f) == 1 && std::fread(&cnt, 4, 1, f) == 1)
        map.emplace(kw, cnt);

    std::fclose(f);
    return map;
}

static void usage(const char* p) {
    std::fprintf(stderr,
        "Usage: %s -x <index.bin> -q <fasta/fastq> [-k <len>] [-a]\n"
        "  -x  index file from tpoptoa_build (required)\n"
        "  -q  query FASTA or FASTQ file (required)\n"
        "  -k  override k-mer length (default: read from index)\n"
        "  -a  suppress absent k-mers (only print found ones)\n",
        p);
}

int main(int argc, char** argv) {
    const char* index_path = nullptr;
    const char* query_path = nullptr;
    std::size_t k_override = 0;
    bool only_found = false;

    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "-x") == 0 && i+1<argc) index_path = argv[++i];
        else if (std::strcmp(argv[i], "-q") == 0 && i+1<argc) query_path = argv[++i];
        else if (std::strcmp(argv[i], "-k") == 0 && i+1<argc) k_override = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-a") == 0)             only_found = true;
        else if (std::strcmp(argv[i], "-h") == 0) { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "Unknown: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (!index_path || !query_path) {
        std::fprintf(stderr, "Error: -x and -q are required.\n");
        usage(argv[0]); return 1;
    }

    std::fprintf(stderr, "[query] loading %s ...\n", index_path);
    auto t0 = std::chrono::steady_clock::now();
    std::size_t k = 0;
    std::unordered_map<tpoptoa::KmerWord, uint32_t> idx;
    try { idx = load_index(index_path, k); }
    catch (const std::exception& e) { std::fprintf(stderr, "Error: %s\n", e.what()); return 1; }
    if (k_override) k = k_override;

    double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "[query] loaded %zu entries (k=%zu) in %.2fs\n",
                 idx.size(), k, load_s);

    uint64_t n_queried = 0, n_found = 0;
    auto tq = std::chrono::steady_clock::now();

    try {
        tpoptoa::iterate_sequences(query_path, [&](const tpoptoa::SeqRecord& rec) {
            tpoptoa::extract_kmers(rec.seq, k, [&](tpoptoa::KmerWord kmer) {
                ++n_queried;
                auto it = idx.find(kmer);
                uint32_t cnt = (it != idx.end()) ? it->second : 0;
                if (it != idx.end()) ++n_found;
                if (!only_found || cnt > 0) {
                    std::printf("%s\t%u\n", tpoptoa::decode_kmer(kmer, k).c_str(), cnt);
                }
            });
        });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what()); return 1;
    }

    double q_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tq).count();
    std::fprintf(stderr,
        "[query] %zu queries in %.3fs  %.1f ns/query  %zu found (%.1f%%)\n",
        n_queried, q_s,
        n_queried > 0 ? q_s * 1e9 / static_cast<double>(n_queried) : 0.0,
        n_found,
        n_queried > 0 ? 100.0 * static_cast<double>(n_found) / static_cast<double>(n_queried) : 0.0);

    return 0;
}
