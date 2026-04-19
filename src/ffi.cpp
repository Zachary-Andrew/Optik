#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>
#include <unordered_map>

#include "../include/kmer.hpp"
#include "../include/kmer_extractor.hpp"
#include "../include/minimizer.hpp"
#include "../include/fasta_reader.hpp"
#include "../include/tp_index.hpp"
#include "../include/elastic_hash.hpp"
#include "../include/myers_align.hpp"

extern "C" {

// TP+OptOA
struct OkIndex { tpoptoa::TinyPointerIndex<uint32_t>* p; };

OkIndex* okidx_new(uint64_t k, uint64_t n) {
    auto* h = new OkIndex;
    h->p = new tpoptoa::TinyPointerIndex<uint32_t>(
        static_cast<std::size_t>(k),
        static_cast<std::size_t>(n));
    return h;
}

void okidx_free(OkIndex* h) { delete h->p; delete h; }
void okidx_stage(OkIndex* h, uint64_t kmer, uint32_t val) { h->p->stage(kmer, val); }
void okidx_build(OkIndex* h) { h->p->build(); }

uint32_t okidx_find(const OkIndex* h, uint64_t kmer) {
    const uint32_t* v = h->p->find(kmer);
    return v ? *v : UINT32_MAX;
}

void okidx_prefetch(const OkIndex* h, uint64_t kmer) { h->p->prefetch_hint(kmer); }
uint64_t okidx_size(const OkIndex* h) { return h->p->size(); }
uint64_t okidx_bytes(const OkIndex* h) { return h->p->bytes(); }
double okidx_bits_per_entry(const OkIndex* h) { return h->p->bits_per_entry(); }

// Open addressing (no tiny pointers)
struct OkOpenAddr { tpoptoa::ElasticHashTable<uint32_t>* p; };

OkOpenAddr* okopen_new(uint64_t capacity) {
    auto* h = new OkOpenAddr;
    h->p = new tpoptoa::ElasticHashTable<uint32_t>(static_cast<std::size_t>(capacity));
    return h;
}

void okopen_free(OkOpenAddr* h) { delete h->p; delete h; }
void okopen_insert(OkOpenAddr* h, uint64_t kmer, uint32_t val) { h->p->insert(kmer, val); }

uint32_t okopen_find(const OkOpenAddr* h, uint64_t kmer) {
    const uint32_t* v = h->p->find(kmer);
    return v ? *v : UINT32_MAX;
}

void okopen_prefetch(const OkOpenAddr* h, uint64_t kmer) { h->p->prefetch_hint(kmer); }
uint64_t okopen_size(const OkOpenAddr* h) { return h->p->size(); }
uint64_t okopen_bytes(const OkOpenAddr* h) { return h->p->bytes(); }

double okopen_bits_per_entry(const OkOpenAddr* h) {
    return h->p->size() > 0 ? (h->p->bytes() * 8.0) / h->p->size() : 0.0;
}

// C++ std unordered_map
struct OkUnorderedMap { std::unordered_map<uint64_t, uint32_t>* p; };

OkUnorderedMap* okumap_new(uint64_t capacity) {
    auto* h = new OkUnorderedMap;
    h->p = new std::unordered_map<uint64_t, uint32_t>();
    h->p->reserve(static_cast<std::size_t>(capacity));
    return h;
}

void okumap_free(OkUnorderedMap* h) { delete h->p; delete h; }
void okumap_insert(OkUnorderedMap* h, uint64_t kmer, uint32_t val) { (*h->p)[kmer] = val; }

uint32_t okumap_find(const OkUnorderedMap* h, uint64_t kmer) {
    auto it = h->p->find(kmer);
    return it != h->p->end() ? it->second : UINT32_MAX;
}

uint64_t okumap_size(const OkUnorderedMap* h) { return h->p->size(); }

uint64_t okumap_bytes(const OkUnorderedMap* h) {
    size_t bucket_bytes = h->p->bucket_count() * sizeof(void*);
    size_t entry_bytes = h->p->size() * (sizeof(uint64_t) + sizeof(uint32_t) + sizeof(void*));
    return bucket_bytes + entry_bytes;
}

double okumap_bits_per_entry(const OkUnorderedMap* h) {
    return h->p->size() > 0 ? (okumap_bytes(h) * 8.0) / h->p->size() : 0.0;
}

// K-mer helpers
uint64_t ok_canonical(uint64_t kmer, uint64_t k) {
    return tpoptoa::canonical(kmer, static_cast<std::size_t>(k));
}

uint64_t ok_kmer_mask(uint64_t k) {
    return tpoptoa::kmer_mask(static_cast<std::size_t>(k));
}

int ok_encode_kmer(const char* seq, uint64_t k, uint64_t* out) {
    tpoptoa::KmerWord w = 0;
    bool ok = tpoptoa::encode_kmer(seq, static_cast<std::size_t>(k), w);
    *out = w;
    return ok ? 1 : 0;
}

// FASTA streaming
typedef void (*SeqCb)(void*, const char*, uint64_t, const char*, uint64_t);

int ok_iterate_fasta(const char* path, SeqCb cb, void* cookie,
                     char* errbuf, uint64_t errbuf_len) {
    try {
        tpoptoa::iterate_sequences(path, [&](const tpoptoa::SeqRecord& r) {
            cb(cookie,
               r.name.data(), (uint64_t)r.name.size(),
               r.seq.data(),  (uint64_t)r.seq.size());
        });
        return 0;
    } catch (const std::exception& e) {
        if (errbuf && errbuf_len)
            std::strncpy(errbuf, e.what(), errbuf_len - 1);
        return -1;
    }
}

// K-mer extraction
typedef void (*KmerCb)(void*, uint64_t);

void ok_extract_kmers(const char* seq, uint64_t len, uint64_t k,
                      KmerCb cb, void* cookie) {
    tpoptoa::extract_kmers(seq, (std::size_t)len, (std::size_t)k,
                           [&](tpoptoa::KmerWord kw){ cb(cookie, kw); });
}

// Minimizer extraction
typedef void (*MiniCb)(void*, uint64_t, uint32_t);

void ok_extract_minimizers(const char* seq, uint64_t len,
                            uint64_t k, uint64_t w,
                            MiniCb cb, void* cookie) {
    tpoptoa::extract_minimizers(
        seq, (std::size_t)len, (std::size_t)k, (std::size_t)w,
        [&](tpoptoa::MinimizerSeed s){ cb(cookie, s.kmer, s.pos); });
}

// Myers alignment
int ok_myers_align(const char* query, int qlen,
                   const char* text, int tlen,
                   int max_ed, int* end_out) {
    auto r = tpoptoa::myers_align(query, qlen, text, tlen, max_ed);
    if (end_out) *end_out = r.query_end;
    return r.edit_distance;
}

}
