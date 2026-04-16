#pragma once

// Locality-preserving insertion and prefetch layer, inspired by:
//   Pibiri, Shibuya & Limasset (2023) "Locality-Preserving Minimal Perfect
//   Hashing of k-mers", Bioinformatics 39:i534.
//
// The key insight is that if we insert keys into a Robin Hood hash table in
// the same order that the hash function would assign them to slots — i.e.
// sorted by primary_slot(key) — then keys with adjacent hash values cluster
// into tight runs of slots. Probe chains for those keys then share cache
// lines, which reduces LLC misses when queries arrive in a similar order.
//
// Build phase: accumulate all (key, value) pairs, sort by primary_slot(key),
// insert in that order. This is O(n log n) and why TP+OptOA builds slower
// than StdOpen, which inserts in arbitrary order.
//
// Query phase: find() optionally issues a __builtin_prefetch. For the
// prefetch to actually hide memory latency it must be issued well before
// the corresponding find() call — use prefetch_hint(queries[i + L]) with
// the main loop doing find(queries[i]). L=8 works well at ~100ns DRAM.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "kmer.hpp"
#include "elastic_hash.hpp"

namespace tpoptoa {

template <typename Value>
class PibiriCache {
public:
    PibiriCache(std::size_t k, std::size_t expected_n)
        : k_(k),
          table_(static_cast<std::size_t>(expected_n * 1.5 + 64)),
          built_(false)
    {
        staging_.reserve(expected_n);
    }

    void stage(KmerWord kmer, const Value& value) {
        assert(!built_);
        staging_.push_back({kmer, value});
    }

    void build() {
        assert(!built_);

        // Sort entries by the primary slot they'll occupy in the hash table.
        // After sorting, consecutive inserts go to consecutive slots, which
        // produces the dense spatial clusters that make probe chains cache-local.
        std::sort(staging_.begin(), staging_.end(),
                  [&](const Entry& a, const Entry& b) noexcept {
                      return table_.primary_slot(a.key) < table_.primary_slot(b.key);
                  });

        for (const auto& e : staging_) {
            bool ok = table_.insert(e.key, e.value);
            assert(ok && "table full — increase capacity multiplier");
            (void)ok;
        }

        std::vector<Entry>().swap(staging_);
        built_ = true;
    }

    // Standard lookup. For best performance use prefetch_hint() + find() in a
    // pipelined loop rather than calling find() alone in a tight loop.
    const Value* find(KmerWord kmer) const noexcept {
        assert(built_);
        return table_.find(kmer);
    }

    Value* find(KmerWord kmer) noexcept {
        return const_cast<Value*>(static_cast<const PibiriCache*>(this)->find(kmer));
    }

    // Issue a prefetch for kmer's primary slot. Call this L iterations before
    // the find(kmer) call so the prefetch has time to complete.
    void prefetch_hint(KmerWord kmer) const noexcept {
        table_.prefetch_hint(kmer);
    }

    bool        is_built()  const noexcept { return built_; }
    std::size_t size()      const noexcept { return table_.size(); }
    std::size_t capacity()  const noexcept { return table_.capacity(); }
    std::size_t bytes()     const noexcept {
        return table_.bytes() + staging_.capacity() * sizeof(Entry);
    }

    std::size_t primary_slot(KmerWord kmer) const noexcept {
        return table_.primary_slot(kmer);
    }

private:
    struct Entry { KmerWord key; Value value; };

    std::size_t             k_;
    ElasticHashTable<Value> table_;
    std::vector<Entry>      staging_;
    bool                    built_;
};

} // namespace tpoptoa
