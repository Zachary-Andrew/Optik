#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "kmer.hpp"
#include "elastic_hash.hpp"

namespace tpoptoa {

// Insertion order = sorted by primary_slot(key). This creates spatial locality
// in the hash table: consecutive inserts go to close slots, so probe chains
// share cache lines.
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

        // Sort by the slot where each key will land.
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

    const Value* find(KmerWord kmer) const noexcept {
        assert(built_);
        return table_.find(kmer);
    }

    Value* find(KmerWord kmer) noexcept {
        return const_cast<Value*>(static_cast<const PibiriCache*>(this)->find(kmer));
    }

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
