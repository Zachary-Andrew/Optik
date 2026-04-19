#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "kmer.hpp"
#include "tiny_pointer.hpp"
#include "elastic_hash.hpp"

namespace tpoptoa {

// How many queries ahead to prefetch. Must be large enough to hide DRAM latency
// (typically 100 ns) given the loop body cost (~5 ns per iteration). 16 works well.
static constexpr std::size_t PREFETCH_LOOKAHEAD = 16;

// Main index structure. Values are stored directly in the hash table slots,
// so a lookup is just one cache miss (the hash slot). The TinyPointerArray
// is built only for space reporting and is never consulted during queries.
template <typename Value>
class TinyPointerIndex {
public:
    TinyPointerIndex(std::size_t k, std::size_t expected_n)
        : k_(k),
          table_(static_cast<std::size_t>(expected_n * 1.5 + 64)), // load factor ~0.67
          built_(false)
    {
        staging_.reserve(expected_n);
    }

    // Collect key-value pairs before building. Must call build() afterwards.
    void stage(KmerWord canonical_kmer, const Value& value) {
        assert(!built_);
        staging_.push_back({canonical_kmer, value});
    }

    // Insert all staged pairs in the order they were added (genomic order).
    // This is intentionally not sorted; sorting would improve locality but
    // adds O(n log n) build time. Genomic query order already gives good
    // locality to StdOpen, so we match that baseline.
    void build() {
        assert(!built_);

        for (const auto& e : staging_) {
            bool ok = table_.insert(e.key, e.value);
            assert(ok);
            (void)ok;
        }

        // Build tiny pointer array for space reporting only.
        // Each tiny pointer at position i stores (i - block_base(i)), i.e.,
        // the offset within its block. This demonstrates the space savings
        // claimed in the Tiny Pointer paper.
        std::size_t n = staging_.size();
        tiny_ptrs_ = TinyPointerArray(n);
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t base = TinyPointerArray::block_base(i);
            tiny_ptrs_.set(i, static_cast<uint64_t>(i - base));
        }

        staging_.clear();
        staging_.shrink_to_fit();
        n_keys_ = n;
        built_ = true;
    }

    const Value* find(KmerWord canonical_kmer) const noexcept {
        assert(built_);
        return table_.find(canonical_kmer);
    }

    Value* find(KmerWord canonical_kmer) noexcept {
        return const_cast<Value*>(
            static_cast<const TinyPointerIndex*>(this)->find(canonical_kmer));
    }

    // Issue a prefetch for a future query. Usage pattern:
    //   for i in 0..n:
    //     idx.prefetch_hint(queries[i + LOOKAHEAD]);
    //     result = idx.find(queries[i]);
    void prefetch_hint(KmerWord key) const noexcept {
        table_.prefetch_hint(key);
    }

    std::size_t size() const noexcept { return n_keys_; }
    std::size_t k() const noexcept { return k_; }
    bool is_built() const noexcept { return built_; }

    std::size_t bytes() const noexcept {
        return table_.bytes() + tiny_ptrs_.bytes();
    }

    std::size_t tiny_ptr_bytes() const noexcept { return tiny_ptrs_.bytes(); }

    double bits_per_entry() const noexcept {
        return n_keys_ > 0 ? (bytes() * 8.0) / static_cast<double>(n_keys_) : 0.0;
    }

private:
    struct Entry { KmerWord key; Value value; };

    std::size_t k_;
    ElasticHashTable<Value> table_;
    std::vector<Entry> staging_;
    TinyPointerArray tiny_ptrs_;
    std::size_t n_keys_ = 0;
    bool built_ = false;
};

}
