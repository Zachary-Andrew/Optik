#pragma once

// Top-level TP+OptOA index.
//
// CHANGES FROM PREVIOUS VERSION
// ──────────────────────────────
// 1. PREFETCH_LOOKAHEAD reduced from 16 → 8.
//    At ~99ns DRAM and ~12ns/query loop body, L=8 gives 96ns of prefetch
//    lead — just enough to cover one DRAM miss. L=16 was issuing prefetches
//    192ns ahead, wasting out-of-order execution window slots on 16
//    outstanding memory requests and evicting useful data from L1/L2
//    before it was needed. Optimal L = ceil(DRAM_ns / loop_ns) = 9 ≈ 8.
//
// 2. Dual-channel replica removed (see elastic_hash.hpp for rationale).
//
// 3. bits_per_entry() forwarded from ElasticHashTable for benchmark reporting.
//
// 4. TinyPointerArray still built for space reporting, not on query hot path.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kmer.hpp"
#include "tiny_pointer.hpp"
#include "elastic_hash.hpp"

namespace tpoptoa {

// Tuned to ceil(DRAM_ns / loop_body_ns) ≈ ceil(99/12) = 9, rounded to 8.
// Change this if your DRAM latency or loop body time differ significantly.
static constexpr std::size_t PREFETCH_LOOKAHEAD = 8;

template <typename Value>
class TinyPointerIndex {
public:
    TinyPointerIndex(std::size_t k, std::size_t expected_n)
        : k_(k),
          table_(static_cast<std::size_t>(expected_n * 1.5 + 64)),
          built_(false)
    {
        staging_.reserve(expected_n);
    }

    void stage(KmerWord canonical_kmer, const Value& value) {
        assert(!built_);
        staging_.push_back({canonical_kmer, value});
    }

    void build() {
        assert(!built_);

        for (const auto& e : staging_) {
            bool ok = table_.insert(e.key, e.value);
            assert(ok && "table full — increase capacity multiplier");
            (void)ok;
        }

        // TinyPointerArray: built for space reporting, not on query hot path.
        std::size_t n = staging_.size();
        tiny_ptrs_ = TinyPointerArray(n);
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t base = TinyPointerArray::block_base(i);
            tiny_ptrs_.set(i, static_cast<uint64_t>(i - base));
        }

        { decltype(staging_) tmp; tmp.swap(staging_); }
        n_keys_ = n;
        built_  = true;
    }

    const Value* find(KmerWord canonical_kmer) const noexcept {
        assert(built_);
        return table_.find(canonical_kmer);
    }

    Value* find(KmerWord canonical_kmer) noexcept {
        return const_cast<Value*>(
            static_cast<const TinyPointerIndex*>(this)->find(canonical_kmer));
    }

    void prefetch_hint(KmerWord key) const noexcept {
        table_.prefetch_hint(key);
    }

    std::size_t size()     const noexcept { return n_keys_; }
    std::size_t k()        const noexcept { return k_; }
    bool        is_built() const noexcept { return built_; }

    std::size_t bytes()          const noexcept { return table_.bytes() + tiny_ptrs_.bytes(); }
    double      bits_per_entry() const noexcept { return table_.bits_per_entry(); }
    std::size_t tiny_ptr_bytes() const noexcept { return tiny_ptrs_.bytes(); }

private:
    struct Entry { KmerWord key; Value value; };

    std::size_t              k_;
    ElasticHashTable<Value>  table_;
    std::vector<Entry>       staging_;
    TinyPointerArray         tiny_ptrs_;
    std::size_t              n_keys_ = 0;
    bool                     built_  = false;
};

} // namespace tpoptoa
