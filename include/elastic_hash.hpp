#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace tpoptoa {

static constexpr uint64_t EMPTY_KEY = std::numeric_limits<uint64_t>::max();
static constexpr uint64_t TOMBSTONE = std::numeric_limits<uint64_t>::max() - 1;

/* Open addressing hash table with Robin Hood displacement.
 When inserting, if the existing key at a slot is closer to its home
than the key we're placing, we swap and continue. This ensures that
keys are stored with roughly equal probe distances, and the maximum
probe distance stays small. For lookups, we can stop early when we
see a key whose probe distance is less than our key's would be.*/
template <typename Value>
class ElasticHashTable {
public:
    struct Slot {
        uint64_t key = EMPTY_KEY;
        Value value = Value{};
    };

    ElasticHashTable() = default;

    explicit ElasticHashTable(std::size_t capacity)
        : slots_(next_pow2(std::max(capacity, std::size_t(1)))),
          mask_(slots_.size() - 1),
          size_(0),
          max_probe_dist_(0)
    {}

    std::size_t capacity() const noexcept { return slots_.size(); }
    std::size_t size() const noexcept { return size_; }
    std::size_t max_probe_dist() const noexcept { return max_probe_dist_; }

    bool insert(uint64_t key, const Value& value) {
        assert(key != EMPTY_KEY && key != TOMBSTONE);
        if (size_ >= slots_.size()) return false;

        std::size_t slot = primary_slot(key);
        std::size_t dist = 0;           // how far we've probed for our key
        uint64_t cur_k = key;
        Value cur_v = value;

        for (std::size_t i = 0; i < slots_.size(); ++i) {
            Slot& s = slots_[slot];

            if (s.key == EMPTY_KEY) {
                s.key = cur_k;
                s.value = cur_v;
                ++size_;
                if (dist > max_probe_dist_) max_probe_dist_ = dist;
                return true;
            }

            // How far has the resident key probed from its home?
            std::size_t rd = probe_distance(s.key, slot);

            // If our key has probed further, we deserve this slot.
            // Evict the resident and continue placing it.
            if (rd < dist) {
                if (dist > max_probe_dist_) max_probe_dist_ = dist;
                std::swap(cur_k, s.key);
                std::swap(cur_v, s.value);
                dist = rd;  // continue with evicted key's distance
            }

            slot = (slot + 1) & mask_;
            ++dist;
        }

        return false;
    }

    const Value* find(uint64_t key) const noexcept {
        assert(key != EMPTY_KEY && key != TOMBSTONE);
        std::size_t slot = primary_slot(key);

        // Only need to check up to max_probe_dist_ slots because Robin Hood
        // ensures no key is further from its home than that.
        for (std::size_t i = 0; i <= max_probe_dist_; ++i) {
            const Slot& s = slots_[slot];

            if (s.key == EMPTY_KEY) return nullptr;
            if (s.key == key) return &s.value;

            // If the resident here is closer to its home than our key would be,
            // our key cannot appear later (would violate Robin Hood ordering).
            if (probe_distance(s.key, slot) < probe_distance(key, slot))
                return nullptr;

            slot = (slot + 1) & mask_;
        }
        return nullptr;
    }

    Value* find(uint64_t key) noexcept {
        return const_cast<Value*>(
            static_cast<const ElasticHashTable*>(this)->find(key));
    }

    // Prefetch the cache line containing key's primary slot.
    // Call this several iterations before find() to hide DRAM latency.
    void prefetch_hint(uint64_t key) const noexcept {
        if (key == EMPTY_KEY || key == TOMBSTONE) return;
        __builtin_prefetch(&slot_ref(primary_slot(key)), 0, 1);
    }

    std::size_t bytes() const noexcept { return slots_.size() * sizeof(Slot); }

    const Slot& slot_ref(std::size_t i) const noexcept { return slots_[i & mask_]; }

    // Fibonacci hashing: multiply by golden ratio reciprocal, take top log2(cap) bits.
    // This preserves locality better than modulo for keys that are close in value.
    std::size_t primary_slot(uint64_t key) const noexcept {
        uint64_t mixed = key * UINT64_C(0x9E3779B97F4A7C15);
        return static_cast<std::size_t>(
            mixed >> (64u - static_cast<unsigned>(__builtin_ctzll(slots_.size()))));
    }

private:
    // Distance from home slot, measured in steps modulo capacity.
    std::size_t probe_distance(uint64_t key, std::size_t cur_slot) const noexcept {
        return (cur_slot + slots_.size() - primary_slot(key)) & mask_;
    }

    static std::size_t next_pow2(std::size_t n) noexcept {
        --n;
        n |= n >> 1; n |= n >> 2; n |= n >> 4;
        n |= n >> 8; n |= n >> 16; n |= n >> 32;
        return n + 1;
    }

    std::vector<Slot> slots_;
    std::size_t mask_ = 0;
    std::size_t size_ = 0;
    std::size_t max_probe_dist_ = 0;
};

}
