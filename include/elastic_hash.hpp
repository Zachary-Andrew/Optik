#pragma once

// Elastic hashing with Robin Hood displacement.
//
// CHANGES FROM PREVIOUS VERSION
// ──────────────────────────────
// 1. PACKED 12-BYTE SLOTS
//    Original: {uint64_t key, uint32_t value} = 16 bytes due to padding.
//    Now:      PackedSlot with __attribute__((packed)) = 12 bytes exactly.
//    For 3.88M k-mers at 1.5× load → 8M slots × 4 bytes saved = 32 MB
//    smaller table. More of the table fits in LLC, reducing cold-miss
//    pressure on the first trials of each benchmark run.
//    Swap uses local temporaries (not std::swap) because packed fields
//    cannot have their address taken.
//
// 2. bits_per_entry() REPORTING
//    Exposes bytes()*8/size() so the benchmark can print per-method storage
//    density alongside ns/query. The value reflects total allocated slots
//    (power-of-2 rounded), not just occupied ones — i.e. it includes the
//    load-factor overhead, which is the honest number for memory budgeting.
//
// 3. REPLICA REMOVED
//    The dual-channel replica from the previous attempt was reverted.
//    It caused the build to touch ~192 MB of memory per trial, evicting
//    the primary table from LLC and making every query pass start cold.
//    The tailslayer hedging only helps when the table is genuinely
//    LLC-cold AND you control physical address placement (1 GB hugepage
//    with /proc/self/pagemap verification). In the benchmark's repeated-
//    trial loop neither condition holds, so the replica was a regression.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace tpoptoa {

static constexpr uint64_t EMPTY_KEY = std::numeric_limits<uint64_t>::max();
static constexpr uint64_t TOMBSTONE = std::numeric_limits<uint64_t>::max() - 1;

// ── Packed 12-byte slot ───────────────────────────────────────────────────────
// Avoids the 4-byte padding that a plain {uint64_t, uint32_t} struct has.
// Note: packed fields cannot be passed to std::swap (UB to take their address),
// so insert() uses explicit temporaries for the Robin Hood swap.
struct alignas(4) PackedSlot {
    uint64_t key   = EMPTY_KEY;
    uint32_t value = 0;
} __attribute__((packed));

static_assert(sizeof(PackedSlot) == 12, "PackedSlot must be 12 bytes");

template <typename Value>
class ElasticHashTable {
    using Slot = PackedSlot;
    static_assert(sizeof(Value) == 4, "ElasticHashTable optimised for 4-byte values");

public:
    ElasticHashTable() = default;

    explicit ElasticHashTable(std::size_t capacity)
        : slots_(next_pow2(std::max(capacity, std::size_t(1)))),
          mask_(slots_.size() - 1),
          size_(0),
          max_probe_dist_(0)
    {}

    std::size_t capacity()       const noexcept { return slots_.size(); }
    std::size_t size()           const noexcept { return size_; }
    std::size_t max_probe_dist() const noexcept { return max_probe_dist_; }

    bool insert(uint64_t key, const Value& value) {
        assert(key != EMPTY_KEY && key != TOMBSTONE);
        if (size_ >= slots_.size()) return false;

        std::size_t slot  = primary_slot(key);
        std::size_t dist  = 0;
        uint64_t    cur_k = key;
        uint32_t    cur_v = static_cast<uint32_t>(value);

        for (std::size_t i = 0; i < slots_.size(); ++i) {
            Slot& s = slots_[slot];

            if (s.key == EMPTY_KEY) {
                s.key   = cur_k;
                s.value = cur_v;
                ++size_;
                if (dist > max_probe_dist_) max_probe_dist_ = dist;
                return true;
            }

            std::size_t rd = probe_distance(s.key, slot);
            if (rd < dist) {
                if (dist > max_probe_dist_) max_probe_dist_ = dist;
                // Cannot use std::swap on packed fields.
                uint64_t tmp_k = s.key;   s.key   = cur_k; cur_k = tmp_k;
                uint32_t tmp_v = s.value; s.value = cur_v; cur_v = tmp_v;
                dist = rd;
            }

            slot = (slot + 1) & mask_;
            ++dist;
        }
        return false;
    }

    const Value* find(uint64_t key) const noexcept {
        assert(key != EMPTY_KEY && key != TOMBSTONE);
        std::size_t slot = primary_slot(key);

        for (std::size_t i = 0; i <= max_probe_dist_; ++i) {
            const Slot& s = slots_[slot];
            if (s.key == EMPTY_KEY) return nullptr;
            if (s.key == key)       return reinterpret_cast<const Value*>(&s.value);
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

    void prefetch_hint(uint64_t key) const noexcept {
        if (key == EMPTY_KEY || key == TOMBSTONE) return;
        __builtin_prefetch(&slots_[primary_slot(key) & mask_], 0, 1);
    }

    // Primary table bytes (no replica).
    std::size_t bytes() const noexcept { return slots_.size() * sizeof(Slot); }

    // Bits of storage per logical key (includes load-factor overhead).
    double bits_per_entry() const noexcept {
        if (size_ == 0) return 0.0;
        return static_cast<double>(bytes() * 8) / static_cast<double>(size_);
    }

    std::size_t primary_slot(uint64_t key) const noexcept {
        uint64_t mixed = key * UINT64_C(0x9E3779B97F4A7C15);
        return static_cast<std::size_t>(
            mixed >> (64u - static_cast<unsigned>(__builtin_ctzll(slots_.size()))));
    }

    const Slot& slot_ref(std::size_t i) const noexcept { return slots_[i & mask_]; }

private:
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
    std::size_t mask_           = 0;
    std::size_t size_           = 0;
    std::size_t max_probe_dist_ = 0;
};

} // namespace tpoptoa
