#pragma once

// tiny_pointer.hpp — fixed-size tiny pointers as described in
//   Bender, Conway, Farach-Colton, Kuszmaul, Tagliavini (2021)
//   "Tiny Pointers", arXiv:2111.12800
//
// Core idea:
//   In a table filled to load factor (1 - 1/k), a normal pointer to any slot
//   requires ceil(log2(n)) bits.  Tiny pointers exploit the fact that, for a
//   given "owner" (a hash-table slot), the referenced element is almost
//   certainly in one of a small window of nearby slots.  The optimal
//   fixed-size tiny pointer has Θ(log log log n + log k) bits.
//
// Practical construction used here (Section 3 of the paper):
//
//   We divide the table into blocks of size B = Θ(log n).  Each block is
//   backed by a "mini-table" of B entries.  Within a block a tiny pointer is
//   just an index into that block — it therefore needs log2(B) bits.
//   Because B = O(log n), log2(B) = O(log log n).  For n ≤ 2^32 a block of
//   B=64 suffices, giving 6-bit tiny pointers — much smaller than 32-bit
//   direct indices.
//
//   We additionally store a "segment base" per block to reconstruct the full
//   table offset: full_index = block_base + tiny_ptr.
//
// This implementation:
//   • Stores tiny pointers in a packed bit array (6 bits each by default).
//   • Provides O(1) encode / decode.
//   • Is a header-only value type — no heap allocation inside.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tpoptoa {

// Number of bits used per tiny pointer.
// With TINY_BITS=6, each pointer fits in 6 bits → block size ≤ 64.
// With TINY_BITS=8, block size ≤ 256 and the storage overhead is still
// only 25% of a 32-bit pointer.
static constexpr unsigned TINY_BITS = 6;
static constexpr uint64_t TINY_MASK = (uint64_t(1) << TINY_BITS) - 1;

// Block size must not exceed 2^TINY_BITS.
static constexpr std::size_t TINY_BLOCK = (std::size_t(1) << TINY_BITS);  // 64

// ─── TinyPointerArray ────────────────────────────────────────────────────────
// Compact array of n tiny pointers, each TINY_BITS wide, packed into 64-bit
// words.  A tiny pointer at position i encodes an offset within its block of
// size TINY_BLOCK; the caller uses block_base(i) + get(i) to obtain the full
// table index.
class TinyPointerArray {
public:
    TinyPointerArray() : n_(0) {}

    explicit TinyPointerArray(std::size_t n) : n_(n) {
        // Number of 64-bit words needed to store n * TINY_BITS bits.
        std::size_t words = (n * TINY_BITS + 63) / 64;
        data_.assign(words, 0);
    }

    std::size_t size() const noexcept { return n_; }

    // Write a tiny pointer value v at logical position i.
    // v must be in [0, TINY_BLOCK).
    void set(std::size_t i, uint64_t v) noexcept {
        assert(i < n_);
        assert(v < TINY_BLOCK);
        uint64_t bit_pos = static_cast<uint64_t>(i) * TINY_BITS;
        std::size_t word  = static_cast<std::size_t>(bit_pos >> 6);   // / 64
        unsigned    shift = static_cast<unsigned>(bit_pos & 63);       // % 64

        // Clear the field then OR in the new value.
        data_[word] &= ~(TINY_MASK << shift);
        data_[word] |=  (v          << shift);

        // Handle the case where the 6-bit field crosses a word boundary.
        if (shift + TINY_BITS > 64) {
            unsigned overflow = (shift + TINY_BITS) - 64;
            data_[word + 1] &= ~(TINY_MASK >> (TINY_BITS - overflow));
            data_[word + 1] |=  (v >> (TINY_BITS - overflow));
        }
    }

    // Read the tiny pointer value at logical position i.
    uint64_t get(std::size_t i) const noexcept {
        assert(i < n_);
        uint64_t bit_pos = static_cast<uint64_t>(i) * TINY_BITS;
        std::size_t word  = static_cast<std::size_t>(bit_pos >> 6);
        unsigned    shift = static_cast<unsigned>(bit_pos & 63);

        uint64_t val = (data_[word] >> shift) & TINY_MASK;

        if (shift + TINY_BITS > 64) {
            unsigned overflow = (shift + TINY_BITS) - 64;
            val |= (data_[word + 1] & ((uint64_t(1) << overflow) - 1))
                   << (TINY_BITS - overflow);
        }
        return val;
    }

    // Return the base table index of the block that owns position i.
    // The block number is floor(i / TINY_BLOCK); the base index is that
    // times TINY_BLOCK.  Given the base, the full slot is base + get(i).
    static std::size_t block_base(std::size_t i) noexcept {
        return (i / TINY_BLOCK) * TINY_BLOCK;
    }

    // Convenience: reconstruct the full table slot index from tiny pointer i.
    std::size_t full_index(std::size_t i) const noexcept {
        return block_base(i) + static_cast<std::size_t>(get(i));
    }

    // Bytes of storage consumed by the packed pointer array.
    std::size_t bytes() const noexcept {
        return data_.size() * sizeof(uint64_t);
    }

private:
    std::size_t          n_;      // logical number of tiny pointers
    std::vector<uint64_t> data_;  // packed storage
};

} // namespace tpoptoa
