#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tpoptoa {

// Each tiny pointer uses only 6 bits, so it can index within a block of 64 slots.
// Given a table with n slots divided into blocks of size 64, a tiny pointer
// stores just the offset within its block. The full index is block_base + offset.
// This works because any slot a pointer might reference is almost always within
// the same block as the pointer itself (locality property).
static constexpr unsigned TINY_BITS = 6;
static constexpr uint64_t TINY_MASK = (uint64_t(1) << TINY_BITS) - 1;
static constexpr std::size_t TINY_BLOCK = (std::size_t(1) << TINY_BITS); // 64

// Packed array of n tiny pointers, each TINY_BITS wide, stored in 64-bit words.
// This is space-efficient: n=1M uses only 0.75 MB instead of 4 MB for 32-bit pointers.
class TinyPointerArray {
public:
    TinyPointerArray() : n_(0) {}

    explicit TinyPointerArray(std::size_t n) : n_(n) {
        std::size_t words = (n * TINY_BITS + 63) / 64;
        data_.assign(words, 0);
    }

    std::size_t size() const noexcept { return n_; }

    void set(std::size_t i, uint64_t v) noexcept {
        assert(i < n_);
        assert(v < TINY_BLOCK);
        uint64_t bit_pos = static_cast<uint64_t>(i) * TINY_BITS;
        std::size_t word = static_cast<std::size_t>(bit_pos >> 6);
        unsigned shift = static_cast<unsigned>(bit_pos & 63);

        // Clear the field (TINY_BITS wide) then OR in the new value
        data_[word] &= ~(TINY_MASK << shift);
        data_[word] |= (v << shift);

        // Handle cross-word boundary if the field doesn't fit in one word
        if (shift + TINY_BITS > 64) {
            unsigned overflow = (shift + TINY_BITS) - 64;
            data_[word + 1] &= ~(TINY_MASK >> (TINY_BITS - overflow));
            data_[word + 1] |= (v >> (TINY_BITS - overflow));
        }
    }

    uint64_t get(std::size_t i) const noexcept {
        assert(i < n_);
        uint64_t bit_pos = static_cast<uint64_t>(i) * TINY_BITS;
        std::size_t word = static_cast<std::size_t>(bit_pos >> 6);
        unsigned shift = static_cast<unsigned>(bit_pos & 63);

        uint64_t val = (data_[word] >> shift) & TINY_MASK;

        if (shift + TINY_BITS > 64) {
            unsigned overflow = (shift + TINY_BITS) - 64;
            val |= (data_[word + 1] & ((uint64_t(1) << overflow) - 1))
                   << (TINY_BITS - overflow);
        }
        return val;
    }

    static std::size_t block_base(std::size_t i) noexcept {
        return (i / TINY_BLOCK) * TINY_BLOCK;
    }

    std::size_t full_index(std::size_t i) const noexcept {
        return block_base(i) + static_cast<std::size_t>(get(i));
    }

    std::size_t bytes() const noexcept {
        return data_.size() * sizeof(uint64_t);
    }

private:
    std::size_t n_;
    std::vector<uint64_t> data_;
};

}
