#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tpoptoa {

// 6 bits per pointer → block size ≤ 64. Enough for typical log n.
static constexpr unsigned TINY_BITS = 6;
static constexpr uint64_t TINY_MASK = (uint64_t(1) << TINY_BITS) - 1;

static constexpr std::size_t TINY_BLOCK = (std::size_t(1) << TINY_BITS);  // 64

// Stores an array of tiny pointers packed into 64‑bit words.
// Each pointer is an offset inside its block (0..TINY_BLOCK-1).
class TinyPointerArray {
public:
    TinyPointerArray() : n_(0) {}

    explicit TinyPointerArray(std::size_t n) : n_(n) {
        // ceil(n * TINY_BITS / 64)
        std::size_t words = (n * TINY_BITS + 63) / 64;
        data_.assign(words, 0);
    }

    std::size_t size() const noexcept { return n_; }

    void set(std::size_t i, uint64_t v) noexcept {
        assert(i < n_);
        assert(v < TINY_BLOCK);
        uint64_t bit_pos = static_cast<uint64_t>(i) * TINY_BITS;
        std::size_t word  = static_cast<std::size_t>(bit_pos >> 6);
        unsigned    shift = static_cast<unsigned>(bit_pos & 63);

        data_[word] &= ~(TINY_MASK << shift);
        data_[word] |=  (v          << shift);

        // If the field crosses a word boundary, write the overflow part.
        if (shift + TINY_BITS > 64) {
            unsigned overflow = (shift + TINY_BITS) - 64;
            data_[word + 1] &= ~(TINY_MASK >> (TINY_BITS - overflow));
            data_[word + 1] |=  (v >> (TINY_BITS - overflow));
        }
    }

    uint64_t get(std::size_t i) const noexcept {
        assert(i < n_);
        uint64_t bit_pos = static_cast<uint64_t>(i) * TINY_BITS;
        std::size_t word  = static_cast<std::size_t>(bit_pos >> 6);
        unsigned    shift = static_cast<unsigned>(bit_pos & 63);

        uint64_t val = (data_[word] >> shift) & TINY_MASK;

        // Read the second word if the field crosses boundary.
        if (shift + TINY_BITS > 64) {
            unsigned overflow = (shift + TINY_BITS) - 64;
            val |= (data_[word + 1] & ((uint64_t(1) << overflow) - 1))
                   << (TINY_BITS - overflow);
        }
        return val;
    }

    // Each block starts at a multiple of TINY_BLOCK.
    static std::size_t block_base(std::size_t i) noexcept {
        return (i / TINY_BLOCK) * TINY_BLOCK;
    }

    // Reconstruct full slot index from tiny pointer at position i.
    std::size_t full_index(std::size_t i) const noexcept {
        return block_base(i) + static_cast<std::size_t>(get(i));
    }

    std::size_t bytes() const noexcept {
        return data_.size() * sizeof(uint64_t);
    }

private:
    std::size_t          n_;
    std::vector<uint64_t> data_;
};

} // namespace tpoptoa
