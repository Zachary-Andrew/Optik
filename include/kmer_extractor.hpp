#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include "kmer.hpp"

namespace tpoptoa {

// Calls cb(kmer) for every canonical k-mer in the sequence.
// Uses a rolling hash to update in O(1) per base instead of O(k).
template <typename Callback>
void extract_kmers(const char* seq, std::size_t len,
                   std::size_t k, Callback&& cb)
{
    if (len < k) return;

    KmerWord mask = kmer_mask(k);
    KmerWord fwd = 0;      // forward k-mer being built
    KmerWord rev = 0;      // reverse complement built in parallel
    std::size_t valid = 0; // count of consecutive non-N bases

    // When adding a new base to the reverse complement, it goes at the
    // most significant end (left side), so we shift by 2*(k-1) bits.
    unsigned rc_shift = static_cast<unsigned>(2 * (k - 1));

    for (std::size_t i = 0; i < len; ++i) {
        uint8_t fb = base_encode(seq[i]);

        if (fb == 0xFF) {
            // N or other ambiguous base breaks the window
            fwd = 0;
            rev = 0;
            valid = 0;
            continue;
        }

        // Roll forward: shift left, add new base, mask to k bits
        fwd = ((fwd << 2) | fb) & mask;

        // Roll reverse complement: shift right (oldest base falls off),
        // put complemented new base at the top, then mask
        uint8_t rb = base_complement(fb);
        rev = (rev >> 2) | (static_cast<KmerWord>(rb) << rc_shift);
        rev &= mask;

        ++valid;
        if (valid >= k) {
            // Canonical = lexicographically smaller of forward and reverse
            KmerWord can = (fwd < rev) ? fwd : rev;
            cb(can);
        }
    }
}

template <typename Callback>
void extract_kmers(const std::string& seq, std::size_t k, Callback&& cb) {
    extract_kmers(seq.data(), seq.size(), k, std::forward<Callback>(cb));
}

}
