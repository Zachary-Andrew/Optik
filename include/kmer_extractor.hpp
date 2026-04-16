#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "kmer.hpp"

namespace tpoptoa {

// Extract all canonical k‑mers from a DNA sequence in O(1) per base.
// For each window, calls cb(canonical_kmer). Skips windows containing N.
template <typename Callback>
void extract_kmers(const char* seq, std::size_t len,
                   std::size_t k, Callback&& cb)
{
    if (len < k) return;

    KmerWord mask    = kmer_mask(k);
    KmerWord fwd     = 0;  // forward k‑mer
    KmerWord rev     = 0;  // reverse complement (maintained in parallel)
    std::size_t valid = 0; // number of non‑N bases in current window

    unsigned rc_shift = static_cast<unsigned>(2 * (k - 1));

    for (std::size_t i = 0; i < len; ++i) {
        uint8_t fb = base_encode(seq[i]);

        if (fb == 0xFF) {
            // Ambiguous base: reset the window.
            fwd   = 0;
            rev   = 0;
            valid = 0;
            continue;
        }

        fwd = ((fwd << 2) | fb) & mask;

        uint8_t rb = base_complement(fb);
        rev = (rev >> 2) | (static_cast<KmerWord>(rb) << rc_shift);
        rev &= mask;

        ++valid;
        if (valid >= k) {
            KmerWord can = (fwd < rev) ? fwd : rev;
            cb(can);
        }
    }
}

template <typename Callback>
void extract_kmers(const std::string& seq, std::size_t k, Callback&& cb) {
    extract_kmers(seq.data(), seq.size(), k, std::forward<Callback>(cb));
}

} // namespace tpoptoa
