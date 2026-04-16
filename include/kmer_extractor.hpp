#pragma once

// kmer_extractor.hpp — extract canonical k-mers from DNA sequences,
// driving the TP+OptOA index build.
//
// Uses a rolling hash to advance the k-mer window in O(1) per base rather
// than re-encoding from scratch.  Skips windows that contain any ambiguous
// base (N, etc.) by resetting the valid-bases counter.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "kmer.hpp"

namespace tpoptoa {

// Visit every valid canonical k-mer in sequence `seq` (of length `len`),
// calling callback(kmer_word) for each one in order of first occurrence.
//
// Duplicate canonical k-mers are NOT deduplicated here — that is the
// caller's responsibility (e.g. using the index's stage() + build()).
template <typename Callback>
void extract_kmers(const char* seq, std::size_t len,
                   std::size_t k, Callback&& cb)
{
    if (len < k) return;

    KmerWord mask    = kmer_mask(k);
    KmerWord fwd     = 0;  // forward k-mer
    KmerWord rev     = 0;  // reverse-complement k-mer (maintained in parallel)
    std::size_t valid = 0; // number of valid (non-N) bases in the current window

    // Precompute the shift needed to place a new RC base at the leftmost
    // position of the k-mer word: shift by 2*(k-1) bits.
    unsigned rc_shift = static_cast<unsigned>(2 * (k - 1));

    for (std::size_t i = 0; i < len; ++i) {
        uint8_t fb = base_encode(seq[i]);

        if (fb == 0xFF) {
            // Ambiguous base — reset the window.
            fwd   = 0;
            rev   = 0;
            valid = 0;
            continue;
        }

        // Roll the forward k-mer: shift left, add new base, mask.
        fwd = ((fwd << 2) | fb) & mask;

        // Roll the reverse-complement k-mer: the RC of the new base goes at
        // the most-significant position.
        uint8_t rb = base_complement(fb);  // complement
        rev = (rev >> 2) | (static_cast<KmerWord>(rb) << rc_shift);
        // Mask to k bits (only needed for k < 32; harmless otherwise).
        rev &= mask;

        ++valid;
        if (valid >= k) {
            // We have a complete k-mer window.
            KmerWord can = (fwd < rev) ? fwd : rev;
            cb(can);
        }
    }
}

// Convenience overload for std::string.
template <typename Callback>
void extract_kmers(const std::string& seq, std::size_t k, Callback&& cb) {
    extract_kmers(seq.data(), seq.size(), k, std::forward<Callback>(cb));
}

} // namespace tpoptoa
