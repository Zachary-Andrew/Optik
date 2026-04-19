#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>

#include "kmer.hpp"
#include "kmer_extractor.hpp"

namespace tpoptoa {

// A minimizer is the lexicographically smallest canonical k-mer in a sliding
// window of w consecutive k-mers.  Selecting ~1/w of all k-mers as seeds for
// the mapper reduces the number of index lookups by w while keeping every
// alignment seeded — any two sequences sharing ≥w consecutive k-mers must
// share at least one minimizer.
//
// We use a monotone deque (sliding-window minimum in O(1) amortised) rather
// than a naive O(w) rescan, so the total cost stays O(L) for a sequence of
// length L.

struct MinimizerSeed {
    KmerWord kmer;
    uint32_t pos;   // 0-based position of the first base of this k-mer
};

template <typename Callback>
void extract_minimizers(const char* seq, std::size_t len,
                        std::size_t k, std::size_t w,
                        Callback&& cb)
{
    if (len < k + w - 1) return;

    // Collect all canonical k-mers with their positions first, then slide
    // the window.  We reuse a small ring rather than a full vector when w is
    // small, but the deque approach is simpler and correct for all w.
    struct Pos { KmerWord kmer; uint32_t pos; };
    std::deque<Pos> dq;   // front = current minimum candidate

    uint32_t idx = 0;
    KmerWord prev_min = ~KmerWord(0);

    extract_kmers(seq, len, k, [&](KmerWord kmer) {
        uint32_t cur_pos = idx++;

        // Evict entries outside the current window from the back of the deque
        // when they're larger than the new entry — they can never be minimum.
        while (!dq.empty() && dq.back().kmer >= kmer)
            dq.pop_back();
        dq.push_back({kmer, cur_pos});

        // Evict expired entries from the front.
        while (dq.front().pos + w <= cur_pos)
            dq.pop_front();

        // Only emit when we have a full window.
        if (cur_pos + 1 >= w) {
            KmerWord min_kmer = dq.front().kmer;
            // Deduplicate: don't emit the same minimizer twice in a row.
            if (min_kmer != prev_min) {
                cb(MinimizerSeed{min_kmer,
                                 static_cast<uint32_t>(dq.front().pos)});
                prev_min = min_kmer;
            }
        }
    });
}

template <typename Callback>
void extract_minimizers(const std::string& seq,
                        std::size_t k, std::size_t w,
                        Callback&& cb)
{
    extract_minimizers(seq.data(), seq.size(), k, w,
                       std::forward<Callback>(cb));
}

}
