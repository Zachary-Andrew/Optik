#pragma once

// kmer.hpp — encoding, decoding, and canonical-form helpers for k-mers.
//
// We pack each base into 2 bits (A=0, C=1, G=2, T=3) so a 31-mer fits in a
// single 64-bit word — convenient, fast, and the standard representation used
// by tools like SSHash (Pibiri 2022) and minimap2 (Li 2021).

#include <cstdint>
#include <cstddef>
#include <string>
#include <stdexcept>
#include <climits>

namespace tpoptoa {

// The maximum k we support with a 64-bit word.
static constexpr std::size_t MAX_K = 32;

// Encode a single DNA character to its 2-bit value.
// Returns 0xFF on anything that isn't A/C/G/T (including N).
inline uint8_t base_encode(char c) noexcept {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:             return 0xFF;
    }
}

// Decode a 2-bit value back to an ASCII base character.
inline char base_decode(uint8_t b) noexcept {
    static constexpr char table[4] = {'A', 'C', 'G', 'T'};
    return table[b & 3];
}

// Complement of a 2-bit base value: XOR with 3 (A<->T, C<->G).
inline uint8_t base_complement(uint8_t b) noexcept {
    return b ^ 3;
}

// A 64-bit packed k-mer.  The most-significant 2 bits hold base 0 (leftmost);
// the least-significant 2 bits hold base k-1 (rightmost).
using KmerWord = uint64_t;

// Build a k-mer word from a string slice of exactly k characters.
// Returns false (and leaves out unchanged) if any ambiguous base is found.
inline bool encode_kmer(const char* seq, std::size_t k, KmerWord& out) noexcept {
    out = 0;
    for (std::size_t i = 0; i < k; ++i) {
        uint8_t b = base_encode(seq[i]);
        if (b == 0xFF) return false;
        out = (out << 2) | b;
    }
    return true;
}

// Reverse-complement a k-mer packed word, returning the RC word.
// We XOR every 2-bit lane with 3 (complement) then reverse the order.
inline KmerWord reverse_complement(KmerWord kmer, std::size_t k) noexcept {
    KmerWord rc = 0;
    for (std::size_t i = 0; i < k; ++i) {
        uint8_t b = kmer & 3;
        rc = (rc << 2) | base_complement(b);
        kmer >>= 2;
    }
    return rc;
}

// Return the canonical (lexicographically smaller of forward / RC) form.
// This is the standard trick for strand-agnostic indexing.
inline KmerWord canonical(KmerWord kmer, std::size_t k) noexcept {
    KmerWord rc = reverse_complement(kmer, k);
    return (kmer < rc) ? kmer : rc;
}

// Decode a k-mer word back to a string.
inline std::string decode_kmer(KmerWord kmer, std::size_t k) {
    std::string s(k, 'N');
    for (std::size_t i = k; i-- > 0;) {
        s[i] = base_decode(static_cast<uint8_t>(kmer & 3));
        kmer >>= 2;
    }
    return s;
}

// Rolling k-mer update: slide one base to the right.
// Given the current packed k-mer, the incoming new base, and a mask that keeps
// only the lowest 2k bits, returns the next k-mer word.
inline KmerWord roll_kmer(KmerWord prev, uint8_t new_base, KmerWord mask) noexcept {
    return ((prev << 2) | new_base) & mask;
}

// The mask that zeroes bits above position 2k-1, used by roll_kmer.
inline KmerWord kmer_mask(std::size_t k) noexcept {
    // Avoid UB for k==32 via conditional.
    return (k < 32) ? ((KmerWord(1) << (2 * k)) - 1)
                    :  ~KmerWord(0);
}

} // namespace tpoptoa
