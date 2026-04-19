#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

namespace tpoptoa {

static constexpr std::size_t MAX_K = 32;

using KmerWord = uint64_t;

// Map A/C/G/T to 2-bit values: A=00, C=01, G=10, T=11
inline uint8_t base_encode(char c) noexcept {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return 0xFF;  // ambiguous
    }
}

inline char base_decode(uint8_t b) noexcept {
    static constexpr char table[4] = {'A', 'C', 'G', 'T'};
    return table[b & 3];
}

// Complement: A<->T (0<->3) and C<->G (1<->2), which is just XOR with 3.
inline uint8_t base_complement(uint8_t b) noexcept {
    return b ^ 3;
}

// Build a packed k-mer from a string of exactly k characters.
// Returns false if any character is not A/C/G/T.
inline bool encode_kmer(const char* seq, std::size_t k, KmerWord& out) noexcept {
    out = 0;
    for (std::size_t i = 0; i < k; ++i) {
        uint8_t b = base_encode(seq[i]);
        if (b == 0xFF) return false;
        out = (out << 2) | b;
    }
    return true;
}

// Reverse complement: complement each base and reverse the order.
inline KmerWord reverse_complement(KmerWord kmer, std::size_t k) noexcept {
    KmerWord rc = 0;
    for (std::size_t i = 0; i < k; ++i) {
        uint8_t b = kmer & 3;                 // extract least significant base
        rc = (rc << 2) | base_complement(b);  // push to rc's right side
        kmer >>= 2;                           // shift original to next base
    }
    return rc;
}

// Canonical form: the smaller of forward and reverse complement.
// This makes indexing strand-agnostic.
inline KmerWord canonical(KmerWord kmer, std::size_t k) noexcept {
    KmerWord rc = reverse_complement(kmer, k);
    return (kmer < rc) ? kmer : rc;
}

// Convert packed k-mer back to a string for human-readable output.
inline std::string decode_kmer(KmerWord kmer, std::size_t k) {
    std::string s(k, 'N');
    for (std::size_t i = k; i-- > 0;) {
        s[i] = base_decode(static_cast<uint8_t>(kmer & 3));
        kmer >>= 2;
    }
    return s;
}

// Slide window: drop oldest base (implicitly by masking), shift left, add new.
inline KmerWord roll_kmer(KmerWord prev, uint8_t new_base, KmerWord mask) noexcept {
    return ((prev << 2) | new_base) & mask;
}

// Mask that keeps only the lowest 2k bits. For k=32, all 64 bits are used.
inline KmerWord kmer_mask(std::size_t k) noexcept {
    return (k < 32) ? ((KmerWord(1) << (2 * k)) - 1) : ~KmerWord(0);
}

}
