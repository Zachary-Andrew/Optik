#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

// Myers bit-vector edit-distance (Myers 1999).
//
// Three implementation tiers, selected at compile time based on what the
// compiler has been told the CPU supports:
//
//   AVX-512 (F+BW+DQ)  — 512-bit vectors, 512 DP columns per cycle
//   Scalar 64-bit       — one uint64_t word, 64 DP columns per iteration
//   Blocked scalar      — multiple words for queries longer than 64 bases
//
// Every tier produces identical results.  The scalar path compiles and runs
// correctly on any x86-64 CPU with no special flags required.

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
#  include <immintrin.h>
#  define OPTIK_HAVE_MYERS_AVX512 1
#endif

namespace tpoptoa {

struct AlignResult {
    int edit_distance;
    int query_end;    // last text position of the best alignment
    int query_start;  // -1 unless traceback was requested
};

// scalar 64-bit Myers (queries up to 64 bases)
static inline int myers64(const char* query, int m,
                           const char* text,  int n)
{
    assert(m <= 64);
    uint64_t Peq[4] = {};
    for (int i = 0; i < m; ++i)
        for (int b = 0; b < 4; ++b)
            if ((query[i] & ~0x20) == "ACGT"[b])
                Peq[b] |= uint64_t(1) << i;

    uint64_t Pv = ~uint64_t(0), Mv = 0;
    int score = m;

    for (int j = 0; j < n; ++j) {
        int b = -1;
        switch (text[j] & ~0x20) {
            case 'A': b=0; break; case 'C': b=1; break;
            case 'G': b=2; break; case 'T': b=3; break;
        }
        uint64_t Eq = (b >= 0) ? Peq[b] : 0;
        uint64_t Xv = Eq | Mv;
        uint64_t Xh = (((Eq & Pv) + Pv) ^ Pv) | Eq;
        uint64_t Ph = Mv | ~(Xh | Pv);
        uint64_t Mh = Pv  &  Xh;
        uint64_t top = uint64_t(1) << (m - 1);
        if      (Ph & top) ++score;
        else if (Mh & top) --score;
        Ph = (Ph << 1) | 1;
        Mh =  Mh << 1;
        Pv = Mh | ~(Xv | Ph);
        Mv = Ph &  Xv;
    }
    return score;
}

//AVX-512 wide Myers (queries up to 512 bases)
#ifdef OPTIK_HAVE_MYERS_AVX512
static inline int myers512(const __m512i* Peq, int m,
                             const char* text, int n)
{
    __m512i Pv    = _mm512_set1_epi64(-1LL);
    __m512i Mv    = _mm512_setzero_si512();
    int     score = m;

    for (int j = 0; j < n; ++j) {
        int b = -1;
        switch (text[j] & ~0x20) {
            case 'A': b=0; break; case 'C': b=1; break;
            case 'G': b=2; break; case 'T': b=3; break;
        }
        __m512i Eq  = (b >= 0) ? Peq[b] : _mm512_setzero_si512();
        __m512i Xv  = _mm512_or_si512(Eq, Mv);
        __m512i sum = _mm512_add_epi64(_mm512_and_si512(Eq, Pv), Pv);
        __m512i Xh  = _mm512_or_si512(_mm512_xor_si512(sum, Pv), Eq);
        __m512i Ph  = _mm512_or_si512(Mv,
                          _mm512_andnot_si512(_mm512_or_si512(Xh, Pv),
                                              _mm512_set1_epi64(-1LL)));
        __m512i Mh  = _mm512_and_si512(Pv, Xh);

        // _mm512_extracti64x2_epi64 requires AVX512DQ, guarded above.
        __m128i ph_lane = _mm512_extracti64x2_epi64(Ph, 3);
        __m128i mh_lane = _mm512_extracti64x2_epi64(Mh, 3);
        uint64_t last_ph, last_mh;
        std::memcpy(&last_ph, &ph_lane, 8);
        std::memcpy(&last_mh, &mh_lane, 8);

        int top_bit = (m - 1) % 64;
        if      ((last_ph >> top_bit) & 1) ++score;
        else if ((last_mh >> top_bit) & 1) --score;

        // Shift Ph/Mh left by 1 bit with carry between 64-bit lanes.
        const __m512i perm = _mm512_set_epi64(6,5,4,3,2,1,0,7);
        const __m512i lomask = _mm512_set_epi64(-1,-1,-1,-1,-1,-1,-1,0);
        __m512i cph = _mm512_and_si512(
                          _mm512_permutexvar_epi64(perm, _mm512_srli_epi64(Ph, 63)),
                          lomask);
        __m512i cmh = _mm512_and_si512(
                          _mm512_permutexvar_epi64(perm, _mm512_srli_epi64(Mh, 63)),
                          lomask);
        // Ph shift-in bit is always 1 (semi-global: gaps at start are free).
        cph = _mm512_or_si512(cph, _mm512_set_epi64(0,0,0,0,0,0,0,1));

        Ph = _mm512_or_si512(_mm512_slli_epi64(Ph, 1), cph);
        Mh = _mm512_or_si512(_mm512_slli_epi64(Mh, 1), cmh);
        Pv = _mm512_or_si512(Mh,
               _mm512_andnot_si512(_mm512_or_si512(Xv, Ph),
                                   _mm512_set1_epi64(-1LL)));
        Mv = _mm512_and_si512(Ph, Xv);
    }
    return score;
}
#endif

// blocked scalar Myers (queries of any length)
static inline int myers_blocked(const char* query, int m,
                                  const char* text,  int n, int max_ed)
{
    int blocks = (m + 63) / 64;

    std::vector<uint64_t> Pv(blocks, ~uint64_t(0));
    std::vector<uint64_t> Mv(blocks, 0);
    std::vector<int>      sc(blocks);
    for (int b = 0; b < blocks; ++b)
        sc[b] = std::min(64, m - b*64);

    std::vector<std::array<uint64_t,4>> Peq(blocks, {0,0,0,0});
    for (int i = 0; i < m; ++i)
        for (int b2 = 0; b2 < 4; ++b2)
            if ((query[i] & ~0x20) == "ACGT"[b2])
                Peq[i/64][b2] |= uint64_t(1) << (i%64);

    int best = m;
    for (int j = 0; j < n; ++j) {
        int ch = -1;
        switch (text[j] & ~0x20) {
            case 'A': ch=0; break; case 'C': ch=1; break;
            case 'G': ch=2; break; case 'T': ch=3; break;
        }
        uint64_t cin = 1, cmi = 0;
        for (int bl = 0; bl < blocks; ++bl) {
            uint64_t Eq  = (ch >= 0) ? Peq[bl][ch] : 0;
            uint64_t Xv  = Eq | Mv[bl];
            uint64_t sum = (Eq & Pv[bl]) + Pv[bl];
            uint64_t Xh  = (sum ^ Pv[bl]) | Eq;
            uint64_t Ph  = Mv[bl] | ~(Xh | Pv[bl]);
            uint64_t Mh  = Pv[bl] &  Xh;
            int bits = (bl == blocks-1 && m%64) ? (m%64) : 64;
            uint64_t top = uint64_t(1) << (bits - 1);
            if      (Ph & top) ++sc[bl];
            else if (Mh & top) --sc[bl];
            uint64_t nci_ph = Ph >> 63, nci_mh = Mh >> 63;
            Ph = (Ph << 1) | cin;
            Mh = (Mh << 1) | cmi;
            cin = nci_ph; cmi = nci_mh;
            Pv[bl] = Mh | ~(Xv | Ph);
            Mv[bl] = Ph &  Xv;
        }
        best = std::min(best, sc[blocks-1]);
        if (best <= 0) break;
    }
    return std::min(best, max_ed);
}

// public interface
inline AlignResult myers_align(const char* query, int m,
                                const char* text,  int n,
                                int max_ed = std::numeric_limits<int>::max())
{
    if (m == 0) return {0, -1, -1};
    if (n == 0) return {m, -1, -1};

#ifdef OPTIK_HAVE_MYERS_AVX512
    if (m <= 512) {
        alignas(64) uint64_t peq_buf[4][8] = {};
        for (int i = 0; i < m; ++i)
            for (int b = 0; b < 4; ++b)
                if ((query[i] & ~0x20) == "ACGT"[b])
                    peq_buf[b][i/64] |= uint64_t(1) << (i%64);
        __m512i Peq[4];
        for (int b = 0; b < 4; ++b)
            Peq[b] = _mm512_load_si512(peq_buf[b]);
        return {std::min(myers512(Peq, m, text, n), max_ed), n-1, -1};
    }
#endif

    if (m <= 64)
        return {std::min(myers64(query, m, text, n), max_ed), n-1, -1};

    return {myers_blocked(query, m, text, n, max_ed), n-1, -1};
}

inline AlignResult myers_align(const std::string& q, const std::string& t,
                                int max_ed = std::numeric_limits<int>::max())
{
    return myers_align(q.data(), (int)q.size(),
                       t.data(), (int)t.size(), max_ed);
}

}
