#pragma once

// Statistical primitives for the benchmark harness.
//
// All functions take a std::vector<double> of raw per-trial observations.
// We follow the same protocol as Wu (2016) for the bitpacking paper: median
// as the primary point estimate (robust to outliers), with mean ± 95% CI for
// significance tests. The CI uses the two-tailed Student t-distribution with
// Bessel-corrected sample variance.
//
// Pairwise significance is assessed by Welch's t-test (unequal variance).
// We also compute Cohen's d to report practical effect size alongside the
// p-value — a statistically significant difference that is negligible in
// effect size is not a meaningful performance win.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

namespace tpoptoa {
namespace stats {

inline double mean(const std::vector<double>& v) {
    assert(!v.empty());
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

// Bessel-corrected sample variance (divides by n-1).
inline double variance(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double m = mean(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return s / static_cast<double>(v.size() - 1);
}

inline double stddev(const std::vector<double>& v) { return std::sqrt(variance(v)); }

inline double median(std::vector<double> v) {
    assert(!v.empty());
    std::size_t n = v.size();
    std::sort(v.begin(), v.end());
    return (n % 2 == 0) ? 0.5 * (v[n/2 - 1] + v[n/2]) : v[n/2];
}

inline double minimum(const std::vector<double>& v) {
    return *std::min_element(v.begin(), v.end());
}
inline double maximum(const std::vector<double>& v) {
    return *std::max_element(v.begin(), v.end());
}

// Two-tailed t critical value for a 95% CI (α=0.05).
// Table entries indexed by df = n-1, valid for df 1..29.
// For df ≥ 30 we use the large-sample approximation 1.960.
inline double t_crit_95(std::size_t n) {
    if (n < 2) return 0.0;
    static const double table[] = {
        0.0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
        2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110,
        2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045
    };
    std::size_t df = n - 1;
    return (df < 30) ? table[df] : 1.960;
}

// 95% CI half-width: t * (s / sqrt(n)).
inline double ci95_half(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    return t_crit_95(v.size()) * stddev(v) / std::sqrt(static_cast<double>(v.size()));
}

// Cohen's d effect size: standardised mean difference, positive when a > b.
// Uses pooled standard deviation.
inline double cohens_d(const std::vector<double>& a, const std::vector<double>& b) {
    double na = static_cast<double>(a.size());
    double nb = static_cast<double>(b.size());
    if (na < 2 || nb < 2) return 0.0;
    double sp = std::sqrt(((na-1)*variance(a) + (nb-1)*variance(b)) / (na+nb-2));
    return (sp == 0.0) ? 0.0 : (mean(a) - mean(b)) / sp;
}

inline const char* cohens_label(double d) {
    double ad = std::fabs(d);
    if (ad < 0.2) return "negligible";
    if (ad < 0.5) return "small";
    if (ad < 0.8) return "medium";
    return "large";
}

// Regularised incomplete beta function I_x(a,b), implemented via Lentz's
// continued fraction. Used to compute the t-distribution CDF.
static double ibeta(double x, double a, double b) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;

    double lbeta = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);

    // Symmetry: use the complement when x is large to improve convergence.
    bool swapped = false;
    double xa = x, aa = a, ba = b;
    if (xa > (aa + 1.0) / (aa + ba + 2.0)) {
        xa = 1.0 - xa; std::swap(aa, ba); swapped = true;
    }

    double qab = aa + ba, qap = aa + 1.0, qam = aa - 1.0;
    double c = 1.0, d = 1.0 - qab * xa / qap;
    if (std::fabs(d) < 1e-30) d = 1e-30;
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m <= 200; ++m) {
        double dm = static_cast<double>(m);
        double num = dm * (ba - dm) * xa / ((qam + 2*dm) * (aa + 2*dm));
        d = 1.0 + num * d; c = 1.0 + num / c;
        if (std::fabs(d) < 1e-30) d = 1e-30;
        if (std::fabs(c) < 1e-30) c = 1e-30;
        d = 1.0 / d; h *= d * c;
        num = -(aa + dm) * (qab + dm) * xa / ((aa + 2*dm) * (qap + 2*dm));
        d = 1.0 + num * d; c = 1.0 + num / c;
        if (std::fabs(d) < 1e-30) d = 1e-30;
        if (std::fabs(c) < 1e-30) c = 1e-30;
        d = 1.0 / d;
        double delta = d * c;
        h *= delta;
        if (std::fabs(delta - 1.0) < 1e-10) break;
    }

    double result = std::exp(aa * std::log(xa) + ba * std::log(1.0 - xa) - lbeta) * h / aa;
    return swapped ? 1.0 - result : result;
}

// CDF of the t-distribution with nu degrees of freedom at point t.
static double t_cdf(double t, double nu) {
    double x = nu / (nu + t * t);
    double p = ibeta(x, nu / 2.0, 0.5) / 2.0;
    return (t >= 0.0) ? 1.0 - p : p;
}

struct WelchResult {
    double t_stat;
    double df;
    double p_value;
    double cohens_d_val;
};

// Welch's two-sample t-test (unequal variances). Two-tailed.
// Also computes Cohen's d for practical significance.
inline WelchResult welch_t_test(const std::vector<double>& a,
                                 const std::vector<double>& b) {
    double na = static_cast<double>(a.size());
    double nb = static_cast<double>(b.size());
    double va = variance(a), vb = variance(b);
    double sea2 = va / na, seb2 = vb / nb;
    double se = std::sqrt(sea2 + seb2);

    WelchResult r{};
    r.cohens_d_val = cohens_d(a, b);

    if (se < 1e-15) {
        r.t_stat = 0.0; r.df = na + nb - 2.0; r.p_value = 1.0;
        return r;
    }

    r.t_stat = (mean(a) - mean(b)) / se;

    // Welch–Satterthwaite degrees of freedom.
    double num = (sea2 + seb2) * (sea2 + seb2);
    double den = (sea2 * sea2) / (na - 1.0) + (seb2 * seb2) / (nb - 1.0);
    r.df = (den > 0.0) ? num / den : na + nb - 2.0;

    double p_upper = 1.0 - t_cdf(std::fabs(r.t_stat), r.df);
    r.p_value = 2.0 * p_upper;
    return r;
}

// Compact significance label.
inline const char* sig_stars(double p) {
    if (p < 0.001) return "***";
    if (p < 0.010) return "**";
    if (p < 0.050) return "*";
    return "ns";
}

} // namespace stats
} // namespace tpoptoa
