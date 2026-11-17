/*
 * @fileoverview Copyright (c) 2021, Stefano Gualandi,
 *               via Ferrata, 5, I-27100, Pavia, Italy
 *
 * @author stefano.gualandi@gmail.com (Stefano Gualandi)
 *
 *
 * External libraries:
 * - Super fast sorting algorithm (C++11)
 *    https://github.com/orlp/pdqsort
 *    (Zlib license)
 *
 * - Efficient parallel sorting algorithm (C++11)
 *    https://github.com/baserinia/parallel-sort
 *    (GPL-3.0 License)
 *
 * - AVX-512 and AVX2 sorting (optional, C++17), see ot1d_simdsort.inc
 *    https://github.com/intel/x86-simd-sort
 *    (BSD 3-Clause License)
 *    Enabled by compiling with -DOT1D_USE_AVX512 and/or -DOT1D_USE_AVX2, and
 *    linking ot1d_avx2.cpp and/or ot1d_avx512.cpp; the best ISA supported by
 *    the CPU is selected at runtime.
 *
 *
 * NOTE:
 *   - https://encyclopediaofmath.org/index.php?title=Wasserstein_metric
 *   - https://www.euroscipy.org/2018/descriptions/Detecting%20anomalies%20using%20statistical%20distances.html
 *   - https://www.kaggle.com/nhan1212/some-statistical-distances/comments
 *   - https://www.datadoghq.com/blog/engineering/robust-statistical-distances-for-machine-learning/
 *
 * Conventions for all the functions below:
 *   - x (size m) and y (size n) are the supports of the two measures;
 *     a and b (when given) are their (nonnegative) weights, which should have
 *     the same total mass. If they do not, only the smaller total mass is
 *     transported (partial plan and value; the Python wrapper warns about it).
 *     Without weights, uniform weights 1/m and 1/n are used.
 *   - Inputs are never modified (sorting is done on internal copies).
 *   - If sorting == false, the inputs are assumed to be already sorted.
 *   - Functions "_withplan" write an optimal plan with at most m+n-1 entries:
 *       map[2k] = index in x, map[2k+1] = index in y, mass[k] = mass moved,
 *     and *map_size = number of entries. The caller must allocate
 *     map with 2*(m+n-1) ints and mass with (m+n-1) doubles.
 *   - p=2 functions return W_2 = sqrt(sum mass * |x-y|^2).
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <utility>
#include <vector>

// Super fast sorting algorithm (C++11)
// https://github.com/orlp/pdqsort
#include "pdqsort.h"

// Efficient parallel sorting algorithm (C++11)
// https://github.com/baserinia/parallel-sort
#include "parasort.h"

#if defined(OT1D_USE_AVX512) || defined(OT1D_USE_AVX2)
#define OT1D_USE_SIMD
#endif

namespace ot1d_simd {

enum Level { NONE = 0, AVX2 = 1, AVX512 = 2 };

// Defined in ot1d_avx512.cpp and ot1d_avx2.cpp (the only files compiled with SIMD flags)
#ifdef OT1D_USE_AVX512
namespace avx512 {
void sort(double* x, size_t n);
void sort_kv(double* key, double* val, size_t n);
void sort_kv(double* key, uint64_t* val, size_t n);
}
#endif
#ifdef OT1D_USE_AVX2
namespace avx2 {
void sort(double* x, size_t n);
void sort_kv(double* key, double* val, size_t n);
void sort_kv(double* key, uint64_t* val, size_t n);
}
#endif

// Best level compiled in and supported by the CPU and the OS. For AVX-512,
// x86-simd-sort needs the "Skylake-X" subsets: F, DQ, VL, BW, CD.
inline int cpu_level() {
#if defined(OT1D_USE_SIMD) && (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
#ifdef OT1D_USE_AVX512
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512cd"))
        return AVX512;
#endif
#ifdef OT1D_USE_AVX2
    if (__builtin_cpu_supports("avx2"))
        return AVX2;
#endif
#endif
    return NONE;
}

// Level in use (default: the best available)
inline int& level() {
    static int l = cpu_level();
    return l;
}

// Selects a level (capped to cpu_level(), mainly for testing and benchmarking);
// returns the level in use after the call. Not thread safe.
inline int set_level(int l) {
    int best = cpu_level();
#ifndef OT1D_USE_AVX512
    if (l == AVX512) l = AVX2;
#endif
#ifndef OT1D_USE_AVX2
    if (l == AVX2) l = NONE;
#endif
    level() = (l < NONE) ? NONE : (l > best ? best : l);
    return level();
}

inline bool enabled() { return level() != NONE; }

// Dispatch: call only if enabled()
#define OT1D_SIMD_DISPATCH(call)                       \
    OT1D_SIMD_DISPATCH_512(call)                       \
    OT1D_SIMD_DISPATCH_2(call)

#ifdef OT1D_USE_AVX512
#define OT1D_SIMD_DISPATCH_512(call) if (level() == AVX512) { avx512::call; return; }
#else
#define OT1D_SIMD_DISPATCH_512(call)
#endif
#ifdef OT1D_USE_AVX2
#define OT1D_SIMD_DISPATCH_2(call) if (level() == AVX2) { avx2::call; return; }
#else
#define OT1D_SIMD_DISPATCH_2(call)
#endif

inline void sort(double* x, size_t n) { OT1D_SIMD_DISPATCH(sort(x, n)) }
inline void sort_kv(double* key, double* val, size_t n) { OT1D_SIMD_DISPATCH(sort_kv(key, val, n)) }
inline void sort_kv(double* key, uint64_t* val, size_t n) { OT1D_SIMD_DISPATCH(sort_kv(key, val, n)) }

#undef OT1D_SIMD_DISPATCH
#undef OT1D_SIMD_DISPATCH_512
#undef OT1D_SIMD_DISPATCH_2

} // namespace ot1d_simd

typedef std::pair<double, double>  dbl_pair_t;

struct dbl_idx_t
{
    dbl_idx_t() : xi(0.0), i(-1) {}
    dbl_idx_t(double _x, int _i) : xi(_x), i(_i) {}

    double xi;
    int i;

    bool operator < ( const dbl_idx_t &r ) const {
        return xi < r.xi;
    }
};


struct dbl_pair_idx_t
{
    dbl_pair_idx_t() : xi(0.0), ai(0.0), i(-1) {}
    dbl_pair_idx_t(double _x, double _a, int _i) : xi(_x), ai(_a), i(_i) {}

    double xi;
    double ai;
    int i;

    bool operator < ( const dbl_pair_idx_t &r ) const {
        if ( xi < r.xi ) return true;
        if ( xi > r.xi ) return false;
        return ai < r.ai;
    }
};

// Light (x, a) pair used when no plan is requested: smaller than dbl_pair_idx_t.
struct dbl_w_t {
    dbl_w_t() : xi(0.0), ai(0.0) {}
    dbl_w_t(double _x, double _a, int) : xi(_x), ai(_a) {}
    double xi;
    double ai;

    // Used by parasort_mu to partition the buckets
    bool operator < ( const dbl_w_t &r ) const { return xi < r.xi; }
};

// SIMD (AVX-512 or AVX2) sorting by the support point (xi): specializations of the hook
// _simd_sort of parasort.h, used both for a whole array (single thread) and for
// each bucket of parasort (multiple threads). Items are sorted with a key-value
// sort on separate arrays (keys = xi, values = payload or position), then rebuilt.
// They return false if SIMD sorting is not available, and the caller uses pdqsort.
#ifdef OT1D_USE_SIMD
template <> inline bool _simd_sort<double>(size_t n, double* x) {
    if (!ot1d_simd::enabled()) return false;
    ot1d_simd::sort(x, n);
    return true;
}

template <> inline bool _simd_sort<dbl_pair_t>(size_t n, dbl_pair_t* v) {
    if (!ot1d_simd::enabled()) return false;
    std::vector<double> key(n), val(n);
    for (size_t k = 0; k < n; k++) { key[k] = v[k].first; val[k] = v[k].second; }
    ot1d_simd::sort_kv(key.data(), val.data(), n);
    for (size_t k = 0; k < n; k++) { v[k].first = key[k]; v[k].second = val[k]; }
    return true;
}

template <> inline bool _simd_sort<dbl_w_t>(size_t n, dbl_w_t* v) {
    if (!ot1d_simd::enabled()) return false;
    std::vector<double> key(n), val(n);
    for (size_t k = 0; k < n; k++) { key[k] = v[k].xi; val[k] = v[k].ai; }
    ot1d_simd::sort_kv(key.data(), val.data(), n);
    for (size_t k = 0; k < n; k++) { v[k].xi = key[k]; v[k].ai = val[k]; }
    return true;
}

template <> inline bool _simd_sort<dbl_idx_t>(size_t n, dbl_idx_t* v) {
    if (!ot1d_simd::enabled()) return false;
    std::vector<double> key(n);
    std::vector<uint64_t> val(n);
    for (size_t k = 0; k < n; k++) { key[k] = v[k].xi; val[k] = (uint64_t)v[k].i; }
    ot1d_simd::sort_kv(key.data(), val.data(), n);
    for (size_t k = 0; k < n; k++) { v[k].xi = key[k]; v[k].i = (int)val[k]; }
    return true;
}

template <> inline bool _simd_sort<dbl_pair_idx_t>(size_t n, dbl_pair_idx_t* v) {
    if (!ot1d_simd::enabled()) return false;
    std::vector<double> key(n);
    std::vector<uint64_t> pos(n);
    for (size_t k = 0; k < n; k++) { key[k] = v[k].xi; pos[k] = k; }
    ot1d_simd::sort_kv(key.data(), pos.data(), n);
    std::vector<dbl_pair_idx_t> w(v, v + n);
    for (size_t k = 0; k < n; k++) v[k] = w[pos[k]];
    return true;
}
#endif

namespace ot1d_detail {

// Below this size the parallel sort is not worth the thread overhead
// (and parasort needs at least one element per thread).
const int PARASORT_MIN_SIZE = 10000;

// With SIMD sorting, a single thread sorts plain values faster than parasort
// with any number of threads; for items (key-value sort), parasort with SIMD
// buckets pays off only on large arrays (measured on a Xeon w9-3495X).
const int PARASORT_MIN_SIZE_SIMD = 1000000;

inline bool use_simd() {
#ifdef OT1D_USE_SIMD
    return ot1d_simd::enabled();
#else
    return false;
#endif
}

inline bool use_parasort(int n, int threads) {
    return threads > 1 && n >= (use_simd() ? PARASORT_MIN_SIZE_SIMD : PARASORT_MIN_SIZE);
}

inline void sort_values(int n, double* x, int threads) {
    if (_simd_sort((size_t)n, x))
        return;
    if (use_parasort(n, threads))
        parasort(n, x, threads);
    else
        pdqsort(x, x + n);
}

// parasort sorts each bucket with _simd_sort (AVX-512 or AVX2, if available)
template <typename T, typename Cmp>
inline void sort_items(std::vector<T>& v, Cmp cmp, int threads) {
    int n = (int)v.size();
    if (use_parasort(n, threads))
        parasort_mu(n, v.data(), Cmp(cmp), threads);  // parasort_mu needs an rvalue
    else if (!_simd_sort(v.size(), v.data()))
        pdqsort(v.begin(), v.end(), cmp);
}

template <int P>
inline double cost(double x, double y) {
    double d = x - y;
    return P == 1 ? std::fabs(d) : d * d;
}

template <int P>
inline double finalize(double z) {
    return P == 1 ? z : std::sqrt(z);
}

// Records one entry of the transportation plan, if requested.
struct PlanWriter {
    int* map;
    double* mass;
    int k;

    PlanWriter(int* _map, double* _mass) : map(_map), mass(_mass), k(0) {}

    void add(int i, int j, double w) {
        if (map == nullptr) return;
        map[2 * k] = i;
        map[2 * k + 1] = j;
        mass[k] = w;
        k++;
    }
};

// North-west corner rule on sorted supports with uniform weights 1/m and 1/n.
// Masses are tracked exactly as integers in units of 1/(m*n): each x_i holds
// n units, each y_j holds m units. This avoids any floating point comparison.
template <int P, typename GetX, typename GetY>
inline double uniform_kernel(int m, int n, GetX getx, GetY gety, PlanWriter& plan,
                             const int* ix, const int* iy) {
    const double unit = 1.0 / ((double)m * (double)n);
    int64_t A = n;
    int64_t B = m;
    int i = 0, j = 0;
    double z = 0.0;

    while (i < m && j < n) {
        int64_t t = A < B ? A : B;
        double w = (double)t * unit;
        z += w * cost<P>(getx(i), gety(j));
        plan.add(ix ? ix[i] : i, iy ? iy[j] : j, w);
        A -= t;
        B -= t;
        if (A == 0) { i++; A = n; }
        if (B == 0) { j++; B = m; }
    }
    return z;
}

inline int item_index(const dbl_pair_idx_t& e, int) { return e.i; }
inline int item_index(const dbl_w_t&, int k) { return k; }

// North-west corner rule on sorted supports with arbitrary weights.
// Residuals smaller than eps (relative to the total mass) are treated as zero,
// so that rounding errors do not produce spurious tiny plan entries.
template <int P, typename Item>
inline double weighted_kernel(int m, int n, const std::vector<Item>& mu, const std::vector<Item>& nu,
                              PlanWriter& plan) {
    double total = 0.0;
    for (int i = 0; i < m; i++) total += std::fabs(mu[i].ai);
    for (int j = 0; j < n; j++) total += std::fabs(nu[j].ai);
    const double eps = 1e-13 * total;

    int i = 0, j = 0;
    double A = mu[0].ai;
    double B = nu[0].ai;
    double z = 0.0;

    while (i < m && j < n) {
        double t = A < B ? A : B;
        if (t > 0) {
            z += t * cost<P>(mu[i].xi, nu[j].xi);
            plan.add(item_index(mu[i], i), item_index(nu[j], j), t);
        }
        A -= t;
        B -= t;
        if (A <= eps) { i++; if (i < m) A = mu[i].ai; }
        if (B <= eps) { j++; if (j < n) B = nu[j].ai; }
    }
    return z;
}

inline void set_size(int* map_size, int k) {
    if (map_size != nullptr) map_size[0] = k;
}

// Uniform weights, m and n arbitrary (also m == n).
template <int P>
inline double uniform(int m, int n, const double* x0, const double* y0,
                      int* map, double* mass, int* map_size, bool sorting, int threads) {
    PlanWriter plan(map, mass);
    if (m <= 0 || n <= 0) {
        set_size(map_size, 0);
        return 0.0;
    }

    double z = 0.0;
    if (map == nullptr) {
        // Value only: sort plain copies of the values
        std::vector<double> x(x0, x0 + m);
        std::vector<double> y(y0, y0 + n);
        if (sorting) {
            sort_values(m, x.data(), threads);
            sort_values(n, y.data(), threads);
        }
        if (m == n) {
            for (int i = 0; i < n; i++)
                z += cost<P>(x[i], y[i]);
            z /= n;
        } else {
            z = uniform_kernel<P>(m, n,
                                  [&](int i) { return x[i]; },
                                  [&](int j) { return y[j]; },
                                  plan, nullptr, nullptr);
        }
    } else {
        // Keep track of the original indices to build the plan
        std::vector<dbl_idx_t> mu, nu;
        mu.reserve(m);
        nu.reserve(n);
        for (int i = 0; i < m; i++) mu.emplace_back(x0[i], i);
        for (int j = 0; j < n; j++) nu.emplace_back(y0[j], j);
        if (sorting) {
            auto cmp = [](const dbl_idx_t& v, const dbl_idx_t& w) { return v.xi < w.xi; };
            sort_items(mu, cmp, threads);
            sort_items(nu, cmp, threads);
        }
        std::vector<int> ix(m), iy(n);
        for (int i = 0; i < m; i++) ix[i] = mu[i].i;
        for (int j = 0; j < n; j++) iy[j] = nu[j].i;
        z = uniform_kernel<P>(m, n,
                              [&](int i) { return mu[i].xi; },
                              [&](int j) { return nu[j].xi; },
                              plan, ix.data(), iy.data());
    }

    set_size(map_size, plan.k);
    return finalize<P>(z);
}

template <int P, typename Item>
inline double weighted_sorted(int m, int n, const double* x, const double* y, const double* a, const double* b,
                              PlanWriter& plan, bool sorting, int threads) {
    std::vector<Item> mu, nu;
    mu.reserve(m);
    nu.reserve(n);
    for (int i = 0; i < m; i++) mu.emplace_back(x[i], a[i], i);
    for (int j = 0; j < n; j++) nu.emplace_back(y[j], b[j], j);

    if (sorting) {
        auto cmp = [](const Item& v, const Item& w) { return v.xi < w.xi; };
        sort_items(mu, cmp, threads);
        sort_items(nu, cmp, threads);
    }
    return weighted_kernel<P>(m, n, mu, nu, plan);
}

// Arbitrary weights.
template <int P>
inline double weighted(int m, int n, const double* x, const double* y, const double* a, const double* b,
                       int* map, double* mass, int* map_size, bool sorting, int threads) {
    PlanWriter plan(map, mass);
    if (m <= 0 || n <= 0) {
        set_size(map_size, 0);
        return 0.0;
    }

    double z = (map == nullptr)
        ? weighted_sorted<P, dbl_w_t>(m, n, x, y, a, b, plan, sorting, threads)
        : weighted_sorted<P, dbl_pair_idx_t>(m, n, x, y, a, b, plan, sorting, threads);
    set_size(map_size, plan.k);
    return finalize<P>(z);
}

} // namespace ot1d_detail

// SIMD sorting level supported (compiled in and supported by the CPU):
// 0 = none (pdqsort/parasort), 1 = AVX2, 2 = AVX-512
inline int OT1D_simd_supported() {
#ifdef OT1D_USE_SIMD
    return ot1d_simd::cpu_level();
#else
    return 0;
#endif
}

// Selects the SIMD sorting level (capped to the supported one) and returns the
// level in use. Not thread safe: call it before computing (mainly for benchmarks).
inline int OT1D_set_simd(int level) {
#ifdef OT1D_USE_SIMD
    return ot1d_simd::set_level(level);
#else
    (void)level;
    return 0;
#endif
}

// Utilities: fast sort
inline void _parasort(int n, double* x, int threads=1) {
    if (n <= 0) return;
    ot1d_detail::sort_values(n, x, threads);
}

inline void _parasort_mu(int n, double* x, double *a, int threads=1) {
    if (n <= 0) return;
#ifdef OT1D_USE_SIMD
    if (ot1d_detail::use_simd() && !ot1d_detail::use_parasort(n, threads)) {
        ot1d_simd::sort_kv(x, a, (size_t)n);  // in place, no copies
        return;
    }
#endif
    std::vector<dbl_pair_t> mu;
    mu.reserve(n);

    for (int i = 0; i < n; i++)
        mu.emplace_back(x[i], a[i]);

    ot1d_detail::sort_items(mu, [](const dbl_pair_t& v, const dbl_pair_t& w) {
        return v.first < w.first;
    }, threads);

    for (int i = 0; i < n; i++) {
        x[i] = mu[i].first;
        a[i] = mu[i].second;
    }
}

// Wasserstein-1D, p=1, no weights, m=n (same size)
inline double OT1Da0(int n, double* x, double* y, bool sorting, int threads=1) {
    return ot1d_detail::uniform<1>(n, n, x, y, nullptr, nullptr, nullptr, sorting, threads);
}

// Wasserstein-1D, p=1, no weights, m != n
inline double OT1Da(int m, int n, double* x, double* y, bool sorting, int threads=1) {
    return ot1d_detail::uniform<1>(m, n, x, y, nullptr, nullptr, nullptr, sorting, threads);
}

// Wasserstein-1D, p=2, no weights, m=n (same size)
inline double OT1Db0(int n, double* x, double* y, bool sorting, int threads=1) {
    return ot1d_detail::uniform<2>(n, n, x, y, nullptr, nullptr, nullptr, sorting, threads);
}

// Wasserstein-1D, p=2, no weights
inline double OT1Db(int m, int n, double* x, double* y, bool sorting, int threads=1) {
    return ot1d_detail::uniform<2>(m, n, x, y, nullptr, nullptr, nullptr, sorting, threads);
}

// Wasserstein-1D, p=1, with weights
inline double OT1Dc(int m, int n, double* x, double* y, double* a, double* b, bool sorting, int threads=1) {
    return ot1d_detail::weighted<1>(m, n, x, y, a, b, nullptr, nullptr, nullptr, sorting, threads);
}

// Wasserstein-1D, p=2, with weights
inline double OT1Dd(int m, int n, double* x, double* y, double* a, double* b, bool sorting, int threads=1) {
    return ot1d_detail::weighted<2>(m, n, x, y, a, b, nullptr, nullptr, nullptr, sorting, threads);
}

// Wasserstein-1D, p=1, without weights, n=m
inline double OT1Da0_withplan(int n, double* x, double* y, int* map, double* mass, int* map_size, bool sorting, int threads=1) {
    return ot1d_detail::uniform<1>(n, n, x, y, map, mass, map_size, sorting, threads);
}

// Wasserstein-1D, p=2, without weights, n=m
inline double OT1Db0_withplan(int n, double* x, double* y, int* map, double* mass, int* map_size, bool sorting, int threads=1) {
    return ot1d_detail::uniform<2>(n, n, x, y, map, mass, map_size, sorting, threads);
}

// Wasserstein-1D, p=1, without weights
inline double OT1Da_withplan(int m, int n, double* x, double* y, int* map, double* mass, int* map_size, bool sorting, int threads=1) {
    return ot1d_detail::uniform<1>(m, n, x, y, map, mass, map_size, sorting, threads);
}

// Wasserstein-1D, p=2, without weights
inline double OT1Db_withplan(int m, int n, double* x, double* y, int* map, double* mass, int* map_size, bool sorting, int threads=1) {
    return ot1d_detail::uniform<2>(m, n, x, y, map, mass, map_size, sorting, threads);
}

// Wasserstein-1D, p=1, with weights
inline double OT1Dc_withplan(int m, int n, double* x, double* y, double* a, double* b, int* map, double* mass, int* map_size, bool sorting, int threads=1) {
    return ot1d_detail::weighted<1>(m, n, x, y, a, b, map, mass, map_size, sorting, threads);
}

// Wasserstein-1D, p=2, with weights
inline double OT1Dd_withplan(int m, int n, double* x, double* y, double* a, double* b, int* map, double* mass, int* map_size, bool sorting, int threads=1) {
    return ot1d_detail::weighted<2>(m, n, x, y, a, b, map, mass, map_size, sorting, threads);
}
