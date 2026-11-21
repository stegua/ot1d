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
 *
 * Since v0.5.0 the same functions, and a second algorithm, are also available
 * in namespace ot1d (64-bit sizes, see the end of this file):
 *   - ot1d::w1, ot1d::w2: distance between two samples;
 *   - ot1d::reference: a sample prepared once (sorted, with prefix sums);
 *   - ot1d::w1_reference, ot1d::w2_reference: distance to a reference;
 *   - ot1d::w1_batch, ot1d::w2_batch: one reference against many samples.
 * Every function takes the algorithm as a parameter:
 *   - Method::Simplex: the north-west corner rule (complementary slackness of
 *     the transportation LP), O(n + m) after sorting;
 *   - Method::Ecdf: the quantile form W_p^p = int_0^1 |F^-1(u) - G^-1(u)|^p du,
 *     with the integrals of F^-1 precomputed on the reference, so a query of
 *     size m costs O(m log m) for sorting it plus O(m) (p = 2) or
 *     O(m log(n/m)) (p = 1), independent of the reference size n.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <atomic>
#include <climits>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>   // madvise: transparent huge pages for the buffers of a reference
#endif

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

inline bool use_parasort(int64_t n, int threads) {
    return n <= INT_MAX && use_parasort((int)n, threads);
}

inline void sort_values(int64_t n, double* x, int threads) {
    if (n <= 1 || _simd_sort((size_t)n, x))
        return;
    if (use_parasort(n, threads))
        parasort((int)n, x, threads);
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

// Sorts the pairs (key[i], val[i]) by key; tmp is a reusable buffer.
inline void sort_pairs(int64_t n, double* key, double* val, int threads, std::vector<dbl_pair_t>& tmp) {
    if (n <= 1) return;
#ifdef OT1D_USE_SIMD
    if (use_simd() && !use_parasort(n, threads)) {
        ot1d_simd::sort_kv(key, val, (size_t)n);  // in place, no copies
        return;
    }
#endif
    tmp.resize((size_t)n);
    for (int64_t i = 0; i < n; i++) tmp[i] = dbl_pair_t(key[i], val[i]);
    auto cmp = [](const dbl_pair_t& v, const dbl_pair_t& w) { return v.first < w.first; };
    if (n <= INT_MAX)
        sort_items(tmp, cmp, threads);
    else                                   // sort_items and parasort use int sizes
        pdqsort(tmp.begin(), tmp.end(), cmp);
    for (int64_t i = 0; i < n; i++) {
        key[i] = tmp[i].first;
        val[i] = tmp[i].second;
    }
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

// Value only, uniform weights, x and y sorted (no copies).
template <int P>
inline double uniform_sorted(int m, int n, const double* x, const double* y) {
    if (m == n) {
        double z = 0.0;
        for (int i = 0; i < n; i++) z += cost<P>(x[i], y[i]);
        return z / n;
    }
    PlanWriter plan(nullptr, nullptr);
    return uniform_kernel<P>(m, n, [&](int i) { return x[i]; }, [&](int j) { return y[j]; },
                             plan, nullptr, nullptr);
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
        z = uniform_sorted<P>(m, n, x.data(), y.data());
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
    std::vector<dbl_pair_t> tmp;
    ot1d_detail::sort_pairs(n, x, a, threads, tmp);
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


// =============================================================================
// namespace ot1d (v0.5.0): 64-bit sizes, choice of the algorithm, references
// =============================================================================
//
// Conventions:
//   - x (size n) is the first sample, or the reference; y (size m) the query.
//     a, w (reference) and b (query) are nonnegative weights with positive sum.
//   - Method::Ecdf and every function on a reference treat both measures as
//     probability measures (the weights are normalized). Method::Simplex on two
//     samples keeps the legacy behaviour (partial transport of the smaller mass).
//   - w1 and w2 accept a == nullptr or b == nullptr: that sample has uniform
//     weights, with the same total mass as the other one. With Method::Ecdf,
//     weights a and b whose total masses differ (relative difference above
//     1e-9) throw std::invalid_argument.
//   - Invalid input (empty sample, NaN or inf, negative weights, zero total
//     weight, more than 2^31-1 points with Method::Simplex) throws
//     std::invalid_argument.
//   - A distance that overflows (for instance, the squares in W2) is +inf.
//
// Numerical precision of Method::Ecdf (the rules of the v0.5.0 plan):
//   - with uniform weights, the quantile positions are exact integers in units
//     of 1/(n m), tracked as (index, offset) pairs, so no intermediate exceeds
//     n + m and nothing can overflow; the sum is divided by n m once at the end;
//   - the prefix sums of the reference and the sum over the query are
//     accumulated with Neumaier's compensated summation, in functions compiled
//     without -ffast-math (which would remove the compensation);
//   - the reference is centered by its mean;
//   - known limit of p = 2: W2^2 has an absolute error of order eps * Var(x),
//     so a W2 below about 1e-7 * std(x) (for instance, a query equal to the
//     reference) is returned as about 1e-8 * std(x) instead of 0. p = 1 has no
//     such floor.

#if defined(__clang__)
#define OT1D_PRECISE_ATTR
#define OT1D_PRECISE_SCOPE _Pragma("clang fp reassociate(off)")
#elif defined(__GNUC__)
#define OT1D_PRECISE_ATTR __attribute__((optimize("no-fast-math")))
#define OT1D_PRECISE_SCOPE
#else
#define OT1D_PRECISE_ATTR
#define OT1D_PRECISE_SCOPE
#endif

// Neumaier compensated summation step: (s + c) += x. Use it only inside
// functions marked OT1D_PRECISE_ATTR / OT1D_PRECISE_SCOPE.
#define OT1D_NEUMAIER(s, c, x)                                         \
    do {                                                               \
        double ot1d_x_ = (x);                                          \
        double ot1d_t_ = (s) + ot1d_x_;                                \
        if (std::fabs(s) >= std::fabs(ot1d_x_))                        \
            (c) += ((s) - ot1d_t_) + ot1d_x_;                          \
        else                                                           \
            (c) += (ot1d_x_ - ot1d_t_) + (s);                          \
        (s) = ot1d_t_;                                                 \
    } while (0)

namespace ot1d {

enum class Method { Simplex = 0, Ecdf = 1 };

namespace detail {

// Allocator of the buffers of a reference:
//   - it leaves new elements uninitialized on resize: the buffers are written
//     once by the fit, and zero-filling them first would touch them twice;
//   - on Linux, large buffers are aligned to 2 MB and marked for transparent
//     huge pages (madvise). Measured at n = 10^7: the page faults of the 240 MB
//     of a new reference took about 60 ms of the 165 ms of the fit with 4 KB
//     pages, and the queries, which jump through the buffers, are faster with
//     fewer TLB misses. If huge pages are not available, nothing changes.
template <class T>
struct uninit_alloc {
    typedef T value_type;
    template <class U> struct rebind { typedef uninit_alloc<U> other; };
    uninit_alloc() = default;
    template <class U> uninit_alloc(const uninit_alloc<U>&) {}

    static bool huge(size_t bytes) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        return bytes >= ((size_t)4 << 20);
#else
        (void)bytes;
        return false;
#endif
    }

    T* allocate(size_t k) {
        if (k > (size_t)-1 / sizeof(T)) throw std::bad_alloc();
        const size_t bytes = k * sizeof(T);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        if (huge(bytes)) {
            void* p = nullptr;
            if (posix_memalign(&p, (size_t)2 << 20, bytes) != 0) throw std::bad_alloc();
            madvise(p, bytes, MADV_HUGEPAGE);  // a hint: its failure is harmless
            return static_cast<T*>(p);
        }
#endif
        return static_cast<T*>(::operator new(bytes));
    }

    void deallocate(T* p, size_t k) {
        if (huge(k * sizeof(T)))
            std::free(p);
        else
            ::operator delete(p);
    }

    template <class U> void construct(U* p) { ::new ((void*)p) U; }
    template <class U, class... A> void construct(U* p, A&&... a) { ::new ((void*)p) U(std::forward<A>(a)...); }
    template <class U> bool operator==(const uninit_alloc<U>&) const { return true; }
    template <class U> bool operator!=(const uninit_alloc<U>&) const { return false; }
};

// ---------------------------------------------------------------- validation

// Bit test on the exponent: unlike std::isfinite, it cannot be folded away by
// -ffast-math (-ffinite-math-only) in the caller's translation unit.
inline bool is_finite(double v) {
    uint64_t u;
    std::memcpy(&u, &v, sizeof u);
    return (u & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}

inline bool all_finite(const double* x, int64_t n) {
    for (int64_t i = 0; i < n; i++)
        if (!is_finite(x[i])) return false;
    return true;
}

// Checks the weights and returns their compensated sum.
OT1D_PRECISE_ATTR inline double check_weights(const double* w, int64_t n, const char* what) {
    OT1D_PRECISE_SCOPE
    double s = 0.0, c = 0.0;
    for (int64_t i = 0; i < n; i++) {
        if (!is_finite(w[i]) || w[i] < 0.0)
            throw std::invalid_argument(std::string(what) + ": weights must be finite and nonnegative");
        OT1D_NEUMAIER(s, c, w[i]);
    }
    s += c;
    if (!(s > 0.0))
        throw std::invalid_argument(std::string(what) + ": weights must have a positive sum");
    return s;
}

inline void check_size(int64_t n, const char* what) {
    if (n <= 0)
        throw std::invalid_argument(std::string(what) + ": the sample must be non-empty");
}

[[noreturn]] inline void throw_not_finite(const char* what) {
    throw std::invalid_argument(std::string(what) + ": the sample contains NaN or inf");
}

inline void check_sample(const double* x, int64_t n, const char* what) {
    check_size(n, what);
    if (!all_finite(x, n)) throw_not_finite(what);
}

// Checks x, copies it to r and returns its compensated sum: one pass over x.
// presorted: also checks that x is sorted ascending.
OT1D_PRECISE_ATTR inline double copy_checked_sum(const double* x, int64_t n, double* r, bool presorted,
                                                 const char* what) {
    OT1D_PRECISE_SCOPE
    double s = 0.0, e = 0.0, prev = x[0];
    bool sorted = true;
    for (int64_t i = 0; i < n; i++) {
        if (!is_finite(x[i])) throw_not_finite(what);
        sorted = sorted && !(x[i] < prev);
        prev = r[i] = x[i];
        OT1D_NEUMAIER(s, e, x[i]);
    }
    if (presorted && !sorted)
        throw std::invalid_argument(std::string(what) + ": the sample is not sorted ascending");
    return s + e;
}

// The inputs are finite, so a result that is not finite comes from an overflow
// (for instance, of the squares): as in IEEE 754 arithmetic, it is +inf.
inline double overflow_to_inf(double z) {
    return is_finite(z) ? z : std::numeric_limits<double>::infinity();
}

inline int to_int(int64_t n) {
    if (n > INT_MAX)
        throw std::invalid_argument("ot1d: Method::Simplex supports at most 2^31-1 points per sample");
    return (int)n;
}

// ---------------------------------------------------------------- search

// First index idx >= lo with a[idx] > v (a sorted, a[lo-1] <= v), found by
// galloping from lo: O(log(idx - lo)) for queries that move forward.
inline int64_t gallop_upper(const double* a, int64_t len, int64_t lo, double v) {
    int64_t hi = lo, step = 1;
    while (hi < len && a[hi] <= v) {
        lo = hi + 1;
        hi += step;
        step <<= 1;
    }
    if (hi > len) hi = len;
    return std::upper_bound(a + lo, a + hi, v) - a;
}

// ---------------------------------------------------------------- fit

// Uniform weights, c = mean(r): P1[i] = sum_{l<i} (r_l - c), P2[i] = sum_{l<i} (r_l - c)^2.
OT1D_PRECISE_ATTR inline void fit_uniform(const double* r, int64_t n, double c, double* P1, double* P2) {
    OT1D_PRECISE_SCOPE
    // Each iteration stores at the index it loads from, not one ahead: the
    // buffers have the same alignment (see uninit_alloc), and a store to
    // P1[i + 1] followed by the load of r[i + 1] would share the low 12 bits of
    // the address (4K aliasing), which made this loop 4 times slower.
    double s1 = 0.0, e1 = 0.0, s2 = 0.0, e2 = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double z = r[i] - c;
        P1[i] = s1 + e1;
        P2[i] = s2 + e2;
        OT1D_NEUMAIER(s1, e1, z);
        OT1D_NEUMAIER(s2, e2, z * z);
    }
    P1[n] = s1 + e1;
    P2[n] = s2 + e2;
}

// Weights w (sum wsum): U[i] = sum_{l<i} w_l / wsum (U[n] = 1), c = weighted
// mean, S1[i] = sum_{l<i} w_l (r_l - c) / wsum, S2 likewise with squares.
OT1D_PRECISE_ATTR inline void fit_weighted(const double* r, const double* w, int64_t n, double wsum,
                                           double& c, double* U, double* S1, double* S2) {
    OT1D_PRECISE_SCOPE
    double s = 0.0, e = 0.0;
    for (int64_t i = 0; i < n; i++) OT1D_NEUMAIER(s, e, (w[i] / wsum) * r[i]);
    c = s + e;
    double su = 0.0, eu = 0.0, s1 = 0.0, e1 = 0.0, s2 = 0.0, e2 = 0.0;
    for (int64_t i = 0; i < n; i++) {   // stores at the index of the loads, as in fit_uniform
        double wi = w[i] / wsum;
        double z = r[i] - c;
        U[i] = su + eu;
        S1[i] = s1 + e1;
        S2[i] = s2 + e2;
        OT1D_NEUMAIER(su, eu, wi);
        OT1D_NEUMAIER(s1, e1, wi * z);
        OT1D_NEUMAIER(s2, e2, wi * z * z);
    }
    U[n] = 1.0;
    S1[n] = s1 + e1;
    S2[n] = s2 + e2;
}

// Per-thread buffers of a query
struct scratch {
    std::vector<double> y, b;
    std::vector<dbl_pair_t> tmp;
    std::vector<dbl_w_t> mu, nu;
};

} // namespace detail

// A sample prepared once (sorted, centered, with compensated prefix sums) for
// many exact W1 / W2 queries. Memory: 24 bytes per point (33 with weights).
// Queries are const and thread safe: many threads can use one reference.
class reference {
public:
    typedef std::vector<double, detail::uninit_alloc<double>> buffer;

    // Copies and sorts x. threads: threads of the parallel sort.
    // presorted == true: x is already sorted ascending (checked), and is only copied.
    reference(const double* x, int64_t n, int threads = 1, bool presorted = false) {
        fit(x, nullptr, n, threads, presorted);
    }

    // Weighted reference: w nonnegative with positive sum (normalized to 1).
    // presorted == true: x (with w) is already sorted ascending by x.
    reference(const double* x, const double* w, int64_t n, int threads = 1, bool presorted = false) {
        fit(x, w, n, threads, presorted);
    }

    int64_t size() const { return (int64_t)r_.size(); }
    double mean() const { return c_; }
    bool uniform() const { return uniform_; }
    size_t nbytes() const {
        return sizeof(double) * (r_.size() + P1_.size() + P2_.size() + U_.size()) +
               sizeof(uint32_t) * tab_.size();
    }

    // Sorted support points
    const buffer& sorted_values() const { return r_; }

    // Normalized weights of the sorted support points (empty if uniform)
    std::vector<double> weights() const {
        std::vector<double> w;
        if (uniform_) return w;
        w.resize(r_.size());
        for (size_t i = 0; i < r_.size(); i++) w[i] = U_[i + 1] - U_[i];
        return w;
    }

    // Raw data, for the query kernels
    const double* r() const { return r_.data(); }
    const double* P1() const { return P1_.data(); }  // unscaled sums if uniform, else scaled by the weights
    const double* P2() const { return P2_.data(); }
    const double* U() const { return U_.data(); }    // cumulative weights (empty if uniform)
    double total_weight() const { return wsum_; }    // sum of w before the normalization (1 if uniform)
    // Weighted reference with n < 2^32: index()[k], k = 0 .. index_size() + 1, is the
    // point that owns the position k / index_size(), see detail::ecdf_eval::owner
    const uint32_t* index() const { return tab_.data(); }
    int64_t index_size() const { return tab_.empty() ? 0 : (int64_t)tab_.size() - 2; }

private:
    buffer r_, P1_, P2_, U_;
    std::vector<uint32_t, detail::uninit_alloc<uint32_t>> tab_;
    double c_ = 0.0, wsum_ = 1.0;
    bool uniform_ = true;

    void fit(const double* x, const double* w, int64_t n, int threads, bool presorted) {
        detail::check_size(n, "reference");
        uniform_ = (w == nullptr);
        r_.resize((size_t)n);
        double sum = detail::copy_checked_sum(x, n, r_.data(), presorted, "reference");
        P1_.resize((size_t)n + 1);
        P2_.resize((size_t)n + 1);
        if (w == nullptr) {
            if (!presorted) ot1d_detail::sort_values(n, r_.data(), threads);
            c_ = sum / (double)n;
            detail::fit_uniform(r_.data(), n, c_, P1_.data(), P2_.data());
            return;
        }
        wsum_ = detail::check_weights(w, n, "reference");
        std::vector<double> ws;
        if (!presorted) {
            ws.assign(w, w + n);
            std::vector<dbl_pair_t> tmp;
            ot1d_detail::sort_pairs(n, r_.data(), ws.data(), threads, tmp);
            w = ws.data();
        }
        U_.resize((size_t)n + 1);
        detail::fit_weighted(r_.data(), w, n, wsum_, c_, U_.data(), P1_.data(), P2_.data());
        if (n <= (int64_t)UINT32_MAX) {            // one entry every 4 points: 1 byte per point
            const int64_t K = n < 4 ? 1 : n / 4;
            tab_.resize((size_t)K + 2);
            const double* U = U_.data();
            int64_t i = 0;
            for (int64_t k = 0; k <= K + 1; k++) {
                const double u = (double)k / (double)K;
                while (i < n - 1 && U[i + 1] <= u) i++;
                tab_[k] = (uint32_t)i;
            }
        }
    }
};

namespace detail {

// ------------------------------------------------- ecdf, uniform weights on both

// Position on the integer quantile axis [0, n m]: pos = i m + off, 0 <= off < m.
// Reference point i owns [i m, (i+1) m), query point j owns [j n, (j+1) n).
// N Qk(pos / N) = m Pk[i] + off z_i^k  (N = n m, z_i = r_i - c).
template <int P>
OT1D_PRECISE_ATTR double ecdf_uniform(const reference& R, const double* y, int64_t m) {
    OT1D_PRECISE_SCOPE
    const int64_t n = R.size();
    const double* r = R.r();
    const double* P1 = R.P1();
    const double* P2 = R.P2();
    const double c = R.mean();
    const double md = (double)m, nd = (double)n;
    const int64_t dq = n / m, dr = n % m;  // a query interval has length n = dq m + dr

    int64_t ia = 0, oa = 0;    // a = ia m + oa (start of the interval of y_j)
    double q1a = 0.0, q2a = 0.0;
    double S = 0.0, E = 0.0;

    for (int64_t j = 0; j < m; j++) {
        int64_t ib = ia + dq, ob = oa + dr;  // b = a + n
        if (ob >= m) { ib++; ob -= m; }
        double q1b, q2b = 0.0;
        if (ob == 0) {         // also covers b = N (ib = n)
            q1b = md * P1[ib];
            if (P == 2) q2b = md * P2[ib];
        } else {
            double z = r[ib] - c, o = (double)ob;
            q1b = md * P1[ib] + o * z;
            if (P == 2) q2b = md * P2[ib] + o * z * z;
        }
        const double yc = y[j] - c;
        double t;
        if (P == 2) {
            t = (q2b - q2a) - 2.0 * yc * (q1b - q1a) + yc * yc * nd;
        } else {
            // F^-1 - y_j changes sign at s = clip(k m, a, b), k = #{r_i <= y_j}.
            // k m <= a iff k <= ia iff r[ia] > y_j; k m >= b iff r[hi] <= y_j,
            // hi = ib - 1 if ob == 0, else ib. Only in between is k searched,
            // in r[ia+1 .. hi), about n / m points.
            const double v = y[j];
            const int64_t hi = (ob == 0) ? ib - 1 : ib;
            if (r[ia] > v) {                                // s = a
                t = (q1b - q1a) - yc * nd;
            } else if (r[hi] <= v) {                        // s = b
                t = yc * nd - (q1b - q1a);
            } else {                                        // a < s = k m < b
                const int64_t k = std::upper_bound(r + ia + 1, r + hi, v) - r;
                double q1s = md * P1[k];
                double sa = (double)((k - ia) * m - oa);    // s - a, at most n
                t = yc * sa - (q1s - q1a) + (q1b - q1s) - yc * (nd - sa);
            }
        }
        OT1D_NEUMAIER(S, E, t);
        ia = ib; oa = ob;
        q1a = q1b; q2a = q2b;
    }
    double z = (S + E) / (nd * md);
    if (P == 1) return z;
    return std::sqrt(z < 0.0 ? 0.0 : z);  // rounding below 0 only: NaN (overflow) is kept for `overflow_to_inf`
}

// ------------------------------------------------- ecdf, weights on either side

// Q1, Q2 of the reference at u in [0, 1] (floating point positions).
struct ecdf_eval {
    const double *r, *P1, *P2, *U;
    double c, nd, scale;   // scale: 1/n for the unscaled prefix sums of a uniform reference
    int64_t n;
    bool uniform;
    const uint32_t* tab;   // index of U (weighted reference), see reference::index()
    int64_t K;
    int64_t cur = 0;       // cursor for nondecreasing u, used without the index

    explicit ecdf_eval(const reference& R)
        : r(R.r()), P1(R.P1()), P2(R.P2()), U(R.U()), c(R.mean()), nd((double)R.size()),
          scale(R.uniform() ? 1.0 / (double)R.size() : 1.0), n(R.size()), uniform(R.uniform()),
          tab(R.index()), K(R.index_size()) {}

    // Weighted reference: the point i that owns u, U[i] <= u < U[i+1] (i <= n-1).
    // A search that starts from the previous position must wait for its cache
    // miss before the next one can start (about 80 ns per query point, measured
    // at n = 10^7). The index gives a short range [tab[k], tab[k+1]] from u
    // alone, searched without branches, so the misses of successive query
    // points overlap (about 4 times faster).
    int64_t owner(double u) {
        if (K == 0) {      // no index (n >= 2^32): gallop from the cursor
            cur = gallop_upper(U, n, cur + 1, u) - 1;
            return cur;
        }
        int64_t k = (int64_t)(u * (double)K);
        if (k > K) k = K;
        if (k < 0) k = 0;
        int64_t lo = tab[k], hi = (int64_t)tab[k + 1] + 1;
        while (lo > 0 && U[lo] > u) lo--;          // rounding of u K
        const double* base = U + lo;               // last index in [lo, hi) with U <= u
        for (int64_t len = hi - lo; len > 1;) {
            int64_t half = len / 2;
            base = (base[half] <= u) ? base + half : base;
            len -= half;
        }
        return base - U;
    }

    // F(y) and Q1(F(y)) for k = #{r_i <= y}
    double cdf(int64_t k) const { return uniform ? (double)k / nd : U[k]; }
    double q1_at(int64_t k) const { return P1[k] * scale; }

    // Returns the index i of the reference point that owns u (U[i] <= u < U[i+1],
    // i <= n-1); inside: u > U[i], that is, r_i has mass below u.
    template <int P>
    int64_t at(double u, double& q1, double& q2, bool& inside) {
        int64_t i;
        double ui;
        if (uniform) {
            i = (int64_t)(u * nd);
            if (i > n - 1) i = n - 1;
            if (i < 0) i = 0;
            ui = (double)i / nd;
        } else {
            i = owner(u);
            ui = U[i];
        }
        double z = r[i] - c, d = u - ui;
        inside = d > 0.0;
        q1 = P1[i] * scale + d * z;
        if (P == 2) q2 = P2[i] * scale + d * z * z;
        return i;
    }
};

// b == nullptr: uniform query weights, else weights with sum bsum. Query sorted.
template <int P>
OT1D_PRECISE_ATTR double ecdf_weighted(const reference& R, const double* y, const double* b, double bsum,
                                       int64_t m) {
    OT1D_PRECISE_SCOPE
    ecdf_eval ev(R);
    const double* r = ev.r;
    const double md = (double)m;

    double a = 0.0, q1a = 0.0, q2a = 0.0;
    bool inside;
    int64_t ia = ev.at<P>(a, q1a, q2a, inside);   // reference point that owns a
    double cs = 0.0, ce = 0.0;   // compensated cumulative query weight
    double S = 0.0, E = 0.0;

    for (int64_t j = 0; j < m; j++) {
        double bpos;
        if (j == m - 1) {
            bpos = 1.0;
        } else if (b == nullptr) {
            bpos = (double)(j + 1) / md;
        } else {
            OT1D_NEUMAIER(cs, ce, b[j]);
            bpos = (cs + ce) / bsum;
            if (bpos > 1.0) bpos = 1.0;
            if (bpos < a) bpos = a;
        }
        const double yc = y[j] - ev.c;
        double t, q1b, q2b = 0.0;
        const int64_t ib = ev.at<P>(bpos, q1b, q2b, inside);
        if (P == 2) {
            t = (q2b - q2a) - 2.0 * yc * (q1b - q1a) + yc * yc * (bpos - a);
        } else {
            // As in ecdf_uniform: F^-1 - y_j changes sign at s = clip(F(y_j), a, bpos).
            // The reference points with mass in [a, bpos] are r[ia .. hi]: s = a if
            // r[ia] > y_j, s = bpos if r[hi] <= y_j; only in between is
            // k = #{r_i <= y_j} searched, in r[ia+1 .. hi), and then s = F(y_j) = U[k].
            const double v = y[j];
            const int64_t hi = inside ? ib : ib - 1;
            if (r[ia] > v) {                                // s = a
                t = (q1b - q1a) - yc * (bpos - a);
            } else if (hi <= ia || r[hi] <= v) {            // s = bpos
                t = yc * (bpos - a) - (q1b - q1a);
            } else {
                const int64_t k = std::upper_bound(r + ia + 1, r + hi, v) - r;
                const double sp = ev.cdf(k), q1s = ev.q1_at(k);
                t = yc * (sp - a) - (q1s - q1a) + (q1b - q1s) - yc * (bpos - sp);
            }
        }
        OT1D_NEUMAIER(S, E, t);
        a = bpos; q1a = q1b; q2a = q2b; ia = ib;
    }
    double z = S + E;
    if (P == 1) return z;
    return std::sqrt(z < 0.0 ? 0.0 : z);  // rounding below 0 only: NaN (overflow) is kept for `overflow_to_inf`
}

// ------------------------------------------------- simplex on a reference

template <int P>
double simplex_reference(const reference& R, const double* y, const double* b, double bsum, int64_t m,
                         scratch& s) {
    const int nr = to_int(R.size()), mq = to_int(m);
    const double* r = R.r();
    double z;
    if (R.uniform() && b == nullptr) {
        z = ot1d_detail::uniform_sorted<P>(nr, mq, r, y);
    } else {
        // Probability measures on both sides
        s.mu.resize(nr);
        s.nu.resize(mq);
        if (R.uniform()) {
            for (int i = 0; i < nr; i++) s.mu[i] = dbl_w_t(r[i], 1.0 / nr, i);
        } else {
            const double* U = R.U();
            for (int i = 0; i < nr; i++) s.mu[i] = dbl_w_t(r[i], U[i + 1] - U[i], i);
        }
        for (int j = 0; j < mq; j++)
            s.nu[j] = dbl_w_t(y[j], b != nullptr ? b[j] / bsum : 1.0 / mq, j);
        ot1d_detail::PlanWriter plan(nullptr, nullptr);
        z = ot1d_detail::weighted_kernel<P>(nr, mq, s.mu, s.nu, plan);
    }
    return ot1d_detail::finalize<P>(z);
}

// ------------------------------------------------- dispatch

// Query already validated; bsum: sum of the weights b (unused if b == nullptr).
template <int P>
double run(const reference& R, const double* y, const double* b, double bsum, int64_t m,
           Method method, bool presorted, scratch& s) {
    if (!presorted) {
        s.y.assign(y, y + m);
        if (b != nullptr) {
            s.b.assign(b, b + m);
            ot1d_detail::sort_pairs(m, s.y.data(), s.b.data(), 1, s.tmp);
            b = s.b.data();
        } else {
            ot1d_detail::sort_values(m, s.y.data(), 1);
        }
        y = s.y.data();
    }
    double z;
    if (method == Method::Simplex) z = simplex_reference<P>(R, y, b, bsum, m, s);
    else if (R.uniform() && b == nullptr) z = ecdf_uniform<P>(R, y, m);
    else z = ecdf_weighted<P>(R, y, b, bsum, m);
    return overflow_to_inf(z);
}

template <int P>
double query(const reference& R, const double* y, const double* b, int64_t m,
             Method method, bool presorted, scratch& s) {
    check_sample(y, m, "query");
    const double bsum = (b != nullptr) ? check_weights(b, m, "query") : 1.0;
    return run<P>(R, y, b, bsum, m, method, presorted, s);
}

template <int P>
void batch(const reference& R, const double* y, const double* b, const int64_t* offsets, int64_t k,
           double* out, Method method, int threads) {
    if (k <= 0) return;
    for (int64_t i = 0; i < k; i++)
        if (offsets[i + 1] <= offsets[i])
            throw std::invalid_argument("batch: the sample must be non-empty");
    if (threads < 1) threads = 1;
    if ((int64_t)threads > k) threads = (int)k;
    const unsigned hw = std::thread::hardware_concurrency();  // 0 if unknown
    if (hw > 0 && (unsigned)threads > hw) threads = (int)hw;

    auto one = [&](int64_t i, scratch& s) {
        int64_t lo = offsets[i], len = offsets[i + 1] - offsets[i];
        out[i] = query<P>(R, y + lo, b != nullptr ? b + lo : nullptr, len, method, false, s);
    };
    if (threads == 1) {
        scratch s;
        for (int64_t i = 0; i < k; i++) one(i, s);
        return;
    }
    std::atomic<int64_t> next(0);
    std::atomic<bool> failed(false);
    std::vector<std::thread> pool;
    std::vector<std::exception_ptr> errs((size_t)threads + 1);
    pool.reserve((size_t)threads);
    try {
        for (int t = 0; t < threads; t++) {
            pool.emplace_back([&, t]() {
                scratch s;
                try {
                    for (int64_t i = next++; i < k && !failed; i = next++) one(i, s);
                } catch (...) {
                    errs[t] = std::current_exception();
                    failed = true;
                }
            });
        }
    } catch (...) {
        // A thread could not be started: stop the others and report it
        // (destroying a joinable std::thread would terminate the process).
        errs[threads] = std::current_exception();
        failed = true;
    }
    for (auto& th : pool) th.join();
    for (auto& e : errs)
        if (e) std::rethrow_exception(e);
}

template <int P>
double pair(const double* x, const double* a, int64_t n, const double* y, const double* b, int64_t m,
            Method method, bool sorting, int threads) {
    if (method == Method::Simplex) {
        check_sample(x, n, "x");
        check_sample(y, m, "y");
        int ni = to_int(n), mi = to_int(m);
        if (a == nullptr && b == nullptr)
            return overflow_to_inf(
                ot1d_detail::uniform<P>(ni, mi, x, y, nullptr, nullptr, nullptr, sorting, threads));
        // Weights on one side only: the other side is uniform with the same total mass
        std::vector<double> fill;
        if (a == nullptr) {
            fill.assign((size_t)n, check_weights(b, m, "y") / (double)n);
            a = fill.data();
        } else if (b == nullptr) {
            fill.assign((size_t)m, check_weights(a, n, "x") / (double)m);
            b = fill.data();
        } else {
            check_weights(a, n, "x");
            check_weights(b, m, "y");
        }
        return overflow_to_inf(
            ot1d_detail::weighted<P>(ni, mi, x, y, a, b, nullptr, nullptr, nullptr, sorting, threads));
    }
    // The reference validates x and a (and does not sort them if sorting == false)
    reference R = (a == nullptr) ? reference(x, n, threads, !sorting) : reference(x, a, n, threads, !sorting);
    check_sample(y, m, "y");
    double sb = 1.0;
    if (b != nullptr) {
        sb = check_weights(b, m, "y");
        const double sa = R.total_weight();
        if (a != nullptr && std::fabs(sa - sb) > 1e-9 * std::fmax(sa, sb))
            throw std::invalid_argument("the weights of x and y have different total mass: Method::Ecdf "
                                        "compares probability measures (Method::Simplex transports the "
                                        "smaller mass)");
    }
    scratch s;
    return run<P>(R, y, b, sb, m, Method::Ecdf, !sorting, s);
}

} // namespace detail

// ----------------------------------------------------------------- public API

// W_p between the samples x (size n) and y (size m), uniform weights.
// sorting == false: x and y are already sorted ascending.
inline double w1(const double* x, int64_t n, const double* y, int64_t m,
                 Method method = Method::Simplex, bool sorting = true, int threads = 1) {
    return detail::pair<1>(x, nullptr, n, y, nullptr, m, method, sorting, threads);
}
inline double w2(const double* x, int64_t n, const double* y, int64_t m,
                 Method method = Method::Simplex, bool sorting = true, int threads = 1) {
    return detail::pair<2>(x, nullptr, n, y, nullptr, m, method, sorting, threads);
}

// Weighted samples (x, a) and (y, b); a or b may be nullptr (uniform weights).
inline double w1(const double* x, const double* a, int64_t n, const double* y, const double* b, int64_t m,
                 Method method = Method::Simplex, bool sorting = true, int threads = 1) {
    return detail::pair<1>(x, a, n, y, b, m, method, sorting, threads);
}
inline double w2(const double* x, const double* a, int64_t n, const double* y, const double* b, int64_t m,
                 Method method = Method::Simplex, bool sorting = true, int threads = 1) {
    return detail::pair<2>(x, a, n, y, b, m, method, sorting, threads);
}

// W_p between the reference and the query y (size m, optional weights b).
// presorted == true: y (and b) are already sorted ascending by y, and are not copied.
inline double w1_reference(const reference& ref, const double* y, int64_t m,
                           Method method = Method::Ecdf, bool presorted = false) {
    detail::scratch s;
    return detail::query<1>(ref, y, nullptr, m, method, presorted, s);
}
inline double w2_reference(const reference& ref, const double* y, int64_t m,
                           Method method = Method::Ecdf, bool presorted = false) {
    detail::scratch s;
    return detail::query<2>(ref, y, nullptr, m, method, presorted, s);
}
inline double w1_reference(const reference& ref, const double* y, const double* b, int64_t m,
                           Method method = Method::Ecdf, bool presorted = false) {
    detail::scratch s;
    return detail::query<1>(ref, y, b, m, method, presorted, s);
}
inline double w2_reference(const reference& ref, const double* y, const double* b, int64_t m,
                           Method method = Method::Ecdf, bool presorted = false) {
    detail::scratch s;
    return detail::query<2>(ref, y, b, m, method, presorted, s);
}

// Many queries against one reference. The k samples are stored one after the
// other in y: sample i is y[offsets[i] .. offsets[i+1]) (offsets has k+1
// entries), and out[i] receives its distance. b (optional, nullptr for uniform
// weights) has the same layout as y. The samples are split over `threads` threads.
inline void w1_batch(const reference& ref, const double* y, const int64_t* offsets, int64_t k, double* out,
                     Method method = Method::Ecdf, int threads = 1) {
    detail::batch<1>(ref, y, nullptr, offsets, k, out, method, threads);
}
inline void w2_batch(const reference& ref, const double* y, const int64_t* offsets, int64_t k, double* out,
                     Method method = Method::Ecdf, int threads = 1) {
    detail::batch<2>(ref, y, nullptr, offsets, k, out, method, threads);
}
inline void w1_batch(const reference& ref, const double* y, const double* b, const int64_t* offsets, int64_t k,
                     double* out, Method method = Method::Ecdf, int threads = 1) {
    detail::batch<1>(ref, y, b, offsets, k, out, method, threads);
}
inline void w2_batch(const reference& ref, const double* y, const double* b, const int64_t* offsets, int64_t k,
                     double* out, Method method = Method::Ecdf, int threads = 1) {
    detail::batch<2>(ref, y, b, offsets, k, out, method, threads);
}

} // namespace ot1d
