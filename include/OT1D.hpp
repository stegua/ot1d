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

namespace ot1d_detail {

// Below this size the parallel sort is not worth the thread overhead
// (and parasort needs at least one element per thread).
const int PARASORT_MIN_SIZE = 10000;

inline bool use_parasort(int n, int threads) {
    return threads > 1 && n >= PARASORT_MIN_SIZE;
}

inline void sort_values(int n, double* x, int threads) {
    if (use_parasort(n, threads))
        parasort(n, x, threads);
    else
        pdqsort(x, x + n);
}

template <typename T, typename Cmp>
inline void sort_items(std::vector<T>& v, Cmp cmp, int threads) {
    int n = (int)v.size();
    if (use_parasort(n, threads))
        parasort_mu(n, v.data(), Cmp(cmp), threads);  // parasort_mu needs an rvalue
    else
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

// Light (x, a) pair used when no plan is requested: smaller than dbl_pair_idx_t.
struct dbl_w_t {
    dbl_w_t() : xi(0.0), ai(0.0) {}
    dbl_w_t(double _x, double _a, int) : xi(_x), ai(_a) {}
    double xi;
    double ai;

    // Used by parasort_mu to partition the buckets
    bool operator < ( const dbl_w_t &r ) const { return xi < r.xi; }
};

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

// Utilities: fast sort
inline void _parasort(int n, double* x, int threads=8) {
    if (n <= 0) return;
    ot1d_detail::sort_values(n, x, threads);
}

inline void _parasort_mu(int n, double* x, double *a, int threads=8) {
    if (n <= 0) return;
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
inline double OT1Da0(int n, double* x, double* y, bool sorting, int threads=8) {
    return ot1d_detail::uniform<1>(n, n, x, y, nullptr, nullptr, nullptr, sorting, threads);
}

// Wasserstein-1D, p=1, no weights, m != n
inline double OT1Da(int m, int n, double* x, double* y, bool sorting, int threads=8) {
    return ot1d_detail::uniform<1>(m, n, x, y, nullptr, nullptr, nullptr, sorting, threads);
}

// Wasserstein-1D, p=2, no weights, m=n (same size)
inline double OT1Db0(int n, double* x, double* y, bool sorting, int threads=8) {
    return ot1d_detail::uniform<2>(n, n, x, y, nullptr, nullptr, nullptr, sorting, threads);
}

// Wasserstein-1D, p=2, no weights
inline double OT1Db(int m, int n, double* x, double* y, bool sorting, int threads=8) {
    return ot1d_detail::uniform<2>(m, n, x, y, nullptr, nullptr, nullptr, sorting, threads);
}

// Wasserstein-1D, p=1, with weights
inline double OT1Dc(int m, int n, double* x, double* y, double* a, double* b, bool sorting, int threads=8) {
    return ot1d_detail::weighted<1>(m, n, x, y, a, b, nullptr, nullptr, nullptr, sorting, threads);
}

// Wasserstein-1D, p=2, with weights
inline double OT1Dd(int m, int n, double* x, double* y, double* a, double* b, bool sorting, int threads=8) {
    return ot1d_detail::weighted<2>(m, n, x, y, a, b, nullptr, nullptr, nullptr, sorting, threads);
}

// Wasserstein-1D, p=1, without weights, n=m
inline double OT1Da0_withplan(int n, double* x, double* y, int* map, double* mass, int* map_size, bool sorting, int threads=8) {
    return ot1d_detail::uniform<1>(n, n, x, y, map, mass, map_size, sorting, threads);
}

// Wasserstein-1D, p=2, without weights, n=m
inline double OT1Db0_withplan(int n, double* x, double* y, int* map, double* mass, int* map_size, bool sorting, int threads=8) {
    return ot1d_detail::uniform<2>(n, n, x, y, map, mass, map_size, sorting, threads);
}

// Wasserstein-1D, p=1, without weights
inline double OT1Da_withplan(int m, int n, double* x, double* y, int* map, double* mass, int* map_size, bool sorting, int threads=8) {
    return ot1d_detail::uniform<1>(m, n, x, y, map, mass, map_size, sorting, threads);
}

// Wasserstein-1D, p=2, without weights
inline double OT1Db_withplan(int m, int n, double* x, double* y, int* map, double* mass, int* map_size, bool sorting, int threads=8) {
    return ot1d_detail::uniform<2>(m, n, x, y, map, mass, map_size, sorting, threads);
}

// Wasserstein-1D, p=1, with weights
inline double OT1Dc_withplan(int m, int n, double* x, double* y, double* a, double* b, int* map, double* mass, int* map_size, bool sorting, int threads=8) {
    return ot1d_detail::weighted<1>(m, n, x, y, a, b, map, mass, map_size, sorting, threads);
}

// Wasserstein-1D, p=2, with weights
inline double OT1Dd_withplan(int m, int n, double* x, double* y, double* a, double* b, int* map, double* mass, int* map_size, bool sorting, int threads=8) {
    return ot1d_detail::weighted<2>(m, n, x, y, a, b, map, mass, map_size, sorting, threads);
}
