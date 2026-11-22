/*
 * Correctness tests for OT1D.hpp.
 *
 * Every solver is compared against an independent reference that integrates
 * |F^-1(t) - G^-1(t)|^p over the merged breakpoints of the two cumulative
 * distribution functions. For the "_withplan" variants, the returned plan is
 * also checked for feasibility (marginals) and for cost consistency.
 *
 * Build & run:  make test   (or: make sanitize)
 */

#include "OT1D.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

static int failures = 0;
static int checks = 0;

// Counts the bytes allocated with operator new, to check that the functions
// called on sorted input read the arrays of the caller without copying them.
// Not under the sanitizers, which replace the allocator themselves.
#if defined(__has_feature)  // Clang
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define OT1D_TEST_SANITIZER
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__) || defined(OT1D_TEST_SANITIZER)
#define COUNT_ALLOCATIONS 0
#else
#define COUNT_ALLOCATIONS 1
#include <atomic>
#include <cstdlib>
#include <new>
static std::atomic<long long> allocated(0);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"  // malloc and free do match here
#endif
void* operator new(std::size_t size) {
    allocated += (long long)size;
    void* p = std::malloc(size ? size : 1);
    if (p == nullptr) throw std::bad_alloc();
    return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
#endif

#define CHECK(cond, ...)                                    \
    do {                                                    \
        ++checks;                                           \
        if (!(cond)) {                                      \
            ++failures;                                     \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                   \
            fprintf(stderr, "\n");                          \
        }                                                   \
    } while (0)

static bool close(double a, double b, double tol = 1e-9) {
    return std::fabs(a - b) <= tol * (1.0 + std::fabs(a) + std::fabs(b));
}

// Reference W_p (p in {1,2}) between discrete measures with equal total mass.
static double reference(const std::vector<double>& x, const std::vector<double>& a,
                        const std::vector<double>& y, const std::vector<double>& b, int p) {
    std::vector<std::pair<double, double>> mu, nu;
    for (size_t i = 0; i < x.size(); i++) mu.emplace_back(x[i], a[i]);
    for (size_t j = 0; j < y.size(); j++) nu.emplace_back(y[j], b[j]);
    std::sort(mu.begin(), mu.end());
    std::sort(nu.begin(), nu.end());

    // Cumulative masses with the compensated sum of the library: where long
    // double is a plain double (ARM, MSVC), a plain running sum is off by 1e-13
    // after 30,000 terms, and on heavy-tailed data that mass, times a squared
    // distance of 1e5, is an error of 1e-8 in this reference value.
    std::vector<long double> cu(1, 0.0L), cv(1, 0.0L);
    ot1d_detail::block_sum su, sv;
    for (auto& e : mu) { su.add(e.second); cu.push_back(su.value()); }
    for (auto& e : nu) { sv.add(e.second); cv.push_back(sv.value()); }

    std::vector<long double> t(cu.begin(), cu.end());
    t.insert(t.end(), cv.begin(), cv.end());
    std::sort(t.begin(), t.end());
    long double total = std::min(cu.back(), cv.back());

    long double z = 0.0L;
    for (size_t k = 0; k + 1 < t.size(); k++) {
        long double lo = t[k], hi = std::min(t[k + 1], total);
        if (hi <= lo) continue;
        long double mid = (lo + hi) / 2;
        size_t i = std::upper_bound(cu.begin(), cu.end(), mid) - cu.begin() - 1;
        size_t j = std::upper_bound(cv.begin(), cv.end(), mid) - cv.begin() - 1;
        long double d = std::fabs((long double)mu[i].first - (long double)nu[j].first);
        z += (hi - lo) * (p == 1 ? d : d * d);
    }
    return p == 1 ? (double)z : std::sqrt((double)z);
}

// Check that (map, mass) is a feasible plan for (a, b) and that its cost matches z.
static void check_plan(const char* name, const std::vector<double>& x, const std::vector<double>& a,
                       const std::vector<double>& y, const std::vector<double>& b, int p,
                       const std::vector<int>& map, const std::vector<double>& mass, int size, double z) {
    int m = (int)x.size(), n = (int)y.size();
    CHECK(size >= 0 && size <= m + n - 1, "%s: plan size %d out of [0, %d]", name, size, m + n - 1);
    if (size < 0 || size > m + n - 1) return;

    std::vector<double> ra(m, 0.0), rb(n, 0.0);
    double cost = 0.0;
    for (int k = 0; k < size; k++) {
        int i = map[2 * k], j = map[2 * k + 1];
        CHECK(i >= 0 && i < m && j >= 0 && j < n, "%s: bad index pair (%d,%d)", name, i, j);
        if (i < 0 || i >= m || j < 0 || j >= n) return;
        CHECK(mass[k] >= -1e-12, "%s: negative mass %g", name, mass[k]);
        ra[i] += mass[k];
        rb[j] += mass[k];
        double d = std::fabs(x[i] - y[j]);
        cost += mass[k] * (p == 1 ? d : d * d);
    }
    if (p == 2) cost = std::sqrt(cost);

    for (int i = 0; i < m; i++)
        CHECK(close(ra[i], a[i], 1e-9), "%s: row marginal %d: %.12g vs %.12g", name, i, ra[i], a[i]);
    for (int j = 0; j < n; j++)
        CHECK(close(rb[j], b[j], 1e-9), "%s: col marginal %d: %.12g vs %.12g", name, j, rb[j], b[j]);
    CHECK(close(cost, z, 1e-9), "%s: plan cost %.12g != returned value %.12g", name, cost, z);
}

static void run_case(std::mt19937& gen, int m, int n, bool weighted, int threads) {
    std::uniform_real_distribution<> U(-5, 5), W(0.1, 2.0);
    std::uniform_int_distribution<> I(0, 4);
    bool ties = (gen() % 3 == 0);  // integer support -> many ties

    std::vector<double> x(m), y(n), a(m), b(n);
    for (auto& v : x) v = ties ? I(gen) : U(gen);
    for (auto& v : y) v = ties ? I(gen) : U(gen);

    if (weighted) {
        double sa = 0, sb = 0;
        for (auto& v : a) sa += (v = W(gen));
        for (auto& v : b) sb += (v = W(gen));
        for (auto& v : a) v /= sa;
        for (auto& v : b) v /= sb;
    } else {
        std::fill(a.begin(), a.end(), 1.0 / m);
        std::fill(b.begin(), b.end(), 1.0 / n);
    }

    const double r1 = reference(x, a, y, b, 1);
    const double r2 = reference(x, a, y, b, 2);
    // Inputs are passed as copies: the API must not depend on caller order.
    std::vector<double> X, Y, A, B;
    auto reset = [&]() { X = x; Y = y; A = a; B = b; };

    char tag[128];
    snprintf(tag, sizeof(tag), "m=%d n=%d w=%d t=%d", m, n, (int)weighted, threads);

    std::vector<int> map(2 * (m + n - 1), -1);
    std::vector<double> mass(m + n - 1, 0.0);
    int sz = -1;
    double z;

    if (!weighted) {
        if (m == n) {
            reset(); z = OT1Da0(n, X.data(), Y.data(), true, threads);
            CHECK(close(z, r1), "OT1Da0 %s: %.12g vs ref %.12g", tag, z, r1);
            reset(); z = OT1Db0(n, X.data(), Y.data(), true, threads);
            CHECK(close(z, r2), "OT1Db0 %s: %.12g vs ref %.12g", tag, z, r2);

            reset(); z = OT1Da0_withplan(n, X.data(), Y.data(), map.data(), mass.data(), &sz, true, threads);
            CHECK(close(z, r1), "OT1Da0_withplan %s: %.12g vs ref %.12g", tag, z, r1);
            check_plan("OT1Da0_withplan", x, a, y, b, 1, map, mass, sz, z);
            reset(); z = OT1Db0_withplan(n, X.data(), Y.data(), map.data(), mass.data(), &sz, true, threads);
            CHECK(close(z, r2), "OT1Db0_withplan %s: %.12g vs ref %.12g", tag, z, r2);
            check_plan("OT1Db0_withplan", x, a, y, b, 2, map, mass, sz, z);
        }
        reset(); z = OT1Da(m, n, X.data(), Y.data(), true, threads);
        CHECK(close(z, r1), "OT1Da %s: %.12g vs ref %.12g", tag, z, r1);
        reset(); z = OT1Db(m, n, X.data(), Y.data(), true, threads);
        CHECK(close(z, r2), "OT1Db %s: %.12g vs ref %.12g", tag, z, r2);

        reset(); z = OT1Da_withplan(m, n, X.data(), Y.data(), map.data(), mass.data(), &sz, true, threads);
        CHECK(close(z, r1), "OT1Da_withplan %s: %.12g vs ref %.12g", tag, z, r1);
        check_plan("OT1Da_withplan", x, a, y, b, 1, map, mass, sz, z);
        reset(); z = OT1Db_withplan(m, n, X.data(), Y.data(), map.data(), mass.data(), &sz, true, threads);
        CHECK(close(z, r2), "OT1Db_withplan %s: %.12g vs ref %.12g", tag, z, r2);
        check_plan("OT1Db_withplan", x, a, y, b, 2, map, mass, sz, z);
    }

    // Weighted solvers must also work on uniform weights.
    reset(); z = OT1Dc(m, n, X.data(), Y.data(), A.data(), B.data(), true, threads);
    CHECK(close(z, r1), "OT1Dc %s: %.12g vs ref %.12g", tag, z, r1);
    reset(); z = OT1Dd(m, n, X.data(), Y.data(), A.data(), B.data(), true, threads);
    CHECK(close(z, r2), "OT1Dd %s: %.12g vs ref %.12g", tag, z, r2);

    reset(); z = OT1Dc_withplan(m, n, X.data(), Y.data(), A.data(), B.data(), map.data(), mass.data(), &sz, true, threads);
    CHECK(close(z, r1), "OT1Dc_withplan %s: %.12g vs ref %.12g", tag, z, r1);
    check_plan("OT1Dc_withplan", x, a, y, b, 1, map, mass, sz, z);
    reset(); z = OT1Dd_withplan(m, n, X.data(), Y.data(), A.data(), B.data(), map.data(), mass.data(), &sz, true, threads);
    CHECK(close(z, r2), "OT1Dd_withplan %s: %.12g vs ref %.12g", tag, z, r2);
    check_plan("OT1Dd_withplan", x, a, y, b, 2, map, mass, sz, z);
}

static void test_readme_example() {
    // Example from src/TestCLI.cpp: x={1,5}, a={2,1}; y={5,10,3}, b={1,1,1}.
    std::vector<double> x = {1, 5}, y = {5, 10, 3}, a = {2, 1}, b = {1, 1, 1};
    std::vector<int> map(8);
    std::vector<double> mass(4);
    int sz = 0;
    double z = OT1Dd_withplan(2, 3, x.data(), y.data(), a.data(), b.data(), map.data(), mass.data(), &sz, true, 1);
    // Plan: 1->3 (1), 1->5 (1), 5->10 (1): cost 4 + 16 + 25 = 45
    CHECK(close(z, std::sqrt(45.0)), "README example W2: %.12g vs %.12g", z, std::sqrt(45.0));
    check_plan("README example", x, a, y, b, 2, map, mass, sz, z);
}

static void test_input_not_modified() {
    std::vector<double> x = {3, 1, 2}, y = {0, 5, 4, 1}, a = {1, 1, 1}, b = {.75, .75, .75, .75};
    auto x0 = x, y0 = y, a0 = a, b0 = b;
    OT1Da(3, 4, x.data(), y.data(), true, 2);
    OT1Dc(3, 4, x.data(), y.data(), a.data(), b.data(), true, 2);
    CHECK(x == x0 && y == y0 && a == a0 && b == b0, "inputs were modified");
}

// The parallel sort, called directly so that small arrays also use the threads:
// sizes around the block boundaries, ties, sorted and reversed input, and more
// threads than elements.
static void test_parallel_sort(std::mt19937& gen) {
    std::uniform_real_distribution<> U(0.0, 1.0);
    std::uniform_int_distribution<> D(0, 9);
    auto less_item = [](const dbl_pair_idx_t& v, const dbl_pair_idx_t& w) { return v.xi < w.xi; };
    const int sizes[] = {1, 2, 3, 7, 8, 9, 63, 64, 65, 1000, 4097, 100003};
    const int threads[] = {2, 3, 4, 5, 7, 8, 16, 33};
    for (int n : sizes)
        for (int t : threads)
            for (int kind = 0; kind < 4; kind++) {
                std::vector<double> x(n);
                for (int i = 0; i < n; i++)
                    x[i] = kind == 0 ? U(gen) : kind == 1 ? D(gen) : kind == 2 ? i : n - i;
                std::vector<double> want(x), v(x);
                std::sort(want.begin(), want.end());
                ot1d_detail::parallel_sort(v.data(), (int64_t)n, t, std::less<double>(),
                                           [](double* p, size_t len) { pdqsort(p, p + len); });
                CHECK(v == want, "parallel_sort values n=%d t=%d kind=%d", n, t, kind);

                std::vector<dbl_pair_idx_t> it;
                for (int i = 0; i < n; i++) it.emplace_back(x[i], 2.0 * x[i], i);
                ot1d_detail::parallel_sort(it.data(), (int64_t)n, t, less_item,
                                           [&](dbl_pair_idx_t* p, size_t len) { pdqsort(p, p + len, less_item); });
                bool ok = true;
                std::vector<char> seen(n, 0);
                for (int i = 0; i < n; i++) {
                    // Sorted by key, and every item still carries its own data
                    ok = ok && it[i].xi == want[i] && it[i].i >= 0 && it[i].i < n && x[it[i].i] == it[i].xi &&
                         it[i].ai == 2.0 * it[i].xi && !seen[it[i].i];
                    if (it[i].i >= 0 && it[i].i < n) seen[it[i].i] = 1;
                }
                CHECK(ok, "parallel_sort items n=%d t=%d kind=%d", n, t, kind);
            }
    // Through the public entry points, above the size where the threads are used
    for (int n : {250000, 6000000}) {
        std::vector<double> x(n), a(n);
        for (int i = 0; i < n; i++) { x[i] = (i % 3) ? U(gen) : D(gen); a[i] = x[i] + 1.0; }
        std::vector<double> want(x);
        std::sort(want.begin(), want.end());
        for (int t : {2, 4, 8}) {
            std::vector<double> v(x), k(x), w(a);
            ot1d_detail::sort_values(n, v.data(), t);
            CHECK(v == want, "sort_values n=%d t=%d", n, t);
            std::vector<dbl_pair_t> tmp;
            ot1d_detail::sort_pairs(n, k.data(), w.data(), t, tmp);
            bool ok = k == want;
            for (int i = 0; ok && i < n; i++) ok = w[i] == k[i] + 1.0;
            CHECK(ok, "sort_pairs n=%d t=%d", n, t);
        }
    }
}

#if COUNT_ALLOCATIONS
// With sorted input and no plan, nothing is copied (and nothing is allocated).
static void test_no_copies(std::mt19937& gen) {
    std::uniform_real_distribution<> U(0.1, 1.0);
    const int n = 5000, m = 3000;
    std::vector<double> x(n), y(m), a(n), b(m), e(n);
    for (auto& v : x) v = U(gen);
    for (auto& v : y) v = U(gen);
    for (auto& v : e) v = U(gen);
    std::sort(x.begin(), x.end());
    std::sort(y.begin(), y.end());
    std::sort(e.begin(), e.end());
    double sa = 0, sb = 0;
    for (auto& v : a) { v = U(gen); sa += v; }
    for (auto& v : b) { v = U(gen); sb += v; }
    for (auto& v : a) v /= sa;
    for (auto& v : b) v /= sb;
    double* X = x.data(); double* Y = y.data(); double* A = a.data(); double* B = b.data(); double* E = e.data();
    const ot1d::Method S = ot1d::Method::Simplex;
    ot1d::reference R(X, n), RW(X, A, n);

    std::function<double()> calls[] = {
        [&] { return OT1Da0(n, X, E, false, 1); },
        [&] { return OT1Db0(n, X, E, false, 1); },
        [&] { return OT1Da(n, m, X, Y, false, 1); },
        [&] { return OT1Db(n, m, X, Y, false, 4); },
        [&] { return OT1Dc(n, m, X, Y, A, B, false, 1); },
        [&] { return OT1Dd(n, m, X, Y, A, B, false, 4); },
        [&] { return ot1d::w1(X, n, Y, m, S, false); },
        [&] { return ot1d::w2(X, n, Y, m, S, false); },
        [&] { return ot1d::w1(X, A, n, Y, B, m, S, false); },
        [&] { return ot1d::w2(X, A, n, Y, B, m, S, false); },
        [&] { return ot1d::w1(X, A, n, Y, nullptr, m, S, false); },
        [&] { return ot1d::w2(X, nullptr, n, Y, B, m, S, false); },
        [&] { return ot1d::w1_reference(R, Y, m, S, true); },
        [&] { return ot1d::w2_reference(R, Y, m, S, true); },
        [&] { return ot1d::w1_reference(R, Y, B, m, S, true); },
        [&] { return ot1d::w2_reference(RW, Y, m, S, true); },
        [&] { return ot1d::w1_reference(RW, Y, B, m, S, true); },
        [&] { return ot1d::w1_reference(R, Y, m, ot1d::Method::Ecdf, true); },
        [&] { return ot1d::w2_reference(RW, Y, B, m, ot1d::Method::Ecdf, true); },
    };
    // The counter does see the copies made when the input must be sorted
    long long start = allocated;
    OT1Da(n, m, X, Y, true, 1);
    CHECK(allocated - start >= (long long)((n + m) * sizeof(double)), "the allocation counter does not count");
    int k = 0;
    for (auto& call : calls) {
        const long long before = allocated;
        double z = call();
        const long long bytes = allocated - before;
        CHECK(bytes == 0 && z > 0.0, "call %d on sorted input allocated %lld bytes (value %g)", k, bytes, z);
        k++;
    }
}
#endif

static void test_unequal_mass() {
    // Total masses 3 and 1.5: only 1.5 is transported (partial plan).
    std::vector<double> x = {0, 1, 2}, y = {0.5, 3}, a = {1, 1, 1}, b = {0.5, 1};
    std::vector<int> map(8);
    std::vector<double> mass(4);
    int sz = 0;
    for (int p = 1; p <= 2; p++) {
        double r = reference(x, a, y, b, p);
        double z = p == 1
            ? OT1Dc_withplan(3, 2, x.data(), y.data(), a.data(), b.data(), map.data(), mass.data(), &sz, true, 1)
            : OT1Dd_withplan(3, 2, x.data(), y.data(), a.data(), b.data(), map.data(), mass.data(), &sz, true, 1);
        CHECK(close(z, r), "unequal mass p=%d: %.12g vs ref %.12g", p, z, r);
        double tot = 0;
        for (int k = 0; k < sz; k++) tot += mass[k];
        CHECK(close(tot, 1.5), "unequal mass p=%d: transported %.12g, expected 1.5", p, tot);
    }
}

#ifdef OT1D_USE_SIMD
// The SIMD sorting (alone, and inside the blocks of the parallel sort for large inputs)
// must give the same results as the generic pdqsort path.
static void test_simd_level(int level) {
    if (ot1d_simd::set_level(level) != level) {
        printf("SIMD level %d not available: skipping its tests\n", level);
        return;
    }
    printf("Testing SIMD level %d\n", level);
    std::mt19937 gen(7);
    std::uniform_real_distribution<> U(0.0, 1.0);
    std::uniform_int_distribution<> D(0, 999);  // many ties
    const int sizes[] = {1, 17, 5000, 1200000};
    for (int n : sizes) {
        int m = n + n / 3;
        std::vector<double> X(m), Y(n), A(m), B(n);
        for (int i = 0; i < m; i++) { X[i] = (i % 2) ? U(gen) : D(gen) / 1000.0; A[i] = U(gen) + 0.1; }
        for (int j = 0; j < n; j++) { Y[j] = (j % 2) ? U(gen) : D(gen) / 1000.0; B[j] = U(gen) + 0.1; }
        double sa = 0, sb = 0;
        for (double v : A) sa += v;
        for (double v : B) sb += v;
        for (double& v : B) v *= sa / sb;
        std::vector<int> map(2 * (m + n));
        std::vector<double> mass(m + n);
        int sz = 0;
        for (int t : {1, 8}) {
            double z[2][8];
            for (int avx = 0; avx < 2; avx++) {
                ot1d_simd::set_level(avx == 1 ? level : ot1d_simd::NONE);
                z[avx][0] = OT1Da(m, n, X.data(), Y.data(), true, t);
                z[avx][1] = OT1Db(m, n, X.data(), Y.data(), true, t);
                z[avx][2] = OT1Dc(m, n, X.data(), Y.data(), A.data(), B.data(), true, t);
                z[avx][3] = OT1Dd(m, n, X.data(), Y.data(), A.data(), B.data(), true, t);
                z[avx][4] = OT1Da_withplan(m, n, X.data(), Y.data(), map.data(), mass.data(), &sz, true, t);
                z[avx][5] = OT1Db0_withplan(n, X.data(), Y.data(), map.data(), mass.data(), &sz, true, t);
                z[avx][6] = OT1Dc_withplan(m, n, X.data(), Y.data(), A.data(), B.data(), map.data(), mass.data(), &sz, true, t);
                z[avx][7] = OT1Dd_withplan(m, n, X.data(), Y.data(), A.data(), B.data(), map.data(), mass.data(), &sz, true, t);
            }
            for (int k = 0; k < 8; k++)
                CHECK(close(z[0][k], z[1][k], 1e-8), "simd=%d n=%d t=%d case %d: %.15g != %.15g",
                      level, n, t, k, z[1][k], z[0][k]);

            // In-place sorting utilities
            std::vector<double> x1(X), x2(X), a1(A), a2(A);
            ot1d_simd::set_level(ot1d_simd::NONE);
            _parasort(m, x1.data(), t);
            _parasort_mu(m, a1.data(), x1.data(), t);  // sort A, carrying (sorted) X along
            ot1d_simd::set_level(level);
            _parasort(m, x2.data(), t);
            _parasort_mu(m, a2.data(), x2.data(), t);
            CHECK(a1 == a2, "simd=%d _parasort_mu keys n=%d t=%d", level, m, t);
            // With ties in A, the order of the values may differ: compare the pairs
            std::vector<std::pair<double, double>> p1, p2;
            for (int i = 0; i < m; i++) { p1.emplace_back(a1[i], x1[i]); p2.emplace_back(a2[i], x2[i]); }
            std::sort(p1.begin(), p1.end());
            std::sort(p2.begin(), p2.end());
            CHECK(p1 == p2, "simd=%d _parasort_mu pairs n=%d t=%d", level, m, t);
        }
    }
    ot1d_simd::set_level(ot1d_simd::cpu_level());
}
#endif

// ============================================================================
// namespace ot1d (v0.5.0): w1/w2, reference, w*_reference, w*_batch
// ============================================================================

static const ot1d::Method METHODS[] = {ot1d::Method::Simplex, ot1d::Method::Ecdf};

static const char* method_name(ot1d::Method me) {
    return me == ot1d::Method::Simplex ? "simplex" : "ecdf";
}

// kind: 0 uniform, 1 ties (integers), 2 heavy tails (Student-t, 2 dof), 3 large offset
static std::vector<double> sample(std::mt19937& gen, int n, int kind) {
    std::uniform_real_distribution<> U(-5, 5);
    std::uniform_int_distribution<> I(0, 4);
    std::student_t_distribution<> T(2.0);
    std::normal_distribution<> N(0.0, 1.0);
    std::vector<double> v(n);
    for (auto& e : v) {
        switch (kind) {
            case 0: e = U(gen); break;
            case 1: e = I(gen); break;
            case 2: e = T(gen); break;
            default: e = 1e6 + N(gen); break;
        }
    }
    return v;
}

static std::vector<double> random_weights(std::mt19937& gen, int n, bool zeros) {
    std::uniform_real_distribution<> W(0.1, 2.0);
    std::vector<double> w(n);
    for (auto& e : w) e = (zeros && gen() % 5 == 0) ? 0.0 : W(gen);
    if (zeros) w[gen() % n] = 1.0;  // positive sum
    return w;
}

static std::vector<double> normalized(const std::vector<double>& w) {
    double s = 0;
    for (double v : w) s += v;
    std::vector<double> r(w);
    for (auto& v : r) v /= s;
    return r;
}

// W_p to a reference / between two samples; b == nullptr calls the unweighted overload.
static double wref(int p, const ot1d::reference& R, const double* y, const double* b, int64_t m,
                   ot1d::Method me, bool presorted = false) {
    if (b == nullptr)
        return p == 1 ? ot1d::w1_reference(R, y, m, me, presorted) : ot1d::w2_reference(R, y, m, me, presorted);
    return p == 1 ? ot1d::w1_reference(R, y, b, m, me, presorted) : ot1d::w2_reference(R, y, b, m, me, presorted);
}

static double wpair(int p, const double* x, const double* a, int64_t n, const double* y, const double* b,
                    int64_t m, ot1d::Method me) {
    if (a == nullptr && b == nullptr)
        return p == 1 ? ot1d::w1(x, n, y, m, me) : ot1d::w2(x, n, y, m, me);
    return p == 1 ? ot1d::w1(x, a, n, y, b, m, me) : ot1d::w2(x, a, n, y, b, m, me);
}

static void run_ot1d_case(std::mt19937& gen, int n, int m, int kind) {
    std::vector<double> x = sample(gen, n, kind), y = sample(gen, m, kind);
    std::vector<double> wa = random_weights(gen, n, gen() % 2), wb = random_weights(gen, m, gen() % 2);
    std::vector<double> ua(n, 1.0 / n), ub(m, 1.0 / m);
    std::vector<double> na = normalized(wa), nb = normalized(wb);
    const double tol = 1e-9;

    // Sorted copy of the query, for presorted = true
    std::vector<std::pair<double, double>> q;
    for (int j = 0; j < m; j++) q.emplace_back(y[j], wb[j]);
    std::sort(q.begin(), q.end());
    std::vector<double> ys(m), bs(m);
    for (int j = 0; j < m; j++) { ys[j] = q[j].first; bs[j] = q[j].second; }

    for (int wr = 0; wr < 2; wr++) {
        ot1d::reference R = wr ? ot1d::reference(x.data(), wa.data(), n, 1 + (n % 3))
                               : ot1d::reference(x.data(), n);
        CHECK(R.size() == n && R.uniform() == !wr, "reference n=%d: size/uniform", n);
        CHECK(std::is_sorted(R.sorted_values().begin(), R.sorted_values().end()), "reference not sorted");
        // A reference built from the sorted data, without sorting, answers the same;
        // so do w1 and w2 with sorting = false
        std::vector<double> rs(R.sorted_values().begin(), R.sorted_values().end()), rw = R.weights();
        ot1d::reference Rs = wr ? ot1d::reference(rs.data(), rw.data(), n, 1, true)
                                : ot1d::reference(rs.data(), n, 1, true);
        for (ot1d::Method me : METHODS)
            for (int p = 1; p <= 2; p++) {
                double z = wref(p, R, y.data(), nullptr, m, me), zs = wref(p, Rs, y.data(), nullptr, m, me);
                double zp = p == 1
                    ? ot1d::w1(rs.data(), wr ? rw.data() : nullptr, n, ys.data(), nullptr, m, me, false)
                    : ot1d::w2(rs.data(), wr ? rw.data() : nullptr, n, ys.data(), nullptr, m, me, false);
                CHECK(close(zs, z, 1e-12) && close(zp, z, tol),
                      "presorted reference w%d %s n=%d m=%d wr=%d: %.15g, pair %.15g vs %.15g",
                      p, method_name(me), n, m, wr, zs, zp, z);
            }
        for (int wq = 0; wq < 2; wq++) {
            const std::vector<double>& A = wr ? na : ua;
            const std::vector<double>& B = wq ? nb : ub;
            double r[3] = {0, reference(x, A, y, B, 1), reference(x, A, y, B, 2)};

            for (ot1d::Method me : METHODS) {
                for (int p = 1; p <= 2; p++) {
                    double z = wref(p, R, y.data(), wq ? wb.data() : nullptr, m, me);
                    double zs = wref(p, R, ys.data(), wq ? bs.data() : nullptr, m, me, true);
                    CHECK(z >= 0.0 && zs >= 0.0, "w%d_reference %s n=%d m=%d kind=%d wr=%d wq=%d: negative %.3g %.3g",
                          p, method_name(me), n, m, kind, wr, wq, z, zs);
                    CHECK(close(z, r[p], tol), "w%d_reference %s n=%d m=%d kind=%d wr=%d wq=%d: %.15g vs %.15g",
                          p, method_name(me), n, m, kind, wr, wq, z, r[p]);
                    CHECK(close(zs, r[p], tol), "w%d_reference presorted %s n=%d m=%d kind=%d wr=%d wq=%d: %.15g vs %.15g",
                          p, method_name(me), n, m, kind, wr, wq, zs, r[p]);

                    // Pairwise functions (normalized weights: equal total masses), with
                    // the weights on both sides, and on the weighted side only
                    double zp = wpair(p, x.data(), wr || wq ? A.data() : nullptr, n,
                                      y.data(), wr || wq ? B.data() : nullptr, m, me);
                    double zo = wpair(p, x.data(), wr ? na.data() : nullptr, n,
                                      y.data(), wq ? nb.data() : nullptr, m, me);
                    CHECK(close(zp, r[p], tol) && close(zo, r[p], tol),
                          "w%d %s n=%d m=%d kind=%d wr=%d wq=%d: %.15g, one-sided %.15g vs %.15g",
                          p, method_name(me), n, m, kind, wr, wq, zp, zo, r[p]);
                }
            }
        }
    }
}

static void test_ot1d_batch(std::mt19937& gen) {
    std::uniform_int_distribution<> S(1, 300);
    for (int wr = 0; wr < 2; wr++) {
        int n = 5000;
        std::vector<double> x = sample(gen, n, 2), wa = random_weights(gen, n, false);
        ot1d::reference R = wr ? ot1d::reference(x.data(), wa.data(), n) : ot1d::reference(x.data(), n);
        const int k = 97;
        std::vector<int64_t> off(1, 0);
        for (int i = 0; i < k; i++) off.push_back(off.back() + S(gen));
        std::vector<double> y = sample(gen, (int)off.back(), 2);
        std::vector<double> b = random_weights(gen, (int)off.back(), true);
        for (int wq = 0; wq < 2; wq++)
            for (ot1d::Method me : METHODS)
                for (int p = 1; p <= 2; p++)
                    for (int t : {1, 4, 200}) {
                        std::vector<double> out(k, -1.0);
                        const double* bb = wq ? b.data() : nullptr;
                        if (p == 1) ot1d::w1_batch(R, y.data(), bb, off.data(), k, out.data(), me, t);
                        else ot1d::w2_batch(R, y.data(), bb, off.data(), k, out.data(), me, t);
                        bool same = true;
                        for (int i = 0; i < k; i++) {
                            const double* yi = y.data() + off[i];
                            int64_t mi = off[i + 1] - off[i];
                            double z = wref(p, R, yi, wq ? bb + off[i] : nullptr, mi, me);
                            same = same && (z == out[i]);
                        }
                        CHECK(same, "w%d_batch %s wr=%d wq=%d threads=%d differs from w%d_reference",
                              p, method_name(me), wr, wq, t, p);
                    }
    }
    // Same layout without weights pointer overload
    std::vector<double> x = sample(gen, 100, 0), y = sample(gen, 30, 0), out(3);
    ot1d::reference R(x.data(), 100);
    int64_t off[4] = {0, 10, 20, 30};
    ot1d::w1_batch(R, y.data(), off, 3, out.data());
    CHECK(out[1] == ot1d::w1_reference(R, y.data() + 10, 10), "w1_batch overload without weights");
}

// Many threads querying one reference at the same time (run under TSan by make sanitize).
static void test_ot1d_concurrent(std::mt19937& gen) {
    int n = 20000;
    std::vector<double> x = sample(gen, n, 0);
    ot1d::reference R(x.data(), n);
    std::vector<std::vector<double>> qs;
    std::vector<double> expected;
    for (int i = 0; i < 64; i++) {
        qs.push_back(sample(gen, 100 + i * 7, 0));
        expected.push_back(ot1d::w1_reference(R, qs.back().data(), (int64_t)qs.back().size()) +
                           ot1d::w2_reference(R, qs.back().data(), (int64_t)qs.back().size()));
    }
    std::vector<int> ok(8, 1);
    std::vector<std::thread> pool;
    for (int t = 0; t < 8; t++)
        pool.emplace_back([&, t]() {
            for (int rep = 0; rep < 4; rep++)
                for (size_t i = t; i < qs.size(); i += 3) {
                    double z = ot1d::w1_reference(R, qs[i].data(), (int64_t)qs[i].size()) +
                               ot1d::w2_reference(R, qs[i].data(), (int64_t)qs[i].size());
                    if (z != expected[i]) ok[t] = 0;
                }
        });
    for (auto& th : pool) th.join();
    for (int t = 0; t < 8; t++) CHECK(ok[t], "concurrent queries: thread %d got a different value", t);
}

// Uniform weights: north-west corner in long double with exact integer masses
// (x_i holds m units, y_j holds n units), the ground truth of the precision test.
static double merge_ld(std::vector<double> x, std::vector<double> y, int p) {
    std::sort(x.begin(), x.end());
    std::sort(y.begin(), y.end());
    int64_t n = x.size(), m = y.size(), i = 0, j = 0, ri = m, rj = n;
    long double tot = 0.0L;
    while (i < n && j < m) {
        int64_t t = std::min(ri, rj);
        long double d = std::fabs((long double)x[i] - (long double)y[j]);
        tot += (long double)t * (p == 1 ? d : d * d);
        ri -= t;
        rj -= t;
        if (ri == 0) { i++; ri = m; }
        if (rj == 0) { j++; rj = n; }
    }
    tot /= (long double)n * (long double)m;
    return (double)(p == 1 ? tot : std::sqrt(tot));
}

// Precision on many ties (n = 10^6): fails if the compensated prefix sums are
// lost, for instance when -ffast-math reassociates them (rules 3 and 4).
// Measured: about 1e-14 with compensation, 2e-11 to 9e-10 without.
static void test_ot1d_precision() {
    std::mt19937 gen(5);
    std::normal_distribution<> N(40.0, 12.0);
    const int n = 1000000, m = 10000;
    std::vector<double> x(n), y(m);
    for (auto& v : x) v = std::floor(N(gen));
    for (auto& v : y) v = std::floor(N(gen) + 0.3);
    ot1d::reference R(x.data(), n);
    for (int p = 1; p <= 2; p++) {
        double r = merge_ld(x, y, p);
        double z = p == 1 ? ot1d::w1_reference(R, y.data(), m) : ot1d::w2_reference(R, y.data(), m);
        double err = std::fabs(z - r) / r;
        printf("Precision on ties, n=%d m=%d: W%d relative error %.2e\n", n, m, p, err);
        CHECK(err < 1e-12, "precision on ties W%d: relative error %.3g (compensation lost?)", p, err);
    }
    // The simplex merge: without the block sums of the kernels, the error on
    // these data is about 1e-12 (and 1e-10 with 3*10^7 points).
    std::vector<double> xs(x), y3(n / 3);
    for (auto& v : y3) v = std::floor(N(gen) + 0.3);
    std::sort(xs.begin(), xs.end());
    std::sort(y.begin(), y.end());
    std::sort(y3.begin(), y3.end());
    const std::vector<double>* others[] = {&y, &y3, &xs};
    for (const std::vector<double>* q : others)
        for (int p = 1; p <= 2; p++) {
            std::vector<double> t(*q);
            if (q == &xs)
                for (auto& v : t) v += 1.0;  // same size: the loop without the merge
            const int64_t mq = (int64_t)t.size();
            double r = merge_ld(xs, t, p);
            double z = p == 1 ? ot1d::w1(xs.data(), n, t.data(), mq, ot1d::Method::Simplex, false)
                              : ot1d::w2(xs.data(), n, t.data(), mq, ot1d::Method::Simplex, false);
            double err = std::fabs(z - r) / r;
            CHECK(err < 2e-14, "precision of the simplex merge W%d, m=%lld: relative error %.3g", p, (long long)mq, err);
        }
    // The same with weights. The exact value must not depend on long double,
    // which is a plain double on ARM and with MSVC: the weights are integers
    // (exact in double, also their partial sums, below 2^53), so the reference
    // merge tracks the masses exactly in 64-bit integers, and adds the terms
    // with the compensated sum of the library.
    // Each weight of y is the sum of three consecutive weights of x (multiples
    // of 2^30) plus a small integer, chosen so that after every three points
    // of x a residual mass of 10 to 400 units is left: about 1e-14 of the
    // total mass, tiny but not a rounding error. With the mass tolerance of
    // v0.5.0 (1e-13 of the total mass) these residuals were dropped, and the
    // error was about 1e-11.
    // (Values without ties, so that a mass moved to the next point changes the cost.)
    const int nw = n, mw = n / 3;
    std::vector<double> xw(nw), yw(mw), a(nw), b(mw);
    for (auto& v : xw) v = N(gen);
    for (auto& v : yw) v = N(gen) + 0.3;
    std::sort(xw.begin(), xw.end());
    std::sort(yw.begin(), yw.end());
    std::uniform_int_distribution<int64_t> K(1, 3), C(10, 400);
    const int64_t Q = (int64_t)1 << 30;
    for (auto& v : a) v = (double)(K(gen) * Q);
    int64_t ahead = 0;  // mass of y in excess of the mass of x, after each group of three
    for (int j = 0; j < mw; j++) {
        const int64_t c = C(gen);
        b[j] = a[3 * j] + a[3 * j + 1] + a[3 * j + 2] + (double)(c - ahead);
        ahead = c;
    }
    // Equal total masses: the points of x after the last group, and the last excess
    for (int i = 3 * mw; i < nw; i++) b[mw - 1] += a[i];
    a[nw - 1] += (double)ahead;
    for (int p = 1; p <= 2; p++) {
        int i = 0, j = 0;
        int64_t A = (int64_t)a[0], B = (int64_t)b[0];
        ot1d_detail::block_sum sum;
        while (i < nw && j < mw) {
            const int64_t t = A < B ? A : B;
            const double d = xw[i] - yw[j];
            sum.add((double)t * (p == 1 ? std::fabs(d) : d * d));
            A -= t;
            B -= t;
            if (A == 0) { i++; if (i < nw) A = (int64_t)a[i]; }
            if (B == 0) { j++; if (j < mw) B = (int64_t)b[j]; }
        }
        CHECK(i == nw && j == mw, "weighted reference merge: masses not exhausted together");
        const double r = p == 1 ? sum.value() : std::sqrt(sum.value());
        double z = p == 1 ? ot1d::w1(xw.data(), a.data(), nw, yw.data(), b.data(), mw, ot1d::Method::Simplex, false)
                          : ot1d::w2(xw.data(), a.data(), nw, yw.data(), b.data(), mw, ot1d::Method::Simplex, false);
        double err = std::fabs(z - r) / r;
        printf("Precision of the weighted simplex merge, n=%d m=%d: W%d relative error %.2e\n", nw, mw, p, err);
        CHECK(err < 1e-13, "precision of the weighted simplex merge W%d: relative error %.3g", p, err);
    }
}

static void test_ot1d_invalid() {
    // NaN from its bits: with -ffast-math, Clang treats the constant NAN as undefined
    double x[3] = {1, 2, 3}, bad[3] = {1, 0, 3}, neg[3] = {1, -1, 3}, zero[3] = {0, 0, 0};
    const uint64_t nan_bits = 0x7FF8000000000000ULL;
    std::memcpy(&bad[1], &nan_bits, sizeof nan_bits);
    int thrown = 0;
    auto expect = [&](std::function<void()> f) {
        try { f(); } catch (const std::invalid_argument&) { thrown++; return; }
        CHECK(false, "invalid input #%d did not throw std::invalid_argument", thrown);
    };
    expect([&] { ot1d::reference R(x, 0); });
    expect([&] { ot1d::reference R(bad, 3); });
    expect([&] { ot1d::reference R(x, neg, 3); });
    expect([&] { ot1d::reference R(x, zero, 3); });
    ot1d::reference R(x, 3);
    expect([&] { ot1d::w1_reference(R, x, 0); });
    expect([&] { ot1d::w2_reference(R, bad, 3); });
    expect([&] { ot1d::w1_reference(R, x, neg, 3); });
    expect([&] { ot1d::w1(x, 3, bad, 3); });
    expect([&] { ot1d::w2(x, 3, bad, 3, ot1d::Method::Ecdf); });
    double half[3] = {0.5, 0.5, 0.5};
    expect([&] { ot1d::w1(x, half, 3, x, x, 3, ot1d::Method::Ecdf); });  // different total masses
    double unsorted[3] = {1, 3, 2};
    expect([&] { ot1d::reference U(unsorted, 3, 1, true); });           // presorted, but not sorted
    expect([&] { ot1d::w1(unsorted, 3, x, 3, ot1d::Method::Ecdf, false); });
    // Squares that overflow: +inf, as in IEEE 754 arithmetic
    double big[3] = {1, 2, 1e200};
    // (bit test: this file is compiled with -ffast-math, which may fold z == inf)
    auto is_inf = [](double z) {
        volatile double stored = z;  // as ot1d::detail::is_finite: Clang assumes that z is finite
        const double back = stored;
        uint64_t u;
        std::memcpy(&u, &back, sizeof u);
        return u == 0x7FF0000000000000ULL;
    };
    ot1d::reference B(big, 3);
    for (ot1d::Method me : METHODS)
        CHECK(is_inf(ot1d::w2(x, 3, big, 3, me)) && is_inf(ot1d::w2(big, 3, x, 3, me)) &&
              is_inf(ot1d::w2_reference(B, x, 3, me)) && is_inf(ot1d::w2_reference(B, x, half, 3, me)),
              "w2 %s with overflow is not +inf", method_name(me));
    int64_t off[3] = {0, 2, 2};
    double out[2];
    expect([&] { ot1d::w1_batch(R, x, off, 2, out); });
    CHECK(thrown == 13, "invalid input: %d of 13 cases threw", thrown);

    // Simplex on sorted samples without weights: the input is checked only when
    // the result is not finite, so a NaN or an inf anywhere must reach the result
    // (sizes on both sides of the block lengths of the kernels, equal or not).
    uint64_t bits[3] = {0x7FF8000000000000ULL, 0x7FF0000000000000ULL, 0xFFF0000000000000ULL};  // NaN, +inf, -inf
    const int ns[] = {1, 2, 63, 64, 65, 1023, 1024, 1025, 5000};
    int cases = 0, caught = 0;
    for (int n : ns)
        for (int m : {n, n + 1, 3 * n + 7, 1})
            for (int p = 1; p <= 2; p++) {
                std::vector<double> u(n), v(m);
                for (int i = 0; i < n; i++) u[i] = 0.5 * i;
                for (int j = 0; j < m; j++) v[j] = 0.25 * j + 1.0;
                auto run = [&] {
                    return p == 1 ? ot1d::w1(u.data(), n, v.data(), m, ot1d::Method::Simplex, false)
                                  : ot1d::w2(u.data(), n, v.data(), m, ot1d::Method::Simplex, false);
                };
                double z0 = run();
                CHECK(z0 > 0.0 && !is_inf(z0), "simplex sorted n=%d m=%d: value %g", n, m, z0);
                for (int side = 0; side < 2; side++) {
                    std::vector<double>& w = side == 0 ? u : v;
                    const int len = (int)w.size();
                    for (int pos : {0, len / 2, len - 1})
                        for (uint64_t bt : bits) {
                            const double keep = w[pos];
                            std::memcpy(&w[pos], &bt, sizeof bt);
                            cases++;
                            try { run(); } catch (const std::invalid_argument&) { caught++; }
                            w[pos] = keep;
                        }
                }
                // An overflow is not an invalid input
                u[n - 1] = 1e200;
                double zo = run();
                CHECK(p == 1 ? (zo > 0.0 && !is_inf(zo)) : is_inf(zo), "simplex sorted n=%d m=%d p=%d with 1e200: %g", n, m, p, zo);
            }
    CHECK(caught == cases, "simplex sorted: %d of %d inputs with NaN or inf threw", caught, cases);
    expect([&] { ot1d::w1(x, 0, x, 3, ot1d::Method::Simplex, false); });
    expect([&] { ot1d::w2(x, 3, x, 0, ot1d::Method::Simplex, false); });
}

static void test_ot1d() {
    printf("Testing namespace ot1d (reference, ecdf and simplex)\n");
    std::mt19937 gen(2026);
    const int sizes[] = {1, 2, 3, 5, 7, 64, 257, 1000};
    for (int n : sizes)
        for (int m : sizes)
            run_ot1d_case(gen, n, m, (n + m) % 4);
    const int large[][2] = {{100000, 777}, {2000, 100000}, {30001, 30001}, {12345, 99}};
    for (auto& s : large)
        for (int kind = 0; kind < 4; kind++)
            run_ot1d_case(gen, s[0], s[1], kind);
    std::uniform_int_distribution<> S(1, 40);
    for (int r = 0; r < 300; r++)
        run_ot1d_case(gen, S(gen), S(gen), r % 4);
    test_ot1d_batch(gen);
    test_ot1d_concurrent(gen);
    test_ot1d_invalid();
    test_ot1d_precision();
}

int main() {
    std::mt19937 gen(2021);

    test_readme_example();
    test_input_not_modified();
    test_unequal_mass();
    test_parallel_sort(gen);
#if COUNT_ALLOCATIONS
    test_no_copies(gen);
#endif
#ifdef OT1D_USE_SIMD
    test_simd_level(ot1d_simd::AVX2);
    test_simd_level(ot1d_simd::AVX512);
#endif

    const int sizes[] = {1, 2, 3, 5, 7, 10, 13, 64, 100, 257, 1000, 3001};
    const int threads[] = {1, 4};
    for (int t : threads)
        for (int m : sizes)
            for (int n : sizes)
                for (int w = 0; w < 2; w++)
                    run_case(gen, m, n, w == 1, t);

    // Large sizes exercise the parallel sort.
    const int large[] = {10000, 20000, 50001};
    for (int m : large)
        for (int n : large)
            for (int w = 0; w < 2; w++)
                run_case(gen, m, n, w == 1, 4);

    // Random fuzzing on small sizes (lots of ties and rounding).
    std::uniform_int_distribution<> S(1, 40);
    for (int r = 0; r < 2000; r++)
        run_case(gen, S(gen), S(gen), r % 2, 1 + (r % 3) * 2);

    test_ot1d();

    printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
