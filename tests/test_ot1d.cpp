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

    std::vector<long double> cu(1, 0.0L), cv(1, 0.0L);
    for (auto& e : mu) cu.push_back(cu.back() + e.second);
    for (auto& e : nu) cv.push_back(cv.back() + e.second);

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
// The SIMD sorting (alone, and inside the buckets of parasort for large inputs)
// must give the same results as the generic pdqsort/parasort path.
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
}

static void test_ot1d_invalid() {
    double x[3] = {1, 2, 3}, bad[3] = {1, NAN, 3}, neg[3] = {1, -1, 3}, zero[3] = {0, 0, 0};
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
        uint64_t u;
        std::memcpy(&u, &z, sizeof u);
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

    // Large sizes exercise the parallel sort (parasort) path.
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
