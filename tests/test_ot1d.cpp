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
#include <cstdio>
#include <random>
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

int main() {
    std::mt19937 gen(2021);

    test_readme_example();
    test_input_not_modified();

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

    printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
