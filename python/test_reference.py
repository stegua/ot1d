"""Correctness tests for ot1d.reference, w1_reference, w2_reference, w1_batch, w2_batch (v0.5.0).

Runs against the NumPy prototype by default; set OT1D_REFERENCE=native to test the C++
implementation (module ot1d, built in place or installed). With the native build, every test
also runs with method='simplex', and the native-only tests at the end check pickling, the
release of the GIL, batches and the pairwise functions w1, w2.

    python -m pytest -q test_reference.py                       # prototype
    OT1D_REFERENCE=native python -m pytest -q test_reference.py # C++ build

Ground truth: a long-double north-west-corner merge written here (exact integer masses).
OT1D's existing merge is used only as a sanity check (it is ~1e-11 accurate on some cases).
"""

from __future__ import annotations

import os
import zlib

import numpy as np
import pytest

import ot1d  # the merge (method='simplex') is a sanity check of the ground truth

NATIVE = os.environ.get("OT1D_REFERENCE") == "native"
if NATIVE:
    api = ot1d
    METHODS = ["ecdf", "simplex"]
else:
    import reference_prototype as api
    METHODS = ["ecdf"]

native_only = pytest.mark.skipif(not NATIVE, reason="needs OT1D_REFERENCE=native")


def dist(R, y, p, **kw):
    """W_p between the reference R and the sample y."""
    return (api.w1_reference if p == 1 else api.w2_reference)(R, y, **kw)


def OT1D(x, y, A=None, B=None, p=1):
    """The v0.4 merge (ot1d.w1 / ot1d.w2 with method='simplex')."""
    return (ot1d.w1 if p == 1 else ot1d.w2)(x, y, A, B)

TOL = 1e-11  # relative. Contract: at least as accurate as the v0.4.1 merge (up to 1.7e-11 on ties;
#              the prototype measured <= 1.6e-12 over 20 seeds, ~1e-14 on heavy tails)

GEN = {
    "normal": lambda r, k: r.normal(0, 1, k),
    "lognormal": lambda r, k: r.lognormal(0, 2, k),
    "student_t2": lambda r, k: r.standard_t(2, k),        # heavy tails: the precision stress test
    "ties": lambda r, k: np.floor(r.normal(40, 12, k)),   # many exact ties
    "offset": lambda r, k: 1e6 + r.normal(0, 1, k),       # large offset, small spread
    "constant": lambda r, k: np.full(k, 3.0),
}
SIZES = [(1, 1), (1, 5), (5, 1), (7, 3), (1000, 1000), (10**5, 777), (10**6, 10**4), (2000, 10**5)]


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def longdouble_merge(x, y, p):
    """Independent ground truth: north-west corner in long double with integer masses."""
    x = np.sort(x).astype(np.longdouble)
    y = np.sort(y).astype(np.longdouble)
    n, m = len(x), len(y)
    i = j = 0
    ri, rj = m, n
    tot = np.longdouble(0)
    while i < n and j < m:
        t = min(ri, rj)
        d = abs(x[i] - y[j])
        tot += t * (d if p == 1 else d * d)
        ri -= t
        rj -= t
        if ri == 0:
            i, ri = i + 1, m
        if rj == 0:
            j, rj = j + 1, n
    v = tot / np.longdouble(n * m)
    return float(v if p == 1 else np.sqrt(v))


@pytest.mark.parametrize("kind", list(GEN))
@pytest.mark.parametrize("n,m", SIZES)
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("method", METHODS)
def test_matches_merge(kind, n, m, p, method):
    r = np.random.default_rng(zlib.crc32(f"{kind}-{n}-{m}".encode()))  # deterministic across runs
    x, y = GEN[kind](r, n), GEN[kind](r, m) + (0.0 if kind == "constant" else 0.2)
    # Ground truth: long-double merge. (OT1D's own merge is only ~1e-11 accurate on some of these
    # cases: it accumulates n + m terms in a plain double running sum, built with -ffast-math.)
    got, want = dist(api.reference(x), y, p, method=method), longdouble_merge(x, y, p)
    assert rel(OT1D(x, y, p=p), want) < 1e-10 if want else True  # sanity check of the truth itself
    if want == 0:
        assert abs(got) < 1e-12
    else:
        # The merge (simplex) accumulates n + m terms in a plain running sum: up to 1.7e-11
        assert rel(got, want) < (TOL if method == "ecdf" else 5e-11), (got, want)


@pytest.mark.parametrize("p", [1, 2])
def test_heavy_tail_against_longdouble(p):
    r = np.random.default_rng(1)
    x, y = r.standard_t(2, 10**5), r.standard_t(2, 777) + 0.2
    assert rel(dist(api.reference(x), y, p), longdouble_merge(x, y, p)) < TOL


def test_identical_sample_w1_is_zero():
    x = np.random.default_rng(2).normal(size=10_000)
    assert abs(api.w1_reference(api.reference(x), x.copy())) < 1e-12


def test_identical_sample_w2_floor():
    """Documented limit: W2^2 has an absolute error ~ eps * variance (expansion of (r - y)^2)."""
    x = 5.0 + 3.0 * np.random.default_rng(2).normal(size=10_000)
    assert api.w2_reference(api.reference(x), x.copy()) ** 2 <= 64 * np.finfo(float).eps * np.var(x)


@pytest.mark.parametrize("p", [1, 2])
def test_query_order_and_presorted(p):
    r = np.random.default_rng(3)
    x, y = r.normal(size=50_000), r.normal(0.1, 1, 3_000)
    R = api.reference(x)
    a = dist(R, y, p)
    assert rel(dist(R, r.permutation(y), p), a) < 1e-15
    assert rel(dist(R, np.sort(y), p, sorting=False), a) < 1e-15


@pytest.mark.parametrize("p", [1, 2])
def test_reference_order_irrelevant(p):
    r = np.random.default_rng(4)
    x, y = r.lognormal(size=20_000), r.lognormal(0.2, 1, 500)
    assert rel(dist(api.reference(x), y, p), dist(api.reference(r.permutation(x)), y, p)) < 1e-14


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("wref", [False, True])
@pytest.mark.parametrize("method", METHODS)
def test_weighted(p, wref, method):
    r = np.random.default_rng(5)
    x, y = r.normal(size=5_000), r.normal(0.3, 1, 800)
    wa = r.random(5_000) if wref else np.ones(5_000)
    wb = r.random(800)
    want = OT1D(x, y, wa / wa.sum(), wb / wb.sum(), p=p)
    got = dist(api.reference(x, wa if wref else None), y, p, B=wb, method=method)
    assert rel(got, want) < 1e-11


def test_symmetry_with_merge_for_equal_sizes():
    r = np.random.default_rng(6)
    x, y = r.normal(size=4096), r.normal(1, 2, 4096)
    R = api.reference(x)
    assert rel(api.w1_reference(R, y), np.mean(np.abs(np.sort(x) - np.sort(y)))) < TOL
    assert rel(api.w2_reference(R, y), np.sqrt(np.mean((np.sort(x) - np.sort(y)) ** 2))) < TOL


@pytest.mark.parametrize("bad", [np.array([]), np.array([1.0, np.nan]), np.array([np.inf])])
def test_rejects_bad_reference(bad):
    with pytest.raises(ValueError):
        api.reference(bad)


def test_rejects_bad_query():
    R = api.reference(np.arange(10.0))
    for y in (np.array([]), np.array([np.nan])):
        for p in (1, 2):
            with pytest.raises(ValueError):
                dist(R, y, p)


@pytest.mark.parametrize("p", [1, 2])
def test_large_positions_no_overflow(p):
    """n = 1e7, m = 1e5: n*m = 1e12 fits int64 but n^2*m does not (regression for an overflow)."""
    r = np.random.default_rng(8)
    x, y = r.lognormal(size=10**7), r.lognormal(0.3, 1, 10**5)
    assert rel(dist(api.reference(x), y, p), OT1D(x, y, p=p)) < 1e-10


# ---------------------------------------------------------------------- native build only


@native_only
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("wref", [False, True])
def test_pickle_roundtrip(p, wref):
    import pickle

    r = np.random.default_rng(9)
    x, y = r.lognormal(size=20_000), r.lognormal(0.1, 1, 900)
    R = ot1d.reference(x, r.random(20_000) if wref else None)
    R2 = pickle.loads(pickle.dumps(R))
    assert len(R2) == len(R) and rel(R2.mean, R.mean) < 1e-14
    assert np.array_equal(R2.sorted_values, R.sorted_values)
    assert rel(dist(R2, y, p), dist(R, y, p)) < 1e-14


@native_only
def test_reference_properties():
    x = np.array([3.0, 1.0, 2.0, 2.0])
    R = ot1d.reference(x)
    assert len(R) == 4 and R.mean == 2.0 and R.weights is None
    assert np.array_equal(R.sorted_values, [1.0, 2.0, 2.0, 3.0])
    assert R.nbytes == 8 * (4 + 2 * 5)
    Rw = ot1d.reference([3.0, 1.0, 2.0, 4.0], [1, 1, 1, 5])
    assert np.allclose(Rw.weights, [0.125, 0.125, 0.125, 0.625])  # sorted by value: 1, 2, 3, 4
    assert rel(Rw.mean, (1 + 2 + 3 + 20) / 8) < 1e-15
    with pytest.raises(TypeError):
        ot1d.w1_reference(x, x)
    with pytest.raises(ValueError):
        ot1d.w1_reference(R, x, method="lp")


@native_only
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("weighted", [False, True])
def test_batch_equals_loop(p, method, weighted):
    r = np.random.default_rng(10)
    x = r.standard_t(3, 30_000)
    Ys = [r.standard_t(3, int(k)) + 0.1 for k in r.integers(1, 2_000, 40)]
    Bs = [r.random(len(y)) for y in Ys] if weighted else None
    R = ot1d.reference(x)
    batch = ot1d.w1_batch if p == 1 else ot1d.w2_batch
    loop = np.array([dist(R, y, p, B=None if Bs is None else b, method=method)
                     for y, b in zip(Ys, Bs or [None] * len(Ys))])
    for threads in (1, 4):
        assert np.array_equal(batch(R, Ys, Bs=Bs, method=method, threads=threads), loop)
        assert np.array_equal(batch(x, Ys, Bs=Bs, method=method, threads=threads), loop)
    # 2D input: one sample per row
    Y2 = r.normal(size=(7, 300))
    assert np.array_equal(batch(R, Y2, method=method), np.array([dist(R, y, p, method=method) for y in Y2]))
    assert len(batch(R, [], method=method)) == 0
    with pytest.raises(ValueError):
        batch(R, [Y2[0], np.array([])])
    with pytest.raises(ValueError):
        batch(R, Ys, A=np.ones(3))


@native_only
@pytest.mark.parametrize("p", [1, 2])
def test_pairwise_methods_agree(p):
    r = np.random.default_rng(11)
    x, y = r.lognormal(size=10_000), r.lognormal(0.2, 1, 3_001)
    a, b = r.random(10_000), r.random(3_001)
    f = ot1d.w1 if p == 1 else ot1d.w2
    want = longdouble_merge(x, y, p)
    assert rel(f(x, y), want) < 1e-10
    assert rel(f(x, y, method="ecdf"), want) < TOL
    assert rel(f(np.sort(x), np.sort(y), method="ecdf", sorting=False), want) < TOL
    # Probability measures: both methods agree
    an, bn = a / a.sum(), b / b.sum()
    z = f(x, y, an, bn)
    assert rel(f(x, y, an, bn, method="ecdf"), z) < 1e-11
    # Total mass M != 1: ecdf normalizes, simplex returns the cost of transporting the mass M
    # (as OT1D in v0.4), that is M W1 or sqrt(M) W2
    M = a.sum()
    assert rel(f(x, y, a, bn * M, method="ecdf"), z) < 1e-11
    assert rel(f(x, y, a, bn * M), z * (M if p == 1 else np.sqrt(M))) < 1e-11
    # One-sided weights: the other side is uniform, with the same total mass
    assert rel(f(x, y, None, bn, method="ecdf"), f(x, y, np.full(10_000, 1e-4), bn)) < 1e-11
    assert rel(f(x, y, None, bn, method="ecdf"), f(x, y, None, bn)) < 1e-11
    with pytest.raises(ValueError):
        f(x, y, a, b, method="ecdf")  # different total masses
    with pytest.warns(RuntimeWarning):
        f(x, y, a, b)                 # simplex: partial transport, as in v0.4
    z, T, W = (ot1d.w1_plan if p == 1 else ot1d.w2_plan)(x, y)
    assert rel(z, want) < 1e-10 and T.shape == (len(W), 2)


@native_only
def test_gil_released():
    """A Python thread keeps running while w1_batch runs in another thread."""
    import threading
    import time

    r = np.random.default_rng(12)
    R = ot1d.reference(r.normal(size=10**6))
    Ys = [r.normal(size=10**5) for _ in range(60)]
    ticks = []
    done = threading.Event()

    def ticker():
        while not done.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.001)

    t = threading.Thread(target=ticker)
    t.start()
    time.sleep(0.01)
    n0 = len(ticks)
    t0 = time.perf_counter()
    ot1d.w1_batch(R, Ys, method="simplex")
    elapsed = time.perf_counter() - t0
    done.set()
    t.join()
    assert elapsed > 0.05, "batch too fast to test the GIL"
    assert len(ticks) - n0 > 5, "the ticker thread did not run: the GIL was not released"
