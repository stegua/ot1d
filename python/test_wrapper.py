# -*- coding: utf-8 -*-
"""
Correctness tests for the OT1D Python wrapper.

Compares OT1D against scipy (W1) and against an independent quantile-based
reference (W1 and W2), checks transportation plans, and input validation.

Run (after `make buildpython`):  cd python && python3 test_wrapper.py
"""
import warnings

import numpy as np
from scipy.stats import wasserstein_distance

from OT1D import OT1D, parasort


def reference(x, y, a, b, p):
    ix, iy = np.argsort(x), np.argsort(y)
    x, a, y, b = x[ix], a[ix], y[iy], b[iy]
    cu, cv = np.concatenate([[0], np.cumsum(a)]), np.concatenate([[0], np.cumsum(b)])
    t = np.unique(np.concatenate([cu, cv]))
    t = t[t <= min(cu[-1], cv[-1])]
    mid = (t[:-1] + t[1:]) / 2
    i = np.clip(np.searchsorted(cu, mid, side='right') - 1, 0, len(x) - 1)
    j = np.clip(np.searchsorted(cv, mid, side='right') - 1, 0, len(y) - 1)
    z = np.sum(np.diff(t) * np.abs(x[i] - y[j]) ** p)
    return z if p == 1 else np.sqrt(z)


def check_plan(x, y, a, b, p, z, T, W):
    m, n = len(x), len(y)
    assert T.shape == (len(W), 2) and len(W) <= m + n - 1
    assert np.allclose(np.bincount(T[:, 0], W, minlength=m), a, rtol=1e-9, atol=1e-12)
    assert np.allclose(np.bincount(T[:, 1], W, minlength=n), b, rtol=1e-9, atol=1e-12)
    c = np.sum(W * np.abs(x[T[:, 0]] - y[T[:, 1]]) ** p)
    assert np.isclose(c if p == 1 else np.sqrt(c), z, rtol=1e-9)


def test_values_and_plans():
    rng = np.random.default_rng(13)
    for _ in range(300):
        m, n = rng.integers(1, 60, size=2)
        if rng.random() < 0.3:
            x, y = rng.integers(0, 5, m).astype(float), rng.integers(0, 5, n).astype(float)
        else:
            x, y = rng.normal(size=m), rng.normal(1, 2, size=n)
        weighted = rng.random() < 0.5
        if weighted:
            a, b = rng.uniform(0.1, 2, m), rng.uniform(0.1, 2, n)
            a, b = a / a.sum(), b / b.sum()
            A, B = a, b
        else:
            a, b = np.full(m, 1 / m), np.full(n, 1 / n)
            A = B = None
        threads = int(rng.choice([1, 4]))
        for p in (1, 2):
            ref = reference(x, y, a, b, p)
            z = OT1D(x, y, A, B, p=p, threads=threads)
            assert np.isclose(z, ref, rtol=1e-9), (m, n, weighted, p, z, ref)
            if p == 1:
                assert np.isclose(z, wasserstein_distance(x, y, a, b), rtol=1e-9)
            z2, T, W = OT1D(x, y, A, B, p=p, threads=threads, plan=True)
            assert np.isclose(z2, ref, rtol=1e-9), (m, n, weighted, p, z2, ref)
            check_plan(x, y, a, b, p, z2, T, W)


def test_large_parallel():
    rng = np.random.default_rng(7)
    x, y = rng.uniform(0, 1, 200000), rng.uniform(1, 2, 150000)
    for p in (1, 2):
        ref = reference(x, y, np.full(len(x), 1 / len(x)), np.full(len(y), 1 / len(y)), p)
        assert np.isclose(OT1D(x, y, p=p, threads=8), ref, rtol=1e-9)
        z, T, W = OT1D(x, y, p=p, threads=8, plan=True)
        assert np.isclose(z, ref, rtol=1e-9)


def test_inputs_not_modified_and_lists():
    x, y = [3.0, 1.0, 2.0], [0.0, 5.0, 4.0, 1.0]
    X, Y = np.array(x), np.array(y)
    assert np.isclose(OT1D(x, y), OT1D(X, Y))
    assert list(X) == x and list(Y) == y


def expect_error(**kw):
    args = dict(X=[1.0, 2.0], Y=[0.0, 3.0])
    args.update(kw)
    try:
        OT1D(args.pop('X'), args.pop('Y'), **args)
    except ValueError:
        return
    raise AssertionError('expected ValueError for {}'.format(kw))


def test_validation():
    expect_error(p=3)
    expect_error(A=[0.5, 0.5])                       # B missing
    expect_error(A=[1.0], B=[0.5, 0.5])              # wrong length
    expect_error(A=[1.5, -0.5], B=[0.5, 0.5])        # negative weight
    expect_error(X=[], Y=[1.0])                      # empty


def test_sorting_flag():
    # With sorting=False the inputs are used in the given order: 3->0, 0->3.
    x, y = np.array([3.0, 0.0]), np.array([0.0, 3.0])
    a = b = np.array([0.5, 0.5])
    for p, expected in ((1, 3.0), (2, 3.0)):
        assert np.isclose(OT1D(x, y, p=p, sorting=False), expected)
        assert np.isclose(OT1D(x, y, a, b, p=p, sorting=False), expected)
        assert np.isclose(OT1D(x, y, p=p, sorting=True), 0.0)
    x, y = np.array([3.0, 0.0, 1.0]), np.array([0.0, 3.0])   # m != n
    assert np.isclose(OT1D(x, y, sorting=False), OT1D(x, y, sorting=False, plan=True)[0])
    assert not np.isclose(OT1D(x, y, sorting=False), OT1D(x, y, sorting=True))


def test_input_types():
    # Lists, tuples and integer arrays are converted
    ref = OT1D(np.array([3.0, 1.0, 2.0]), np.array([0.0, 5.0]))
    assert np.isclose(OT1D([3, 1, 2], (0, 5)), ref)
    assert np.isclose(OT1D(np.array([3, 1, 2]), np.array([0, 5])), ref)


def test_parasort():
    rng = np.random.default_rng(3)
    for n in (0, 1, 5, 1000, 50000):
        x = rng.normal(size=n)
        a = x * 10  # weights follow their points
        xs = np.sort(x)
        y = x.copy()
        parasort(y, threads=4)
        assert np.array_equal(y, xs)
        parasort(x, a, threads=4)
        assert np.array_equal(x, xs) and np.allclose(a, xs * 10)
    # In-place sorting needs float64 arrays: anything else raises a clear error
    for bad in ([3.0, 1.0], np.array([3, 1]), np.array([[1.0, 2.0]]), np.array([3.0, 1.0, 2.0])[::2]):
        try:
            parasort(bad)
        except TypeError:
            continue
        raise AssertionError('expected TypeError for {!r}'.format(bad))
    try:
        parasort(np.array([1.0, 2.0]), np.array([1.0]))
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError for different lengths')


def test_unequal_mass_partial_plan():
    x, y = np.array([0.0, 1.0, 2.0]), np.array([0.5, 3.0])
    a, b = np.array([1.0, 1.0, 1.0]), np.array([0.5, 1.0])   # 3 vs 1.5
    for p in (1, 2):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            z, T, W = OT1D(x, y, a, b, p=p, plan=True)
        assert any(issubclass(e.category, RuntimeWarning) for e in w)
        assert np.isclose(z, reference(x, y, a, b, p))
        assert np.isclose(W.sum(), 1.5)
        assert np.all(np.bincount(T[:, 0], W, minlength=3) <= a + 1e-12)
        assert np.allclose(np.bincount(T[:, 1], W, minlength=2), b)


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_') and callable(f):
            f()
            print('ok', name)
    print('All tests passed')
