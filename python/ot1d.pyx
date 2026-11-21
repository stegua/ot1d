# distutils: language=c++

# -*- coding: utf-8 -*-
#
# @fileoverview Copyright (c) 2019-2026, Stefano Gualandi,
#                via Ferrata, 1, I-27100, Pavia, Italy
#
#  @author stefano.gualandi@gmail.com (Stefano Gualandi)
#
"""Exact Wasserstein distances W1 and W2 between discrete measures on the real line.

Two algorithms, selected by the parameter ``method`` of every function:

* ``'simplex'``: the north-west corner rule, derived from the complementary
  slackness conditions of the transportation linear program; O(n + m) after
  sorting both samples.
* ``'ecdf'``: the quantile form W_p^p = int_0^1 |F^-1(u) - G^-1(u)|^p du, with the
  integrals of F^-1 precomputed on a fixed :class:`reference`; a query of size m
  costs O(m log m) to sort it plus O(m) (W2) or O(m log(n/m)) (W1), independent of
  the size n of the reference.

Functions:

* ``w1(X, Y)``, ``w2(X, Y)``: distance between two samples (default ``'simplex'``);
* ``w1_plan``, ``w2_plan``: distance and an optimal transportation plan;
* ``reference(X)``: a sample prepared once (sorted, with prefix sums);
* ``w1_reference(ref, Y)``, ``w2_reference(ref, Y)``: distance to a reference
  (default ``'ecdf'``);
* ``w1_batch(X_or_ref, Ys)``, ``w2_batch``: one reference against several samples;
* ``parasort``, ``simd``, ``set_simd``: sorting utilities; ``OT1D``: the v0.4 function.

Every function releases the GIL while running the C++ code.
"""

import warnings

import numpy as np

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libc.stdint cimport int64_t

from _ot1d_cpp cimport (OT1Da0, OT1Db0, OT1Da, OT1Db, OT1Dc, OT1Dd,
                        OT1Da0_withplan, OT1Db0_withplan, OT1Da_withplan, OT1Db_withplan,
                        OT1Dc_withplan, OT1Dd_withplan,
                        OT1D_simd_supported, OT1D_set_simd, _parasort, _parasort_mu,
                        Method, c_reference, c_w1, c_w2,
                        c_w1_reference, c_w2_reference, c_w1_batch, c_w2_batch)

__version__ = "0.5.0"

cdef Method _method(method) except *:
    if method == 'simplex':
        return Method.Simplex
    if method == 'ecdf':
        return Method.Ecdf
    raise ValueError("method must be 'simplex' or 'ecdf', got {!r}".format(method))


def _sample(V, name):
    """1D float64 C-contiguous array (no copy if it already is one)."""
    V = np.ascontiguousarray(V, dtype=float)
    if V.ndim != 1:
        raise ValueError("{} must be one-dimensional".format(name))
    if len(V) == 0:
        raise ValueError("{} must be non-empty".format(name))
    return V


def _weights(W, n, name):
    if W is None:
        return None
    W = np.ascontiguousarray(W, dtype=float)
    if W.ndim != 1 or len(W) != n:
        raise ValueError("{} must be one-dimensional with the same length as its sample".format(name))
    return W  # values (finite, nonnegative, positive sum) are checked in C++


#--------------------------------------------------------------
# Two samples
#--------------------------------------------------------------
def _pair(int p, X, Y, A, B, method, bool sorting, int threads):
    cdef Method me = _method(method)
    X = _sample(X, 'X')
    Y = _sample(Y, 'Y')
    A = _weights(A, len(X), 'A')
    B = _weights(B, len(Y), 'B')
    sa = sb = None
    if A is not None and B is not None:
        sa, sb = A.sum(), B.sum()
        if me == Method.Ecdf and sa > 0 and sb > 0 and not np.isclose(sa, sb, rtol=1e-9, atol=0.0):
            raise ValueError("A and B have different total mass ({} != {}): method='ecdf' compares "
                             "probability measures; use method='simplex' for a partial "
                             "transport".format(sa, sb))

    cdef const double[::1] Xmv = X
    cdef const double[::1] Ymv = Y
    cdef const double[::1] Amv
    cdef const double[::1] Bmv
    cdef const double* a = NULL
    cdef const double* b = NULL
    cdef int64_t n = len(X), m = len(Y)
    cdef double z
    if A is not None:
        Amv = A
        a = &Amv[0]
    if B is not None:
        Bmv = B
        b = &Bmv[0]
    with nogil:
        if p == 1:
            z = c_w1(&Xmv[0], a, n, &Ymv[0], b, m, me, sorting, threads)
        else:
            z = c_w2(&Xmv[0], a, n, &Ymv[0], b, m, me, sorting, threads)
    # The warning of 'simplex' comes after the call, once C++ has validated the weights
    if me == Method.Simplex and sa is not None:
        if not np.isclose(sa, sb, rtol=1e-9, atol=0.0):
            warnings.warn("A and B have different total mass ({} != {}): only the smaller mass "
                          "is transported, and the returned value is partial".format(sa, sb),
                          RuntimeWarning, stacklevel=1)
    return z


def w1(X, Y, A=None, B=None, method='simplex', sorting=True, threads=1):
    """Wasserstein distance of order 1 between the samples X and Y.

    X, Y: support points (1D, any length). A, B: optional nonnegative weights of
    X and Y (if only one is given, the other sample has uniform weights with the
    same total mass). method: 'simplex' (default) or 'ecdf'.

    The two methods treat weights that do not sum to 1 differently:
    'simplex' returns the cost of transporting the total mass M, as OT1D in
    v0.4 (M W1, or sqrt(M) W2, if A and B both sum to M), and with different
    total masses it transports only the smaller one (RuntimeWarning);
    'ecdf' normalizes A and B to probability measures, and raises ValueError
    if their total masses differ. With probability measures (or no weights)
    the two methods return the same value.

    sorting=False: X and Y (with their weights) are already sorted ascending.
    threads: threads of the parallel sort (useful only on large inputs).
    """
    return _pair(1, X, Y, A, B, method, sorting, threads)


def w2(X, Y, A=None, B=None, method='simplex', sorting=True, threads=1):
    """Wasserstein distance of order 2 between the samples X and Y (see w1)."""
    return _pair(2, X, Y, A, B, method, sorting, threads)


def w1_plan(X, Y, A=None, B=None, sorting=True, threads=1):
    """W1 and an optimal transportation plan (method 'simplex').

    Returns (z, T, W): the row T[k] = (i, j) means that the mass W[k] moves from
    X[i] to Y[j]. A and B must be given both or neither.
    """
    return OT1D(X, Y, A, B, p=1, sorting=sorting, threads=threads, plan=True)


def w2_plan(X, Y, A=None, B=None, sorting=True, threads=1):
    """W2 and an optimal transportation plan (see w1_plan)."""
    return OT1D(X, Y, A, B, p=2, sorting=sorting, threads=threads, plan=True)


#--------------------------------------------------------------
# Reference
#--------------------------------------------------------------
cdef class reference:
    """A sample X (optional weights A) prepared once for many exact W1 / W2 queries.

    ref = reference(X, A=None, threads=1, sorting=True) copies and sorts X (threads: threads of
    the parallel sort) and stores the prefix sums used by method='ecdf': 24 bytes
    per point (32 with weights). Query it with w1_reference, w2_reference,
    w1_batch and w2_batch; queries are thread safe and release the GIL.
    Weights are normalized to a probability measure.

    Known limit of W2 with method='ecdf': W2^2 has an absolute error of order
    eps * Var(X), so a W2 below about 1e-7 * std(X) (for instance, a query equal
    to the reference) is returned as about 1e-8 * std(X) instead of 0.

    sorting=False: X (with A) is already sorted ascending, and is only copied
    (ValueError if it is not sorted).

    Picklable (the reference is rebuilt from its sorted values and weights).
    """
    cdef unique_ptr[c_reference] _ref

    def __cinit__(self, X, A=None, int threads=1, bool sorting=True):
        X = _sample(X, 'X')
        A = _weights(A, len(X), 'A')
        cdef const double[::1] Xmv = X
        cdef const double[::1] Amv
        cdef int64_t n = len(X)
        cdef c_reference* r
        if A is None:
            with nogil:
                r = new c_reference(&Xmv[0], n, threads, not sorting)
        else:
            Amv = A
            with nogil:
                r = new c_reference(&Xmv[0], &Amv[0], n, threads, not sorting)
        self._ref.reset(r)

    def __len__(self):
        return self._ref.get().size()

    @property
    def mean(self):
        """Mean of the (weighted) reference."""
        return self._ref.get().mean()

    @property
    def nbytes(self):
        """Memory used by the reference, in bytes."""
        return self._ref.get().nbytes()

    @property
    def sorted_values(self):
        """Sorted support points (a read-only copy)."""
        cdef int64_t n = self._ref.get().size()
        cdef const double* r = self._ref.get().r()
        out = np.empty(n, dtype=float)
        cdef double[::1] o = out
        cdef int64_t i
        with nogil:
            for i in range(n):
                o[i] = r[i]
        out.flags.writeable = False
        return out

    @property
    def weights(self):
        """Normalized weights of the sorted support points, or None if uniform."""
        if self._ref.get().uniform():
            return None
        cdef int64_t n = self._ref.get().size()
        cdef const double* U = self._ref.get().U()
        out = np.empty(n, dtype=float)
        cdef double[::1] o = out
        cdef int64_t i
        with nogil:
            for i in range(n):
                o[i] = U[i + 1] - U[i]
        out.flags.writeable = False
        return out

    def __reduce__(self):
        return (reference, (self.sorted_values, self.weights, 1, False))

    def __repr__(self):
        return "ot1d.reference(n={}, mean={!r}, {})".format(
            len(self), self.mean, "uniform weights" if self._ref.get().uniform() else "weighted")


def _query(int p, reference ref, Y, B, method, bool sorting):
    cdef Method me = _method(method)
    Y = _sample(Y, 'Y')
    B = _weights(B, len(Y), 'B')
    cdef const double[::1] Ymv = Y
    cdef const double[::1] Bmv
    cdef const double* b = NULL
    cdef int64_t m = len(Y)
    cdef double z
    cdef const c_reference* r = ref._ref.get()
    if B is not None:
        Bmv = B
        b = &Bmv[0]
    with nogil:
        if p == 1:
            z = c_w1_reference(r[0], &Ymv[0], b, m, me, not sorting)
        else:
            z = c_w2_reference(r[0], &Ymv[0], b, m, me, not sorting)
    return z


def _as_reference(ref):
    if not isinstance(ref, reference):
        raise TypeError("ref must be an ot1d.reference (build it with ot1d.reference(X, A))")
    return ref


def w1_reference(ref, Y, B=None, method='ecdf', sorting=True):
    """Wasserstein distance of order 1 between the reference ref and the sample Y.

    B: optional nonnegative weights of Y (normalized). method: 'ecdf' (default) or
    'simplex' (merge of the sorted reference with the sorted query, O(n + m)).
    sorting=False: Y (with B) is already sorted ascending and is not copied.
    """
    return _query(1, _as_reference(ref), Y, B, method, sorting)


def w2_reference(ref, Y, B=None, method='ecdf', sorting=True):
    """Wasserstein distance of order 2 between the reference ref and the sample Y (see w1_reference)."""
    return _query(2, _as_reference(ref), Y, B, method, sorting)


def _chunks(Ys, name):
    """List of 1D samples, or a 2D array (one sample per row) -> (buffer, offsets, lengths)."""
    if isinstance(Ys, np.ndarray) and Ys.ndim == 2:
        k, c = Ys.shape
        return (np.ascontiguousarray(Ys, dtype=float).reshape(-1),
                np.arange(k + 1, dtype=np.int64) * c, [c] * k)
    parts = [np.asarray(y, dtype=float) for y in Ys]  # asarray keeps scalars 0-dimensional
    for y in parts:
        if y.ndim != 1:
            raise ValueError("{} must be a list of one-dimensional samples, or a 2D array with "
                             "one sample per row".format(name))
    lengths = [len(y) for y in parts]
    offsets = np.zeros(len(parts) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    buf = np.concatenate(parts) if parts else np.zeros(0)
    return np.ascontiguousarray(buf, dtype=float), offsets, lengths


def _batch(int p, X, Ys, A, Bs, method, int threads):
    cdef Method me = _method(method)
    cdef reference ref
    if isinstance(X, reference):
        if A is not None:
            raise ValueError("A must be None when X is a reference (its weights are fixed)")
        ref = X
    else:
        ref = reference(X, A, threads)
    Y, offsets, lengths = _chunks(Ys, 'Ys')
    cdef int64_t k = len(lengths)
    out = np.empty(k, dtype=float)
    if k == 0:
        return out
    B = None
    if Bs is not None:
        B, _, blen = _chunks(Bs, 'Bs')
        if blen != lengths:
            raise ValueError("Bs must have the same number and lengths of samples as Ys")
    cdef const double[::1] Ymv = Y if len(Y) > 0 else np.zeros(1)
    cdef const int64_t[::1] Omv = offsets
    cdef double[::1] Zmv = out
    cdef const double[::1] Bmv
    cdef const double* b = NULL
    cdef const c_reference* r = ref._ref.get()
    if B is not None:
        Bmv = B if len(B) > 0 else np.zeros(1)
        b = &Bmv[0]
    with nogil:
        if p == 1:
            c_w1_batch(r[0], &Ymv[0], b, &Omv[0], k, &Zmv[0], me, threads)
        else:
            c_w2_batch(r[0], &Ymv[0], b, &Omv[0], k, &Zmv[0], me, threads)
    return out


def w1_batch(X, Ys, A=None, Bs=None, method='ecdf', threads=1):
    """W1 between one reference and several samples: returns a numpy array.

    X: a sample (with optional weights A), or an ot1d.reference (then A must be None).
    Ys: a list of 1D samples, or a 2D array with one sample per row.
    Bs: optional weights, with the same layout as Ys.
    method: 'ecdf' (default) or 'simplex'. threads: the samples are split over
    this many threads (and the reference, if built here, is sorted with them).
    """
    return _batch(1, X, Ys, A, Bs, method, threads)


def w2_batch(X, Ys, A=None, Bs=None, method='ecdf', threads=1):
    """W2 between one reference and several samples (see w1_batch)."""
    return _batch(2, X, Ys, A, Bs, method, threads)


#--------------------------------------------------------------
# v0.4 function (kept for compatibility)
#--------------------------------------------------------------
def OT1D(X, Y, A=None, B=None, p=1, sorting=True, threads=1, plan=False):
    """v0.4 interface: W_p (p = 1 or 2) between X and Y, method 'simplex'.

    With plan=True returns (z, T, W), see w1_plan. Prefer w1, w2, w1_plan, w2_plan.
    """
    # Accept lists, tuples and arrays of any numeric type (no copy if already
    # a C-contiguous array of float64)
    X = np.ascontiguousarray(X, dtype=float)
    Y = np.ascontiguousarray(Y, dtype=float)
    if A is not None:
        A = np.ascontiguousarray(A, dtype=float)
    if B is not None:
        B = np.ascontiguousarray(B, dtype=float)

    if p not in (1, 2):
        raise ValueError("p must be 1 or 2, got {}".format(p))
    if (A is None) != (B is None):
        raise ValueError("weights A and B must be given both or neither")
    if X.ndim != 1 or Y.ndim != 1:
        raise ValueError("X and Y must be one-dimensional")
    if len(X) == 0 or len(Y) == 0:
        raise ValueError("X and Y must be non-empty")
    if A is not None:
        if A.ndim != 1 or len(A) != len(X):
            raise ValueError("A must be one-dimensional with the same length as X")
        if B.ndim != 1 or len(B) != len(Y):
            raise ValueError("B must be one-dimensional with the same length as Y")
        if np.any(A < 0) or np.any(B < 0):
            raise ValueError("weights must be nonnegative")
        sa, sb = A.sum(), B.sum()
        if not np.isclose(sa, sb, rtol=1e-9, atol=0.0):
            warnings.warn("A and B have different total mass ({} != {}): only the smaller mass "
                          "is transported, and the returned value and plan are partial".format(sa, sb),
                          RuntimeWarning, stacklevel=1)

    if plan:
        # Compute and return an optimal transportation plan
        m = len(X)
        n = len(Y)

        T = np.ascontiguousarray(np.zeros(2*(n+m-1), dtype=np.int32))
        W = np.ascontiguousarray(np.zeros(n+m-1, dtype=float))

        # Wassertein of order 1
        if p == 1:
            if A is None and B is None:
                z, ns = OT_1Da_plan(X, Y, T, W, sorting, threads)
            else:
                z, ns = OT_1Dc_plan(X, Y, A, B, T, W, sorting, threads)

        # Wassertein of order 2
        if p == 2:
            if A is None and B is None:
                z, ns = OT_1Db_plan(X, Y, T, W, sorting, threads)
            else:
                z, ns = OT_1Dd_plan(X, Y, A, B, T, W, sorting, threads)

        # Resize to contain only non-null maps
        T.resize((ns, 2))
        W.resize(ns)
        return z, T, W
    else:
        # Compute only the optimal value

        # Wassertein of order 1
        if p == 1:
            if A is None and B is None:
                return OT_1Da(X, Y, sorting, threads)
            else:
                return OT_1Dc(X, Y, A, B, sorting, threads)

        # Wassertein of order 2
        if p == 2:
            if A is None and B is None:
                return OT_1Db(X, Y, sorting, threads)
            else:
                return OT_1Dd(X, Y, A, B, sorting, threads)


_SIMD_LEVELS = ('none', 'avx2', 'avx512')
_simd_level = OT1D_simd_supported()


def simd():
    """SIMD instruction set used for sorting: 'avx512', 'avx2' or 'none'
    (pdqsort/parasort). By default, the best one supported by the CPU."""
    return _SIMD_LEVELS[_simd_level]


def set_simd(level=None):
    """Select the SIMD sorting: 'avx512', 'avx2', 'none', or None for the best
    supported one (for instance, to compare timings). A level not supported is
    lowered to the best supported one. Returns the level in use.
    Not thread safe: do not call it while other threads are running OT1D."""
    global _simd_level
    if level is None:
        level = _SIMD_LEVELS[OT1D_simd_supported()]
    if level not in _SIMD_LEVELS:
        raise ValueError("level must be one of {} or None, got {!r}".format(_SIMD_LEVELS, level))
    _simd_level = OT1D_set_simd(_SIMD_LEVELS.index(level))
    return _SIMD_LEVELS[_simd_level]


def _check_inplace(V, name):
    if not (isinstance(V, np.ndarray) and V.dtype == np.float64 and V.ndim == 1
            and V.flags['C_CONTIGUOUS'] and V.flags['WRITEABLE']):
        raise TypeError("parasort sorts in place: {} must be a writeable, one-dimensional, "
                        "C-contiguous numpy array of float64 (for instance, "
                        "np.ascontiguousarray({}, dtype=float))".format(name, name))


# Support for fast sorting (in place)
def parasort(X, A=None, int threads=1):
    _check_inplace(X, 'X')
    if A is not None:
        _check_inplace(A, 'A')
        if len(A) != len(X):
            raise ValueError("X and A must have the same length")
    if len(X) == 0:
        return

    cdef double[::1] Xmv = X
    cdef double[::1] Amv
    cdef int n = len(X)
    if A is None:
        with nogil:
            _parasort(n, &Xmv[0], threads)
        return

    Amv = A
    with nogil:
        _parasort_mu(n, &Xmv[0], &Amv[0], threads)


#--------------------------------------------------------------
# FROM C++
#
# The GIL is released around every call into C++: the computation
# touches only the raw buffers of the numpy arrays (kept alive by the
# memoryviews), so other Python threads can run concurrently. The samples and
# weights are never modified (the C++ functions copy them), so read-only arrays
# are accepted: the casts only remove the const of the memoryviews.
#--------------------------------------------------------------
cdef double OT_1Da(const double[::1] Xmv, const double[::1] Ymv, bool sorting, int threads):
    cdef int m = Xmv.shape[0]
    cdef int n = Ymv.shape[0]
    cdef double z
    with nogil:
        if m == n:
            z = OT1Da0(n, <double*>&Xmv[0], <double*>&Ymv[0], sorting, threads)
        else:
            z = OT1Da(m, n, <double*>&Xmv[0], <double*>&Ymv[0], sorting, threads)
    return z


cdef double OT_1Db(const double[::1] Xmv, const double[::1] Ymv, bool sorting, int threads):
    cdef int m = Xmv.shape[0]
    cdef int n = Ymv.shape[0]
    cdef double z
    with nogil:
        if m == n:
            z = OT1Db0(n, <double*>&Xmv[0], <double*>&Ymv[0], sorting, threads)
        else:
            z = OT1Db(m, n, <double*>&Xmv[0], <double*>&Ymv[0], sorting, threads)
    return z


cdef double OT_1Dc(const double[::1] Xmv, const double[::1] Ymv, const double[::1] Amv, const double[::1] Bmv,
                   bool sorting, int threads):
    cdef int m = Xmv.shape[0]
    cdef int n = Ymv.shape[0]
    cdef double z
    with nogil:
        z = OT1Dc(m, n, <double*>&Xmv[0], <double*>&Ymv[0], <double*>&Amv[0], <double*>&Bmv[0], sorting, threads)
    return z


cdef double OT_1Dd(const double[::1] Xmv, const double[::1] Ymv, const double[::1] Amv, const double[::1] Bmv,
                   bool sorting, int threads):
    cdef int m = Xmv.shape[0]
    cdef int n = Ymv.shape[0]
    cdef double z
    with nogil:
        z = OT1Dd(m, n, <double*>&Xmv[0], <double*>&Ymv[0], <double*>&Amv[0], <double*>&Bmv[0], sorting, threads)
    return z


cdef (double, int) OT_1Da_plan(const double[::1] Xmv, const double[::1] Ymv, int[::1] Tmv, double[::1] Wmv,
                               bool sorting, int threads):
    cdef int m = Xmv.shape[0]
    cdef int n = Ymv.shape[0]
    cdef int Smap = 0
    cdef double z
    with nogil:
        if m == n:
            z = OT1Da0_withplan(n, <double*>&Xmv[0], <double*>&Ymv[0], &Tmv[0], &Wmv[0], &Smap, sorting, threads)
        else:
            z = OT1Da_withplan(m, n, <double*>&Xmv[0], <double*>&Ymv[0], &Tmv[0], &Wmv[0], &Smap, sorting, threads)
    return z, Smap


cdef (double, int) OT_1Db_plan(const double[::1] Xmv, const double[::1] Ymv, int[::1] Tmv, double[::1] Wmv,
                               bool sorting, int threads):
    cdef int m = Xmv.shape[0]
    cdef int n = Ymv.shape[0]
    cdef int Smap = 0
    cdef double z
    with nogil:
        if m == n:
            z = OT1Db0_withplan(n, <double*>&Xmv[0], <double*>&Ymv[0], &Tmv[0], &Wmv[0], &Smap, sorting, threads)
        else:
            z = OT1Db_withplan(m, n, <double*>&Xmv[0], <double*>&Ymv[0], &Tmv[0], &Wmv[0], &Smap, sorting, threads)
    return z, Smap


cdef (double, int) OT_1Dc_plan(const double[::1] Xmv, const double[::1] Ymv, const double[::1] Amv, const double[::1] Bmv,
                               int[::1] Tmv, double[::1] Wmv, bool sorting, int threads):
    cdef int m = Xmv.shape[0]
    cdef int n = Ymv.shape[0]
    cdef int Smap = 0
    cdef double z
    with nogil:
        z = OT1Dc_withplan(m, n, <double*>&Xmv[0], <double*>&Ymv[0], <double*>&Amv[0], <double*>&Bmv[0], &Tmv[0], &Wmv[0], &Smap,
                           sorting, threads)
    return z, Smap


cdef (double, int) OT_1Dd_plan(const double[::1] Xmv, const double[::1] Ymv, const double[::1] Amv, const double[::1] Bmv,
                               int[::1] Tmv, double[::1] Wmv, bool sorting, int threads):
    cdef int m = Xmv.shape[0]
    cdef int n = Ymv.shape[0]
    cdef int Smap = 0
    cdef double z
    with nogil:
        z = OT1Dd_withplan(m, n, <double*>&Xmv[0], <double*>&Ymv[0], <double*>&Amv[0], <double*>&Bmv[0], &Tmv[0], &Wmv[0], &Smap,
                           sorting, threads)
    return z, Smap
