# distutils: language=c++

# -*- coding: utf-8 -*-
#
# @fileoverview Copyright (c) 2019-2021, Stefano Gualandi,
#                via Ferrata, 1, I-27100, Pavia, Italy
#
#  @author stefano.gualandi@gmail.com (Stefano Gualandi)
#

import warnings

from libcpp cimport bool

import numpy as np

from OT1D cimport OT1Da0 as OT1Da0
from OT1D cimport OT1Db0 as OT1Db0

from OT1D cimport OT1Da as OT1Da
from OT1D cimport OT1Db as OT1Db
from OT1D cimport OT1Dc as OT1Dc
from OT1D cimport OT1Dd as OT1Dd

from OT1D cimport OT1Da0_withplan as OT1Da0_withplan
from OT1D cimport OT1Db0_withplan as OT1Db0_withplan

from OT1D cimport OT1Da_withplan as OT1Da_withplan
from OT1D cimport OT1Db_withplan as OT1Db_withplan
from OT1D cimport OT1Dc_withplan as OT1Dc_withplan
from OT1D cimport OT1Dd_withplan as OT1Dd_withplan

from OT1D cimport OT1D_simd_supported, OT1D_set_simd
from OT1D cimport _parasort as _parasort
from OT1D cimport _parasort_mu as _parasort_mu

# One function for optimal transport
def OT1D(X, Y, A=None, B=None, p=1, sorting=True, threads=1, plan=False):
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
                          RuntimeWarning, stacklevel=2)

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
# memoryviews), so other Python threads can run concurrently.
#--------------------------------------------------------------
cdef double OT_1Da(double[::1] Xmv, double[::1] Ymv, bool sorting, int threads):
    cdef int m = Xmv.shape[0]
    cdef int n = Ymv.shape[0]
    cdef double z
    with nogil:
        if m == n:
            z = OT1Da0(n, &Xmv[0], &Ymv[0], sorting, threads)
        else:
            z = OT1Da(m, n, &Xmv[0], &Ymv[0], sorting, threads)
    return z


cdef double OT_1Db(double[::1] Xmv, double[::1] Ymv, bool sorting, int threads):
    cdef int m = Xmv.shape[0]
    cdef int n = Ymv.shape[0]
    cdef double z
    with nogil:
        if m == n:
            z = OT1Db0(n, &Xmv[0], &Ymv[0], sorting, threads)
        else:
            z = OT1Db(m, n, &Xmv[0], &Ymv[0], sorting, threads)
    return z


cdef double OT_1Dc(double[::1] Xmv, double[::1] Ymv, double[::1] Amv, double[::1] Bmv,
                   bool sorting, int threads):
    cdef int m = Xmv.shape[0]
    cdef int n = Ymv.shape[0]
    cdef double z
    with nogil:
        z = OT1Dc(m, n, &Xmv[0], &Ymv[0], &Amv[0], &Bmv[0], sorting, threads)
    return z


cdef double OT_1Dd(double[::1] Xmv, double[::1] Ymv, double[::1] Amv, double[::1] Bmv,
                   bool sorting, int threads):
    cdef int m = Xmv.shape[0]
    cdef int n = Ymv.shape[0]
    cdef double z
    with nogil:
        z = OT1Dd(m, n, &Xmv[0], &Ymv[0], &Amv[0], &Bmv[0], sorting, threads)
    return z


cdef (double, int) OT_1Da_plan(double[::1] Xmv, double[::1] Ymv, int[::1] Tmv, double[::1] Wmv,
                               bool sorting, int threads):
    cdef int m = Xmv.shape[0]
    cdef int n = Ymv.shape[0]
    cdef int Smap = 0
    cdef double z
    with nogil:
        if m == n:
            z = OT1Da0_withplan(n, &Xmv[0], &Ymv[0], &Tmv[0], &Wmv[0], &Smap, sorting, threads)
        else:
            z = OT1Da_withplan(m, n, &Xmv[0], &Ymv[0], &Tmv[0], &Wmv[0], &Smap, sorting, threads)
    return z, Smap


cdef (double, int) OT_1Db_plan(double[::1] Xmv, double[::1] Ymv, int[::1] Tmv, double[::1] Wmv,
                               bool sorting, int threads):
    cdef int m = Xmv.shape[0]
    cdef int n = Ymv.shape[0]
    cdef int Smap = 0
    cdef double z
    with nogil:
        if m == n:
            z = OT1Db0_withplan(n, &Xmv[0], &Ymv[0], &Tmv[0], &Wmv[0], &Smap, sorting, threads)
        else:
            z = OT1Db_withplan(m, n, &Xmv[0], &Ymv[0], &Tmv[0], &Wmv[0], &Smap, sorting, threads)
    return z, Smap


cdef (double, int) OT_1Dc_plan(double[::1] Xmv, double[::1] Ymv, double[::1] Amv, double[::1] Bmv,
                               int[::1] Tmv, double[::1] Wmv, bool sorting, int threads):
    cdef int m = Xmv.shape[0]
    cdef int n = Ymv.shape[0]
    cdef int Smap = 0
    cdef double z
    with nogil:
        z = OT1Dc_withplan(m, n, &Xmv[0], &Ymv[0], &Amv[0], &Bmv[0], &Tmv[0], &Wmv[0], &Smap,
                           sorting, threads)
    return z, Smap


cdef (double, int) OT_1Dd_plan(double[::1] Xmv, double[::1] Ymv, double[::1] Amv, double[::1] Bmv,
                               int[::1] Tmv, double[::1] Wmv, bool sorting, int threads):
    cdef int m = Xmv.shape[0]
    cdef int n = Ymv.shape[0]
    cdef int Smap = 0
    cdef double z
    with nogil:
        z = OT1Dd_withplan(m, n, &Xmv[0], &Ymv[0], &Amv[0], &Bmv[0], &Tmv[0], &Wmv[0], &Smap,
                           sorting, threads)
    return z, Smap
