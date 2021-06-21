# distutils: language=c++

# -*- coding: utf-8 -*-
#
# @fileoverview Copyright (c) 2019-2021, Stefano Gualandi,
#                via Ferrata, 1, I-27100, Pavia, Italy
#
#  @author stefano.gualandi@gmail.com (Stefano Gualandi)
#

import numpy as np

from OT1D cimport OT1Da0 as OT1Da0
from OT1D cimport OT1Db0 as OT1Db0
from OT1D cimport OT1Da as OT1Da
from OT1D cimport OT1Db as OT1Db
from OT1D cimport OT1Dc as OT1Dc
from OT1D cimport OT1Dd as OT1Dd
from OT1D cimport _parasort as _parasort
from OT1D cimport _parasort_mu as _parasort_mu

# One function for optimal transport
def OT1D(X, Y, A=None, B=None, p=1, sorting=True, threads=8):
    if type(X) == list or not X.flags['C_CONTIGUOUS'] or X.dtype != float:
        X = np.ascontiguousarray(X, dtype=float) 

    if type(Y) == list or not Y.flags['C_CONTIGUOUS'] or Y.dtype != float:
        Y = np.ascontiguousarray(Y, dtype=float) 

    if A is not None and (type(A) == list or not A.flags['C_CONTIGUOUS'] or A.dtype != float):
        A = np.ascontiguousarray(A, dtype=float) 

    if B is not None and (type(B) == list or not B.flags['C_CONTIGUOUS'] or B.dtype != float):
        B = np.ascontiguousarray(B, dtype=float) 

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


# Support for fast sorting
def parasort(X, A=None, threads=8):
    if not X.flags['C_CONTIGUOUS'] or type(X) == list or X.dtype != float:
        X = np.ascontiguousarray(X, dtype=float) 
    cdef double[::1] Xmv = X
    if A is None:
        return _parasort(len(X), &Xmv[0], threads)

    if not A.flags['C_CONTIGUOUS'] or type(A) == list or A.dtype != float:
        A = np.ascontiguousarray(A, dtype=float) 

    cdef double[::1] Amv = A    
    return _parasort_mu(len(X), &Xmv[0], &Amv[0], threads)


#--------------------------------------------------------------
# FROM C++
#--------------------------------------------------------------
cdef double OT_1Da(X, Y, sorting=True, threads=8):
    m = len(X)
    n = len(Y)
    cdef double[::1] Xmv = X
    cdef double[::1] Ymv = Y
    if m == n:
        return OT1Da0(n, &Xmv[0], &Ymv[0], sorting, threads) 
    return OT1Da(m, n, &Xmv[0], &Ymv[0], sorting, threads)


cdef double OT_1Db(X, Y, sorting=True, threads=8):
    m = len(X)
    n = len(Y)
    cdef double[::1] Xmv = X
    cdef double[::1] Ymv = Y
    if m == n:
        return OT1Db0(n, &Xmv[0], &Ymv[0], sorting, threads)     
    return OT1Db(m, n, &Xmv[0], &Ymv[0], sorting, threads)


cdef double OT_1Dc(X, Y, A, B, sorting=True, threads=8):
    m = len(X)
    n = len(Y)
    cdef double[::1] Xmv = X
    cdef double[::1] Ymv = Y
    cdef double[::1] Amv = A
    cdef double[::1] Bmv = B
    
    return OT1Dc(m, n, &Xmv[0], &Ymv[0], &Amv[0], &Bmv[0], sorting, threads)


cdef double OT_1Dd(X, Y, A, B, sorting=True, threads=8):
    m = len(X)
    n = len(Y)
    cdef double[::1] Xmv = X        
    cdef double[::1] Ymv = Y 
    cdef double[::1] Amv = A
    cdef double[::1] Bmv = B
    
    return OT1Dd(m, n, &Xmv[0], &Ymv[0], &Amv[0], &Bmv[0], sorting, threads)

