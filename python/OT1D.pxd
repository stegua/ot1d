# -*- coding: utf-8 -*-
#
# @fileoverview Copyright (c) 2019-2021, Stefano Gualandi,
#                via Ferrata, 1, I-27100, Pavia, Italy
#
#  @author stefano.gualandi@gmail.com (Stefano Gualandi)
#

# C++ bool: without this import, Cython treats `bool` as the Python bool
# object and passes a non-null PyObject*, so `sorting` was always true.
from libcpp cimport bool

# All functions are pure C++ (no Python API), so they can run without the GIL.
# `except +` turns C++ exceptions (e.g. std::bad_alloc) into Python exceptions.
cdef extern from "OT1D.hpp" nogil:
	cdef double OT1Da0(int, double*, double*, bool, int) except +
	cdef double OT1Db0(int, double*, double*, bool, int) except +

	cdef double OT1Da(int, int, double*, double*, bool, int) except +
	cdef double OT1Db(int, int, double*, double*, bool, int) except +
	cdef double OT1Dc(int, int, double*, double*, double*, double*, bool, int) except +
	cdef double OT1Dd(int, int, double*, double*, double*, double*, bool, int) except +

	cdef double OT1Da0_withplan(int, double*, double*, int*, double*, int*, bool, int) except +
	cdef double OT1Db0_withplan(int, double*, double*, int*, double*, int*, bool, int) except +

	cdef double OT1Da_withplan(int, int, double*, double*, int*, double*, int*, bool, int) except +
	cdef double OT1Db_withplan(int, int, double*, double*, int*, double*, int*, bool, int) except +
	cdef double OT1Dc_withplan(int, int, double*, double*, double*, double*, int*, double*, int*, bool, int) except +
	cdef double OT1Dd_withplan(int, int, double*, double*, double*, double*, int*, double*, int*, bool, int) except +
	
	cdef int OT1D_simd_supported()
	cdef int OT1D_set_simd(int)

	cdef void _parasort(int, double*, int) except +
	cdef void _parasort_mu(int, double*, double*, int) except +