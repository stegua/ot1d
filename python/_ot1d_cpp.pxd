# -*- coding: utf-8 -*-
#
# @fileoverview Copyright (c) 2019-2026, Stefano Gualandi,
#                via Ferrata, 1, I-27100, Pavia, Italy
#
#  @author stefano.gualandi@gmail.com (Stefano Gualandi)
#
# Declarations of the C++ header OT1D.hpp, used by ot1d.pyx. The C++ names that
# clash with the Python names of the module (w1, reference, ...) get a "c_"
# prefix here, with their C++ name in quotes.

# C++ bool: without this import, Cython treats `bool` as the Python bool
# object and passes a non-null PyObject*, so `sorting` was always true.
from libcpp cimport bool
from libc.stdint cimport int64_t

# All functions are pure C++ (no Python API), so they can run without the GIL.
# `except +` turns C++ exceptions into Python exceptions (std::invalid_argument
# into ValueError, std::length_error into ValueError, std::bad_alloc into MemoryError).
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

# namespace ot1d (v0.5.0)
cdef extern from "OT1D.hpp" namespace "ot1d" nogil:
	cdef enum class Method:
		Simplex
		Ecdf

	cdef cppclass c_reference "ot1d::reference":
		c_reference(const double*, int64_t, int) except +
		c_reference(const double*, const double*, int64_t, int) except +
		int64_t size()
		double mean()
		bool uniform()
		size_t nbytes()
		const double* r()
		const double* U()

cdef extern from "OT1D.hpp" nogil:
	# Weighted overloads; a, b may be NULL (uniform weights)
	cdef double c_w1 "ot1d::w1"(const double*, const double*, int64_t,
	                            const double*, const double*, int64_t, Method, bool, int) except +
	cdef double c_w2 "ot1d::w2"(const double*, const double*, int64_t,
	                            const double*, const double*, int64_t, Method, bool, int) except +

	cdef double c_w1_reference "ot1d::w1_reference"(const c_reference&, const double*, const double*, int64_t,
	                                                Method, bool) except +
	cdef double c_w2_reference "ot1d::w2_reference"(const c_reference&, const double*, const double*, int64_t,
	                                                Method, bool) except +

	cdef void c_w1_batch "ot1d::w1_batch"(const c_reference&, const double*, const double*, const int64_t*,
	                                      int64_t, double*, Method, int) except +
	cdef void c_w2_batch "ot1d::w2_batch"(const c_reference&, const double*, const double*, const int64_t*,
	                                      int64_t, double*, Method, int) except +
