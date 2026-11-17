/*
 * @fileoverview Copyright (c) 2026, Stefano Gualandi,
 *               via Ferrata, 5, I-27100, Pavia, Italy
 *
 * @author stefano.gualandi@gmail.com (Stefano Gualandi)
 *
 * SIMD sorting routines, based on x86-simd-sort (Intel, BSD 3-Clause License)
 *    https://github.com/intel/x86-simd-sort
 *
 * Body shared by ot1d_avx512.cpp and ot1d_avx2.cpp: each of them defines
 * OT1D_SIMD_NS (the namespace) and is compiled with its own ISA flags, so the
 * same x86-simd-sort call picks the AVX-512 or the AVX2 implementation.
 *
 * These are the ONLY translation units compiled with SIMD flags. Link them
 * AFTER the object files of the rest of the program, and ot1d_avx2.o before
 * ot1d_avx512.o: for any inline function or template instantiated in more than
 * one file, the linker keeps the first copy, which is then safe on every CPU
 * that reaches it.
 *
 * The functions are called by OT1D.hpp only if the CPU supports the ISA at
 * runtime (see ot1d_simd::cpu_level()).
 */

#include <cstddef>
#include <cstdint>

#include "x86-simd-sort/x86simdsort-static-incl.h"

namespace ot1d_simd {
namespace OT1D_SIMD_NS {

// NaN values (if any) are moved to the end.
void sort(double* x, size_t n) {
    x86simdsortStatic::qsort(x, n, true);
}

void sort_kv(double* key, double* val, size_t n) {
    x86simdsortStatic::keyvalue_qsort(key, val, n, true);
}

void sort_kv(double* key, uint64_t* val, size_t n) {
    x86simdsortStatic::keyvalue_qsort(key, val, n, true);
}

} // namespace OT1D_SIMD_NS
} // namespace ot1d_simd
