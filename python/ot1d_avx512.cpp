/*
 * AVX-512 sorting (x86-simd-sort), see ot1d_simdsort.inc. Compile with C++17 and
 *    -mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512cd
 */
#define OT1D_SIMD_NS avx512
#include "ot1d_simdsort.inc"
