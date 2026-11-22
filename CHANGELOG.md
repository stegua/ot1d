# Changelog

## 0.5.1 (in development)

### Fixed
* `method='simplex'` (and `OT1D`), samples without weights and of different sizes: the rounding
  error of the merge grew with the number of points, up to 1.6e-10 relative for 3*10^7 points with
  many ties. The terms are now added in short blocks, and the blocks with compensated summation:
  the relative error stays near 1e-15, at the same speed.
* `method='simplex'` (and `OT1D`) with weights: the merge treated residual masses below 1e-13 of
  the total mass as zero, which limited the relative precision to about 1e-11. The threshold is
  now 1e-15, and the error is near 1e-15. A plan can now contain entries with a mass between
  1e-15 and 1e-13 of the total, which were dropped before.
* `method='ecdf'`: W1 could be slightly negative (a few 1e-17 times the scale of the data) on
  samples very close to the reference, for instance drawn from its points. A sum that rounding
  leaves below 0 is now returned as 0, as it already was for W2.
* Clang with `-ffast-math` (the build of the Python module on macOS): NaN and inf in the input
  were not rejected, and a distance that overflows was not `inf`, because Clang assumes that no
  `double` argument or return value is NaN or inf and removed the tests. The tests now read the
  bits of the values from memory.
* MSVC with `/fp:fast`: a distance that overflows was not `inf`. Under MSVC the header is now
  always compiled with the precise floating point model.
* `OT1D(..., plan=True)`, `w1_plan` and `w2_plan` raised `ValueError: cannot resize an array` on
  Python 3.14 and later.

### Changed
* `sorting=False` (and `method='simplex'` on a reference) no longer copies the input: the merge
  reads the arrays of the caller, with or without weights. Before, every call copied both samples
  (240 MB more memory for a sample of 3*10^7 points). On sorted samples, one thread: two samples of
  10^7 points take 8.5 ms instead of 84 ms (89 ms instead of 222 ms with weights); 10^4 points
  against 10^7 take 10 ms instead of 48 ms (24 ms instead of 86 ms with weights on both, on a
  reference).
* `method='simplex'` with `sorting=False` and no weights: the samples are searched for NaN and inf
  only when the result is not finite (such a value always reaches the result), which saves one more
  pass over both samples: 9 ms instead of 22 ms for two samples of 10^7 points. The errors raised
  are the same.
* The parallel sort (`threads > 1`) is now part of `OT1D.hpp`: one block per thread, sorted in
  parallel and then merged in parallel. It replaces the library parasort, the only GPL v3 file of
  the package, which is removed. On 10^7 values with 8 threads and without SIMD it takes 70 ms
  instead of 112 ms. With SIMD sorting, the threads are used only where they help: values from
  5 million points and 4 threads (10^7 values: 44 ms with 8 threads, 66 ms with one), and items
  with a plan; pairs (value, weight) are sorted by one thread, which was already faster than the
  previous parallel sort (156 ms instead of 236 ms with 8 threads).

### Packaging
* Wheels for CPython 3.8 to 3.14 on Linux (x86-64, aarch64), macOS (Intel, Apple Silicon) and
  Windows (x86-64), built and tested by the GitHub workflow `wheels.yml` and uploaded to PyPI on a
  version tag: no compiler is needed to install them.
* The package declares its licence (EUPL-1.2) and requires only `numpy` (`cython` is needed only to
  build from source). The Windows build no longer requires a CPU with AVX2.

## 0.5.0 (2026-11-21)

### New
* A second exact algorithm, `method='ecdf'`: the quantile form of 1D optimal transport, with the
  integrals of the reference quantile function precomputed as prefix sums. Every function takes the
  parameter `method`: `'simplex'` (the north-west corner rule of the previous versions) or `'ecdf'`.
* `ot1d.reference(X, A=None, threads=1, sorting=True)`: a sample prepared once (sorted, centered,
  with compensated prefix sums), 24 bytes per point (33 with weights), picklable. With
  `sorting=False` the sample is already sorted, and is only copied.
* `ot1d.w1_reference`, `ot1d.w2_reference`: distance between a reference and a sample. With
  `method='ecdf'` a sample of size m costs O(m log m) to sort it plus O(m) (W2) or O(m log(n/m))
  (W1), independent of the size n of the reference: 0.1 ms for 10^4 points against 10^7 reference
  points, instead of 49 ms for the merge with the reference sorted once.
* `ot1d.w1_batch`, `ot1d.w2_batch`: one reference against several samples (a list or a 2D array),
  split over threads.
* `ot1d.w1`, `ot1d.w2`, `ot1d.w1_plan`, `ot1d.w2_plan`: the distance between two samples, with the
  new names; `w1` and `w2` accept weights on one side only.
* C++: the same functions in `namespace ot1d` of `OT1D.hpp`, with 64-bit sizes, and the enumeration
  `ot1d::Method`. Invalid input (empty sample, NaN or inf, negative weights, zero total weight,
  different total masses with `Method::Ecdf`) throws `std::invalid_argument`; `w1` and `w2`
  accept a null pointer for the weights of one side.
* Tests: `python/test_reference.py` (both methods against a long double merge),
  `python/reference_prototype.py` (NumPy specification of the ecdf method),
  `python/bench_reference.py`; `make sanitize` also runs ThreadSanitizer.

### Changed
* The Python module is now `ot1d` (lowercase): replace `from OT1D import OT1D, parasort` with
  `from ot1d import OT1D, parasort`. The function `OT1D` and the old C++ functions are unchanged.
* Weights that do not sum to 1: `method='simplex'` keeps the semantics of `OT1D` (cost of
  transporting the total mass, partial transport of unequal masses with a `RuntimeWarning`);
  `method='ecdf'` normalizes to probability measures and raises `ValueError` on unequal masses.

* A distance that overflows (for instance, the squares in W2 with values above about 1e154) is
  `inf` with both methods, as in IEEE 754 arithmetic.
* `OT1D`, `w1_plan` and `w2_plan` accept read-only arrays.

* On Linux, the arrays of a large reference use transparent huge pages (if the system allows
  them with `madvise`): preparing a reference of 10^7 points takes 0.11 s instead of 0.15 s
  (1.3 times `np.sort`), and the queries are up to 1.7 times faster.

### Known limits
* With `method='ecdf'`, W2 squared has an absolute error of order 1e-16 times the variance of the
  reference: a W2 below about 1e-7 times its standard deviation is returned as about 1e-8 times it,
  instead of 0. W1 has no such limit.
* `method='simplex'` supports at most 2^31 - 1 points per sample (as before).

## 0.4.1
* Release the GIL, default `threads=1`, SIMD sorting (x86-simd-sort, AVX2 and AVX-512).
