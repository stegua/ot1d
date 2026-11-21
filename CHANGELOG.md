# Changelog

## 0.5.0 (unreleased)

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

### Known limits
* With `method='ecdf'`, W2 squared has an absolute error of order 1e-16 times the variance of the
  reference: a W2 below about 1e-7 times its standard deviation is returned as about 1e-8 times it,
  instead of 0. W1 has no such limit.
* `method='simplex'` supports at most 2^31 - 1 points per sample (as before).

## 0.4.1
* Release the GIL, default `threads=1`, SIMD sorting (x86-simd-sort, AVX2 and AVX-512).
