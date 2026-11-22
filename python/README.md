# OT1D: Discrete Optimal Transport in 1D by Linear Programming

[![build-update Actions Status](https://github.com/stegua/ot1d/workflows/build-update/badge.svg)](https://github.com/stegua/ot1d/actions)  [![Downloads](https://pepy.tech/badge/ot1d)](https://pepy.tech/project/ot1d) [![Downloads](https://pepy.tech/badge/ot1d/month)](https://pepy.tech/project/ot1d) [![Downloads](https://pepy.tech/badge/ot1d/week)](https://pepy.tech/project/ot1d)

The OT1D library offers a simple but efficient implementation of an algorithm to compute the Kantorovich-Wasserstein distance between two empirical measures defined in dimension 1, that is, measures whose support points lie in **R**.
We designed the algorithm by directly exploiting the [complementary slackness](https://en.wikipedia.org/wiki/Linear_programming#Complementary_slackness) conditions of Linear Programming.
Since version 0.5.0, a second exact algorithm, based on the empirical cumulative distribution functions, compares one fixed *reference* sample against many other samples in a time that does not depend on the size of the reference (see [Many samples against one reference](#many-samples-against-one-reference)).
The implementation focuses on efficiency rather than genericity, and we try to be as efficient as possible in several notable cases.
We implemented the core algorithm in standard C++11, and we provide a Python 3 wrapper, which can be installed with:

```bash
pip3 install ot1d
```

The **OT1D** library provides an implementation of Optimal Transport in 1D that is faster than:

1. [SciPy](https://scipy.org/): it is at least 5x faster than [scipy.stats.wasserstein_distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html), and up to 30x faster on sorted input.
2. [POT](https://pythonot.github.io/): it is at least 3x faster than [ot.lp.wasserstein_1d](https://pythonot.github.io/gen_modules/ot.lp.html#ot.lp.wasserstein_1d), and up to 45x faster on sorted input.

The actual speedup depends on your computer (e.g., the number of cores), your OS, and your compiler. To run a performance test on your computer, see below, or run the Python script [OT1D_test.py](https://github.com/stegua/ot1d/blob/main/python/OT1D_test.py). For some strange reason, the speedups on Mac laptops are larger than on other architectures.

**REMARK:** If you find instances where **OT1D** is slower, please let us know.

### DotLIB
This tiny library is part of [dotlib](https://github.com/stegua/dotlib), a larger project to develop Optimal Transport algorithms based on efficient Linear Programming implementations.

### Related projects
* [not1d](https://github.com/pietroppeter/not1d) by Pietro Peterlongo: a partial port of OT1D to [Nim](https://nim-lang.org/), which can also be used from Python. It computes the distance (without the transportation plan) on a single thread, using a radix sort.

## Basic Usage: Colab Notebook

The simplest way to test this library is to run the following notebook on Colab:

| Date | Notebook | Link |
|:-|:-|:-|
|**[2026/11/21]**|*Testing and evaluating OT1D*|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stegua/ot1d/blob/main/notebooks/Testing_OT1D.ipynb)|


## Usage
Since version 0.5.0 the module is called `ot1d` (lowercase), and offers the following functions:

```python
import ot1d

z = ot1d.w1(x, y, A=None, B=None, method='simplex', sorting=True, threads=1)   # W1 between two samples
z = ot1d.w2(x, y, A=None, B=None, method='simplex', sorting=True, threads=1)   # W2
z, T, W = ot1d.w1_plan(x, y, A=None, B=None, sorting=True, threads=1)          # W1 and an optimal plan
z, T, W = ot1d.w2_plan(x, y, A=None, B=None, sorting=True, threads=1)          # W2 and an optimal plan

ref = ot1d.reference(x, A=None, threads=1)                                     # a sample prepared once
z = ot1d.w1_reference(ref, y, B=None, method='ecdf', sorting=True)             # W1 between ref and y
z = ot1d.w2_reference(ref, y, B=None, method='ecdf', sorting=True)             # W2
zs = ot1d.w1_batch(x_or_ref, ys, A=None, Bs=None, method='ecdf', threads=1)    # W1 between x and each y in ys
zs = ot1d.w2_batch(x_or_ref, ys, A=None, Bs=None, method='ecdf', threads=1)    # W2
```

The parameter `method` selects the algorithm:

* `'simplex'` (default for `w1` and `w2`): the north-west corner rule, derived from the complementary slackness conditions of the transportation linear program. After sorting both samples, it walks once over the n + m support points.
* `'ecdf'` (default for `reference`, `w1_reference`, `w2_reference`, `w1_batch`, `w2_batch`): the quantile form of 1D optimal transport, which integrates |F<sup>-1</sup>(u) - G<sup>-1</sup>(u)|<sup>p</sup> using prefix sums precomputed on the reference. A sample of size m costs O(m log m) to sort it, plus O(m) for W2 and O(m log(n/m)) at most for W1, whatever the size n of the reference.

Both methods are exact and return the same value for probability measures (no weights, or weights summing to 1). With weights of total mass M different from 1, `'simplex'` returns, as `OT1D` always did, the cost of transporting the mass M (that is, M W1 or sqrt(M) W2), and if the two total masses differ it transports only the smaller one, with a `RuntimeWarning`; `'ecdf'` normalizes the weights to probability measures, and raises a `ValueError` if the two total masses differ.

**Migrating from 0.4:** replace `from OT1D import OT1D, parasort` with `from ot1d import OT1D, parasort` (or `import ot1d`); the function `OT1D` is unchanged, and `OT1D(x, y, p=1)` equals `ot1d.w1(x, y)`.

### The function OT1D (v0.4 interface)
The main function of versions up to 0.4 is still available:

```python
z = ot1d.OT1D(x, y, A=None, B=None, p=1, sorting=True, threads=1, plan=False)
```

The parameters of the function are:

* `x`: the support points of the first measure.
* `y`: the support points of the second measure.
* `A`: the (nonnegative) weights of the first measure. If equal to `None`, all the samples have the same mass.
* `B`: the (nonnegative) weights of the second measure. If equal to `None`, all the samples have the same mass. `A` and `B` must be given together. If their total masses differ, a `RuntimeWarning` is issued and only the smaller total mass is transported (partial value and plan).
* `p`: the order of the Wasserstein distance (`p=1` or `p=2`).
* `sorting`: if equal to `True`, the function sorts the support points given as input; if equal to `False`, the input must already be sorted.
* `threads`: the number of threads used by the parallel sorting algorithm (default `1`). Threads only speed up the sorting phase, so they matter only with `sorting=True`, and only on very large samples. **In most cases you do not need them: keep `threads=1`.** On x86-64 CPUs (Linux, macOS and, since version 0.5.1, Windows), where the SIMD sorting is used, the extra threads are ignored for samples of fewer than 5 million points (and with fewer than 4 threads), and for samples with weights of any size; at 10 million points, 8 threads sort in 44 ms instead of 66 ms. Only the functions that return a plan use them earlier, from 200,000 points. Without the SIMD sorting (ARM, including Apple Silicon), the threads are used from 200,000 points. In `w1_batch` and `w2_batch` the parameter has a different role: the samples are split over the threads, which helps as soon as there are several samples.
* `plan`: if equal to `True`, the function also returns an optimal transportation plan, as a triple `z, T, W`: the row `T[k] = (i, j)` means that the mass `W[k]` moves from `x[i]` to `y[j]` (see the example [`interpolate.py`](https://github.com/stegua/ot1d/blob/main/python/interpolate.py)).

The first four parameters can be given as numpy arrays (*preferred*) or Python lists.

All the functions release the GIL while running the C++ code, so several calls can run in parallel from different Python threads (for instance, with a `ThreadPoolExecutor`); in this case, keep `threads=1`.

### Many samples against one reference
Drift monitoring compares many samples (for instance, the chunks of a data stream) with one fixed reference sample, often much larger. The reference is prepared once, by sorting it and computing the prefix sums of its values and of their squares; every query then costs about as much as sorting the query itself:

```python
import numpy as np
import ot1d

rng = np.random.default_rng(0)
x = rng.lognormal(0, 1, 10_000_000)                          # reference: 10 million points
chunks = [rng.lognormal(0.1, 1, 10_000) for _ in range(100)]

ref = ot1d.reference(x)                                       # sort and prefix sums, once
z = ot1d.w1_reference(ref, chunks[0])                         # one sample
zs = ot1d.w1_batch(ref, chunks, threads=4)                    # all the samples, on 4 threads
```

* `ot1d.reference(x, A=None, threads=1, sorting=True)` copies and sorts `x` (with optional weights `A`) and uses 24 bytes per point (33 with weights). With `sorting=False`, `x` (and `A`) must be already sorted by `x`, and is only copied. It has `len(ref)`, and the read-only attributes `mean`, `nbytes`, `sorted_values`, `weights`. It can be pickled.
* `ot1d.w1_reference(ref, y, B=None, method='ecdf', sorting=True)` returns W1 between the reference and the sample `y` (with optional weights `B`); with `sorting=False`, `y` (and `B`) must be already sorted by `y`. `method='simplex'` merges the sorted reference with the sorted query instead, in O(n + m).
* `ot1d.w1_batch(x_or_ref, ys, A=None, Bs=None, method='ecdf', threads=1)` returns a numpy array with the distances between the reference (an `ot1d.reference`, or a sample `x` with weights `A`, prepared on the fly) and each sample of `ys`, a list of 1D arrays or a 2D array with one sample per row. `Bs` gives the weights of the samples, in the same layout. The samples are split over `threads` threads.
* `w2_reference` and `w2_batch` are the same for W2.

Time per sample of 10<sup>4</sup> points (lognormal data, 1 thread, Intel Xeon w9-3495X with AVX-512):

| reference size | `w1_reference` (ecdf) | `w2_reference` (ecdf) | `OT1D`, reference sorted once | [NannyML](https://github.com/NannyML/nannyml) binned W1 (10<sup>4</sup> bins, approximate) |
|---:|---:|---:|---:|---:|
| 10<sup>5</sup> | 0.056 ms | 0.055 ms | 0.13 ms | 0.16 ms |
| 10<sup>6</sup> | 0.067 ms | 0.051 ms | 1.1 ms | 0.16 ms |
| 10<sup>7</sup> | 0.076 ms | 0.051 ms | 10 ms | 0.13 ms |

Preparing the reference of 10<sup>7</sup> points takes 0.11 seconds (sorting alone takes 0.08 seconds), or 0.06 seconds if it is already sorted (`sorting=False`). Weights make the queries slower, because the quantile positions are no longer integers: against the reference of 10<sup>7</sup> points, a weighted sample takes 0.20 ms (W1) and 0.21 ms (W2); with a weighted reference, a sample takes 0.30 ms without weights and 0.50 ms with weights (preparing that reference takes 0.32 seconds).

The third column is the simplex merge of the two sorted samples, which reads all the points of the reference: since version 0.5.1 it no longer copies them (it took 48 ms at 10<sup>7</sup> points). When both samples are large and already sorted, this merge is the faster method: two sorted samples of 10<sup>7</sup> points take 9 ms with `ot1d.w1(x, y, sorting=False)`, and 84 ms with `method='ecdf'`. In this case (sorted samples without weights) the merge runs at the speed of the memory, so the samples are searched for NaN and inf only if the result is not finite.

On Linux, the arrays of a large reference are allocated on transparent huge pages (2 MB), when the system allows them on request (the setting `madvise` or `always` in `/sys/kernel/mm/transparent_hugepage/enabled`, the default of the main distributions). The timings above use them; with standard 4 KB pages, preparing the reference of 10<sup>7</sup> points takes 0.15 seconds, and a sample about 0.10 ms. The benchmark is [bench_reference.py](https://github.com/stegua/ot1d/blob/main/python/bench_reference.py).

The values agree with a long double north-west corner merge to about 1e-14 relative error (at least 1e-11 in all the tests): the quantile positions are exact integers, and the prefix sums are accumulated with compensated (Neumaier) summation. One known limit concerns W2 only: its square has an absolute error of order 1e-16 times the variance of the reference, so a W2 smaller than about 1e-7 times the standard deviation of the reference (for instance, a sample equal to the reference) is returned as about 1e-8 times that standard deviation, instead of 0. W1 has no such limit.

### Sorting at the speed of light
In addition, we expose the following in-place parallel sorting function:
```python
parasort(x, A=None, threads=1)
```

The parameters of the function are:

* `x`: the support points of a given measure.
* `A`: the weights of the given measure. If equal to `None`, only the support points are sorted.
* `threads`: the number of threads used by the parallel sorting algorithm.

The first two parameters must be numpy arrays of floats, since they are sorted in place.

## Details
Given two empirical distributions, the Kantorovich-Wasserstein distance is given by the optimal solution of a linear program, known as the transportation problem.
While this is a general linear program, when the costs are defined among points belonging to the real line,
the problem can be solved with an algorithm having worst-case time complexity of *O(n log n)*.
This can be shown by first writing the dual linear program, and then the complementary slackness conditions.

The key step of the algorithm is sorting the two arrays of support points *x* and *y*.
We sort the arrays with the very fast [pdqsort](https://github.com/orlp/pdqsort) (Zlib license). With `threads > 1` and at least 200,000 points, the array is cut into one block per thread, the blocks are sorted in parallel, and then merged in parallel. Since version 0.5.1 this parallel sort is part of `OT1D.hpp`, and replaces the library parasort (GPL v3) used by the previous versions; the function `ot1d.parasort` keeps its name.

On x86-64 CPUs, the Python wrapper sorts with the vectorized quicksort of [x86-simd-sort](https://github.com/intel/x86-simd-sort) (BSD 3-Clause License), the same library used by `np.sort`. The best instruction set supported by the CPU is selected at runtime: AVX-512 (for instance, Intel Skylake-X and later, AMD Zen 4 and later), AVX2 (almost every x86-64 CPU since 2013), or none, falling back to pdqsort. With AVX-512 a single thread sorts 3 to 8 times faster than pdqsort, and with AVX2 about 2.5 times faster, so with SIMD sorting `threads` is used only for samples without weights of at least 5 million points (from 4 threads), and for the functions that return a plan. You can check which one is used with `ot1d.simd()`, and select another one with `ot1d.set_simd('avx2')` (or `'avx512'`, `'none'`), for instance to compare timings. To build without them, set the environment variable `OT1D_NO_SIMD=1` before installing.

In C++, the header `OT1D.hpp` offers the same functions in the namespace `ot1d`, with 64-bit sizes: `ot1d::w1`, `ot1d::w2`, the class `ot1d::reference`, `ot1d::w1_reference`, `ot1d::w2_reference`, `ot1d::w1_batch` and `ot1d::w2_batch` (samples given one after the other in a single buffer, with an array of offsets), and the enumeration `ot1d::Method { Simplex, Ecdf }`; the functions `OT1Da`, ..., `OT1Dd_withplan` of version 0.4 are unchanged. See the comments in the header.
The header uses pdqsort by default. To use the SIMD sorting, compile `include/ot1d_avx2.cpp` and/or `include/ot1d_avx512.cpp` (C++17) with the corresponding flags, compile your code with `-DOT1D_USE_AVX2` and/or `-DOT1D_USE_AVX512`, and link these object files last, AVX2 first (see the target `test-simd` in the `Makefile`).

### Prerequisites for compilation

To compile the source code and the Python wrapper, you only need:

* A C++ compiler compliant with the `C++11` standard (`C++17` for the optional SIMD sorting).
* [cython](https://cython.org/)
* [numpy](https://numpy.org/)

You might need to install the Python development headers, which on Linux can be installed with:

```bash
apt install python3-dev  # Ubuntu
```

### Installation

To install `OT1D`, you can run the following command:

```bash
pip3 install ot1d
```

Since version 0.5.1, PyPI offers compiled wheels for CPython 3.8 to 3.14 on Linux (x86-64 and aarch64), macOS (Intel and Apple Silicon) and Windows (x86-64), so no compiler is needed there. On the other platforms, `pip` compiles the source distribution, with the prerequisites listed above.

### Testing

To test the library, you can run the following command:

```bash
python3 basic_test.py
```

A basic usage example is the following:

```python
import numpy as np
import ot1d

np.random.seed(13)

N = 1000000

# Uniform samples
x = np.random.uniform(1, 2, N)
y = np.random.uniform(0, 1, N)

z = ot1d.w2(x, y)

print('Wasserstein distance of order 2, W2(x,y) =', z)
```
and the output should be similar to:
```
Wasserstein distance of order 2, W2(x,y) = 1.0002549459050794
```

### Testing for correctness
The repository includes three test suites, which compare every function against an independent reference implementation, and check that the transportation plans are feasible and optimal:

```bash
make test                         # C++ tests (also: make sanitize, with AddressSanitizer, UBSan and ThreadSanitizer)
cd python && python3 test_wrapper.py  # Python wrapper tests (requires scipy)
cd python && OT1D_REFERENCE=native python3 -m pytest test_reference.py  # both methods (requires pytest)
```

Without `OT1D_REFERENCE=native`, `test_reference.py` runs on [reference_prototype.py](https://github.com/stegua/ot1d/blob/main/python/reference_prototype.py), the NumPy implementation of the `'ecdf'` method that served as its specification.

### Testing for performance
The following results can be reproduced by running the following command (you need to have installed [scipy](https://scipy.org/) and [pot](https://pythonot.github.io/)):
```bash
python3 OT1D_test.py
```
whose output should be similar to the following (but it depends on your platform). The *speedup* is the ratio between the runtime of each library and the runtime of OT1D.
TEST 5 checks that the Wasserstein distance of order 2 computed by OT1D matches the one computed by POT (SciPy only computes W1), on instances from 1 to 5 million points; note that POT returns W2 squared, so the script compares OT1D with its square root.
```
--------------- TEST 3: Unsorted input (average runtime) --------------------
For OT1D using 8 threads

running test . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
Testing W1, samples of deltas, n=m
Scipy: average time = 0.1335 speedup = 17.9
POT  : average time = 0.0897 speedup = 12.0
OT1D : average time = 0.0074 speedup = 1.0

Testing W2, samples of deltas, n=m
POT  : average time = 0.0886 speedup = 11.9
OT1D : average time = 0.0075 speedup = 1.0

Testing W1, samples with weights
Scipy: average time = 0.147 speedup = 5.8
POT  : average time = 0.0979 speedup = 3.8
OT1D : average time = 0.0255 speedup = 1.0

Testing W2, samples with weights
POT  : average time = 0.0938 speedup = 3.7
OT1D : average time = 0.0252 speedup = 1.0

--------------- TEST 4: Sorted input (average runtime) --------------------
For OT1D using 8 threads

running test . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
Parallel sorting: time = 0.011 sec

Testing W1, samples of deltas, n=m
Scipy: average time = 0.0603 speedup = 154.3
POT  : average time = 0.0793 speedup = 202.9
OT1D : average time = 0.0004 speedup = 1.0

Testing W2, samples of deltas, n=m
POT  : average time = 0.0799 speedup = 198.1
OT1D : average time = 0.0004 speedup = 1.0

Testing W1, samples with weights
Scipy: average time = 0.069 speedup = 12.4
POT  : average time = 0.0867 speedup = 15.6
OT1D : average time = 0.0056 speedup = 1.0

Testing W2, samples with weights
POT  : average time = 0.0879 speedup = 16.7
OT1D : average time = 0.0053 speedup = 1.0

--------------- TEST 5: W2 against POT (correctness) --------------------
Comparing OT1D with sqrt(POT), since POT returns W2^2

m =       1, n =       1, weights = False | POT: z = 4.98342038 time = 0.0001 | OT1D: z = 4.98342038 time = 0.0000
m =       1, n =       1, weights = True  | POT: z = 4.98342038 time = 0.0001 | OT1D: z = 4.98342038 time = 0.0000
m =       2, n =       3, weights = False | POT: z = 3.46826482 time = 0.0001 | OT1D: z = 3.46826482 time = 0.0000
m =       2, n =       3, weights = True  | POT: z = 4.10324470 time = 0.0001 | OT1D: z = 4.10324470 time = 0.0000
m =      10, n =      10, weights = False | POT: z = 2.86894031 time = 0.0001 | OT1D: z = 2.86894031 time = 0.0000
m =      10, n =      10, weights = True  | POT: z = 2.93894066 time = 0.0001 | OT1D: z = 2.93894066 time = 0.0000
m =      17, n =       5, weights = False | POT: z = 3.18112337 time = 0.0001 | OT1D: z = 3.18112337 time = 0.0000
m =      17, n =       5, weights = True  | POT: z = 3.40960305 time = 0.0001 | OT1D: z = 3.40960305 time = 0.0000
m =    1000, n =    1000, weights = False | POT: z = 3.18595988 time = 0.0002 | OT1D: z = 3.18595988 time = 0.0000
m =    1000, n =    1000, weights = True  | POT: z = 3.23862020 time = 0.0002 | OT1D: z = 3.23862020 time = 0.0000
m =    1000, n =     777, weights = False | POT: z = 3.12133828 time = 0.0001 | OT1D: z = 3.12133828 time = 0.0000
m =    1000, n =     777, weights = True  | POT: z = 3.08777081 time = 0.0001 | OT1D: z = 3.08777081 time = 0.0000
m =  100000, n =  100000, weights = False | POT: z = 3.16226321 time = 0.0127 | OT1D: z = 3.16226321 time = 0.0009
m =  100000, n =  100000, weights = True  | POT: z = 3.15995238 time = 0.0145 | OT1D: z = 3.15995238 time = 0.0039
m =  100000, n =   65537, weights = False | POT: z = 3.15278684 time = 0.0092 | OT1D: z = 3.15278684 time = 0.0008
m =  100000, n =   65537, weights = True  | POT: z = 3.15374537 time = 0.0105 | OT1D: z = 3.15374537 time = 0.0031
m = 5000000, n = 5000000, weights = False | POT: z = 3.16297776 time = 1.2020 | OT1D: z = 3.16297776 time = 0.0912
m = 5000000, n = 5000000, weights = True  | POT: z = 3.16249183 time = 1.2938 | OT1D: z = 3.16249183 time = 0.3450

All W2 values match POT: max relative error = 1.42e-11
```

### Timings on different machines
The table below gives the timings of the same short benchmark ([bench_platform.py](https://github.com/stegua/ot1d/blob/main/tests/bench_platform.py)) on the machines where the package is tested, as a reference for the reader. Every value is the median of several runs, with one thread unless stated otherwise; the samples are lognormal, without weights unless stated otherwise. On Apple Silicon there is no vectorized sort (x86-simd-sort needs AVX2 or AVX-512), so sorting uses pdqsort, and there the threads help from 200,000 points.

<!-- platforms: begin -->
| | Mac Studio | Lenovo ThinkPad P1 | Linux workstation |
|:--|--:|--:|--:|
| CPU | Apple M4 Max | Intel Core Ultra 9 185H | Intel Xeon w9-3495X |
| Cores (logical), memory | 16, 128 GB | 22, 63 GB | 112, 125 GB |
| System | macOS 15.6 | Windows 11 | Ubuntu 22.04.5 LTS |
| Compiler | Apple Clang 17.0 | MSVC 19.52 | GCC 13.4 |
| Python, NumPy | 3.15.2, 2.5.5 | 3.12.9, 2.5.5 | 3.12.12, 2.2.6 |
| SIMD sorting | none | avx2 | avx512 |
| Version of ot1d, date | 0.5.1.dev0, 2026-11-22 | 0.5.1.dev0, 2026-11-22 | 0.5.1.dev0, 2026-11-22 |
| `np.sort(x)`, 10^7 values (s) | 0.328 | 0.13 | 0.0837 |
| `ot1d.parasort(x)`, 10^7 values, 1 thread (s) | 0.189 | 0.155 | 0.0636 |
| `ot1d.parasort(x, threads=8)` (s) | 0.0522 | 0.087 | 0.0541 |
| `ot1d.reference(x)`, 10^7 points (s) | 0.252 | 0.215 | 0.115 |
| `ot1d.reference(x, A)`, with weights (s) | 0.829 | 0.441 | 0.318 |
| `w1_reference`, 10^4 against 10^7 points, `'ecdf'` (ms) | 0.107 | 0.218 | 0.0713 |
| `w2_reference`, same sizes, `'ecdf'` (ms) | 0.106 | 0.246 | 0.0526 |
| `w1_reference`, same sizes, `'simplex'` (ms) | 9.98 | 22.2 | 10.8 |
| `w1(x, y, sorting=False)`, 10^7 sorted points each (ms) | 1.25 | 8.65 | 9.2 |
| `w1(x, y, method='ecdf', sorting=False)` (ms) | 50.1 | 124 | 83.6 |
| `w1(x, y)`, 10^7 unsorted points each, 1 thread (s) | 0.241 | 0.391 | 0.264 |
| `w1_batch`, per sample of 10^4 points, 1 thread (ms) | 0.201 | 0.383 | 0.0785 |
| `w1_batch`, per sample, 8 threads (ms) | 0.0297 | 0.0824 | 0.0195 |
<!-- platforms: end -->

Each column is written by the machine itself: after a checkout, `bash tests/check_platform.sh --save` (on Windows, `tests\check_platform.ps1 -Save`) runs the tests and the benchmark, records the timings in `tests/platforms/` and rebuilds this table. With the package already installed, `python tests/bench_platform.py --save` does only the last two steps.

Please contact us by email if you encounter any issues.

### Use of AI tools
The simplex-based algorithm (`method='simplex'`, the only one up to release 0.4) was designed and originally implemented by hand by the author.
For release 0.4.0, the author used [Claude Code](https://claude.com/claude-code) (the Anthropic CLI) with the Opus 5.5 model to debug and test the library.
In that process, Claude found and fixed several bugs, restructured parts of the C++ header [OT1D.hpp](https://github.com/stegua/ot1d/blob/main/include/OT1D.hpp) while fixing them, and wrote the test suites.

Release 0.5.0 is different: it was written by Claude (Claude Code, with the Opus 5.5 and Fable 5.2 models) under the supervision of the author.
This covers the implementation of the ecdf-based algorithm (`method='ecdf'`) in C++ and Python, the new functions (`w1`, `w2`, `reference`, `w1_reference`, `w1_batch`, and so on), their tests, the benchmark, and the documentation of the release.
The author set the requirements, chose the names and the interface, took the design decisions, and directed the reviews of the code, but did not write this code by hand.

Every change was reviewed and committed by the author, who remains solely responsible for the code.

### Author and maintainer
Stefano Gualandi, stefano.gualandi@gmail.com.
