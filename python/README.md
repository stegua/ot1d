# OT1D: Discrete Optimal Transport in 1D by Linear Programming

[![build-update Actions Status](https://github.com/stegua/ot1d/workflows/build-update/badge.svg)](https://github.com/stegua/ot1d/actions)  [![Downloads](https://pepy.tech/badge/ot1d)](https://pepy.tech/project/ot1d) [![Downloads](https://pepy.tech/badge/ot1d/month)](https://pepy.tech/project/ot1d) [![Downloads](https://pepy.tech/badge/ot1d/week)](https://pepy.tech/project/ot1d)

The OT1D library offers a simple but efficient implementation of an algorithm to compute the Kantorovich-Wasserstein distance between two empirical measures defined in dimension 1, that is, measures whose support points lie in **R**.
We designed the algorithm by directly exploiting the [complementary slackness](https://en.wikipedia.org/wiki/Linear_programming#Complementary_slackness) conditions of Linear Programming.
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
|**[2021/06/21]**|*Testing and evaluating OT1D*|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stegua/ot1d/blob/main/notebooks/Testing_OT1D.ipynb)|


## Usage
The main function of the **OT1D** library is the following:

```python
z = OT1D(x, y, A=None, B=None, p=1, sorting=True, threads=8, plan=False)
```

The parameters of the function are:

* `x`: the support points of the first measure.
* `y`: the support points of the second measure.
* `A`: the (nonnegative) weights of the first measure. If equal to `None`, all the samples have the same mass.
* `B`: the (nonnegative) weights of the second measure. If equal to `None`, all the samples have the same mass. `A` and `B` must be given together. If their total masses differ, a `RuntimeWarning` is issued and only the smaller total mass is transported (partial value and plan).
* `p`: the order of the Wasserstein distance (`p=1` or `p=2`).
* `sorting`: if equal to `True`, the function sorts the support points given as input; if equal to `False`, the input must already be sorted.
* `threads`: the number of threads used by the parallel sorting algorithm.
* `plan`: if equal to `True`, the function also returns an optimal transportation plan, as a triple `z, T, W`: the row `T[k] = (i, j)` means that the mass `W[k]` moves from `x[i]` to `y[j]` (see the example [`interpolate.py`](https://github.com/stegua/ot1d/blob/main/python/interpolate.py)).

The first four parameters can be given as numpy arrays (*preferred*) or Python lists.

### Sorting at the speed of light
In addition, we expose the following in-place parallel sorting function:
```python
parasort(x, A=None, threads=8)
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
We sort the arrays by using a customized parallel sorting algorithm implemented in C++, which combines the very fast [pdqsort](https://github.com/orlp/pdqsort)
with [parasort](https://github.com/baserinia/parallel-sort). See the linked webpages for the licenses of these two libraries.


### Prerequisites for compilation

To compile the source code and the Python wrapper, you only need:

* A C++ compiler compliant with the `C++11` standard.
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

### Testing

To test the library, you can run the following command:

```bash
python3 basic_test.py
```

A basic usage example is the following:

```python
import numpy as np
from OT1D import OT1D, parasort

np.random.seed(13)

N = 1000000

# Uniform samples
x = np.random.uniform(1, 2, N)
y = np.random.uniform(0, 1, N)

z = OT1D(x, y, p=2, sorting=True, threads=16)

print('Wasserstein distance of order 2, W2(x,y) =', z)
```
and the output should be similar to:
```
Wasserstein distance of order 2, W2(x,y) = 1.0002549459050794
```

### Testing for correctness
The repository includes two test suites, which compare every function against an independent reference implementation, and check that the transportation plans are feasible and optimal:

```bash
make test                         # C++ tests (also: make sanitize)
cd python && python3 test_wrapper.py  # Python wrapper tests (requires scipy)
```

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
Scipy: average time = 0.18 speedup = 9.7
POT  : average time = 0.168 speedup = 9.0
OT1D : average time = 0.019 speedup = 1.0

Testing W2, samples of deltas, n=m
POT  : average time = 0.124 speedup = 6.6
OT1D : average time = 0.019 speedup = 1.0

Testing W1, samples with weights
Scipy: average time = 0.193 speedup = 5.1
POT  : average time = 0.165 speedup = 4.3
OT1D : average time = 0.038 speedup = 1.0

Testing W2, samples with weights
POT  : average time = 0.12 speedup = 3.1
OT1D : average time = 0.039 speedup = 1.0

--------------- TEST 4: Sorted input (average runtime) --------------------
For OT1D using 8 threads

running test . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
Parallel sorting: time = 0.027 sec

Testing W1, samples of deltas, n=m
Scipy: average time = 0.077 speedup = 30.1
POT  : average time = 0.116 speedup = 45.5
OT1D : average time = 0.003 speedup = 1.0

Testing W2, samples of deltas, n=m
POT  : average time = 0.09 speedup = 35.5
OT1D : average time = 0.003 speedup = 1.0

Testing W1, samples with weights
Scipy: average time = 0.087 speedup = 6.7
POT  : average time = 0.117 speedup = 9.0
OT1D : average time = 0.013 speedup = 1.0

Testing W2, samples with weights
POT  : average time = 0.087 speedup = 6.5
OT1D : average time = 0.013 speedup = 1.0

--------------- TEST 5: W2 against POT (correctness) --------------------
Comparing OT1D with sqrt(POT), since POT returns W2^2

m =       1, n =       1, weights = False | POT: z = 4.98342038 time = 0.0002 | OT1D: z = 4.98342038 time = 0.0000
m =       1, n =       1, weights = True  | POT: z = 4.98342038 time = 0.0001 | OT1D: z = 4.98342038 time = 0.0000
m =       2, n =       3, weights = False | POT: z = 3.46826482 time = 0.0001 | OT1D: z = 3.46826482 time = 0.0000
m =       2, n =       3, weights = True  | POT: z = 4.10324470 time = 0.0001 | OT1D: z = 4.10324470 time = 0.0000
m =      10, n =      10, weights = False | POT: z = 2.86894031 time = 0.0001 | OT1D: z = 2.86894031 time = 0.0000
m =      10, n =      10, weights = True  | POT: z = 2.93894066 time = 0.0001 | OT1D: z = 2.93894066 time = 0.0000
m =      17, n =       5, weights = False | POT: z = 3.18112337 time = 0.0001 | OT1D: z = 3.18112337 time = 0.0000
m =      17, n =       5, weights = True  | POT: z = 3.40960305 time = 0.0001 | OT1D: z = 3.40960305 time = 0.0000
m =    1000, n =    1000, weights = False | POT: z = 3.18595988 time = 0.0002 | OT1D: z = 3.18595988 time = 0.0000
m =    1000, n =    1000, weights = True  | POT: z = 3.23862020 time = 0.0002 | OT1D: z = 3.23862020 time = 0.0001
m =    1000, n =     777, weights = False | POT: z = 3.12133828 time = 0.0002 | OT1D: z = 3.12133828 time = 0.0000
m =    1000, n =     777, weights = True  | POT: z = 3.08777081 time = 0.0002 | OT1D: z = 3.08777081 time = 0.0001
m =  100000, n =  100000, weights = False | POT: z = 3.16226321 time = 0.0168 | OT1D: z = 3.16226321 time = 0.0061
m =  100000, n =  100000, weights = True  | POT: z = 3.15995238 time = 0.0175 | OT1D: z = 3.15995238 time = 0.0067
m =  100000, n =   65537, weights = False | POT: z = 3.15278684 time = 0.0136 | OT1D: z = 3.15278684 time = 0.0041
m =  100000, n =   65537, weights = True  | POT: z = 3.15374537 time = 0.0181 | OT1D: z = 3.15374537 time = 0.0058
m = 5000000, n = 5000000, weights = False | POT: z = 3.16297776 time = 1.9577 | OT1D: z = 3.16297776 time = 0.1855
m = 5000000, n = 5000000, weights = True  | POT: z = 3.16249183 time = 1.9836 | OT1D: z = 3.16249183 time = 0.4073

All W2 values match POT: max relative error = 1.42e-11
```

Please contact us by email if you encounter any issues.

### Author and maintainer
Stefano Gualandi, stefano.gualandi@gmail.com.
