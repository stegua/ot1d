# -*- coding: utf-8 -*-
"""Short benchmark of ot1d on this machine, run by tests/check_platform.sh.

It prints the timings quoted in the README (measured there on an Intel Xeon
w9-3495X with AVX-512), so that they can be compared on another platform.
"""
import os
import statistics
import time

import numpy as np
import ot1d


def med(f, reps=7):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def row(name, seconds, unit="ms"):
    scale = 1e3 if unit == "ms" else 1.0
    print("  {:58s} {:10.3f} {}".format(name, seconds * scale, unit))


cores = os.cpu_count() or 1
r = np.random.default_rng(7)
n, m = 10_000_000, 10_000
x = r.lognormal(0, 1, n)
a = r.random(n) + 0.1
y = r.lognormal(0.1, 1, m)
b = r.random(m) + 0.1
xs, ys = np.sort(x), np.sort(r.lognormal(0.1, 1, n))

print("SIMD sorting: {}, cores: {}".format(ot1d.simd(), cores))
print("Reference of 10^7 points (README: sort 0.08 s, prepare 0.11 s, 0.06 s if sorted, 0.32 s with weights)")
row("np.sort", med(lambda: np.sort(x), 5), "s")
row("ot1d.reference(x)", med(lambda: ot1d.reference(x), 5), "s")
row("ot1d.reference(x, sorting=False), x sorted", med(lambda: ot1d.reference(xs, sorting=False), 5), "s")
row("ot1d.reference(x, A)", med(lambda: ot1d.reference(x, a), 5), "s")

R, RW = ot1d.reference(x), ot1d.reference(x, a)
print("One sample of 10^4 points against it (README: 0.076, 0.051, 0.20, 0.30, 10 ms)")
row("w1_reference, ecdf", med(lambda: ot1d.w1_reference(R, y), 51))
row("w2_reference, ecdf", med(lambda: ot1d.w2_reference(R, y), 51))
row("w1_reference, ecdf, weighted sample", med(lambda: ot1d.w1_reference(R, y, b), 51))
row("w1_reference, ecdf, weighted reference", med(lambda: ot1d.w1_reference(RW, y), 51))
row("w1_reference, simplex", med(lambda: ot1d.w1_reference(R, y, method="simplex"), 21))

print("Two samples of 10^7 points (README, sorted input: simplex 9 ms, OT1D 8.5 ms, ecdf 84 ms)")
row("w1, simplex, sorting=False", med(lambda: ot1d.w1(xs, ys, sorting=False), 9))
row("OT1D, sorting=False", med(lambda: ot1d.OT1D(xs, ys, p=1, sorting=False), 9))
row("w1, ecdf, sorting=False", med(lambda: ot1d.w1(xs, ys, method="ecdf", sorting=False), 5))
row("w1, simplex, unsorted, threads=1", med(lambda: ot1d.w1(x, ys[::-1], threads=1), 3), "s")

print("Threads (parallel sort) on 10^7 points")
for t in sorted({1, 2, 4, 8, min(16, cores)}):
    if t > cores:
        continue
    t1 = med(lambda: ot1d.parasort(x.copy(), threads=t), 3) - med(lambda: x.copy(), 3)
    t2 = med(lambda: ot1d.w1_plan(x[:2_000_000], y, threads=t), 3)
    print("  threads={:2d}: sort of 10^7 values {:7.3f} s   w1_plan, 2*10^6 against 10^4 points {:7.3f} s".format(t, t1, t2))

chunks = [r.lognormal(0.1, 1, m) for _ in range(200)]
print("w1_batch, 200 samples of 10^4 points")
for t in sorted({1, min(8, cores)}):
    row("threads={}".format(t), med(lambda: ot1d.w1_batch(R, chunks, threads=t), 5) / len(chunks))
