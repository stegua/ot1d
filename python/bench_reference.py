#!/usr/bin/env python
"""Synthetic benchmark for ot1d.reference (v0.5.0): many chunks against one fixed reference.

Per-chunk time (median), fit time and resident memory, single thread, for:

  merge            OT1D(ref, chunk, p)                       stateless: sorts both every call
  merge_presorted  OT1D(ref_sorted, sort(chunk), sorting=False)  reference sorted once (v0.4.x best)
  ref_proto        reference_prototype.Reference.distance   NumPy spec of the new algorithm
  ref_native       ot1d.w1_reference(ot1d.reference(ref), chunk)   the C++ implementation, method='ecdf'
  binned_1e4       NannyML's binned W1 (10,000 bins, tail fix), the approximation to beat (p=1 only)
  binned_1e6       same with 1,000,000 bins (needed for <1.5% error on the NYC taxi data)

Expected outcome (what v0.5.0 must show): ref_native per-chunk time flat in n_ref, at or below
binned_1e4, and orders of magnitude below merge_presorted for n_ref >> m; values equal to the
merge (relative error column).

    python bench_reference.py                      # default grid, writes bench_reference.csv
    python bench_reference.py --quick              # small grid, ~1 minute
    python bench_reference.py --n-ref 1e8 --chunk 1e4   # one point, large reference
"""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import time

import numpy as np
import ot1d
from ot1d import OT1D

from reference_prototype import Reference as RefProto

RefNative = ot1d.reference


def timeit(fn, min_time=0.25, min_reps=3, max_reps=100):
    fn()
    ts = []
    t0 = time.perf_counter()
    while len(ts) < max_reps and (len(ts) < min_reps or time.perf_counter() - t0 < min_time):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def once(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def sample(dist, k, shift, rng):
    if dist == "normal":
        return rng.normal(shift, 1.0, k)
    if dist == "lognormal":
        return rng.lognormal(shift, 1.0, k)
    if dist == "ties":
        return np.floor(rng.normal(40 + 5 * shift, 12, k))
    raise ValueError(dist)


class Binned:
    """NannyML 0.13.1 binned W1 with the tail fix, on NumPy arrays (no pandas overhead)."""

    def __init__(self, ref, n_bins):
        h, self.e = np.histogram(ref, bins=n_bins)
        self.lo, self.hi, self.B = self.e[0], self.e[-1], n_bins
        self.w = self.e[1] - self.e[0]
        self.cdf = np.cumsum(h / len(ref))

    @staticmethod
    def _ecdf(v):
        v = np.sort(v)
        x, c = np.unique(v, return_counts=True)
        return x, np.cumsum(c) / len(v)

    def distance(self, y):
        m = len(y)
        ys, yb = y[y < self.lo], y[y > self.hi]
        ns, nb = len(ys), len(yb)
        ts = tb = 0.0
        if ns:
            x, c = self._ecdf(np.concatenate((ys, [self.lo])))
            ts = np.sum(c[:-1] * np.diff(x)) * (ns + 1) / m
        if nb:
            x, c = self._ecdf(np.concatenate(([self.hi], yb)))
            tb = np.sum((1 - c)[:-1] * np.diff(x)) * (nb + 1) / m
        h, _ = np.histogram(y, bins=self.B, range=(self.lo, self.hi))
        cdf = np.cumsum(h / m) + ns / m
        return float(np.sum(np.abs(self.cdf - cdf)) * self.w + ts + tb)

    def nbytes(self):
        return self.e.nbytes + self.cdf.nbytes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-ref", nargs="+", type=float, default=[1e4, 1e5, 1e6, 1e7])
    ap.add_argument("--chunk", nargs="+", type=float, default=[1e3, 1e4, 1e5])
    ap.add_argument("--dists", nargs="+", default=["normal", "lognormal", "ties"])
    ap.add_argument("--p", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--quick", action="store_true", help="n_ref 1e4..1e6, chunk 1e3..1e4, normal only")
    ap.add_argument("--no-binned-1e6", action="store_true")
    ap.add_argument("--out", default="bench_reference.csv")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.quick:
        a.n_ref, a.chunk, a.dists = [1e4, 1e5, 1e6], [1e3, 1e4], ["normal"]

    rng = np.random.default_rng(a.seed)
    fields = ["dist", "p", "n_ref", "m", "variant", "seconds_per_chunk", "fit_seconds", "state_mb",
              "value", "relerr_vs_merge", "host"]
    host = platform.node()
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for dist in a.dists:
            for n in map(int, a.n_ref):
                ref = sample(dist, n, 0.0, rng)
                fits = {}
                ref_sorted, fits["merge_presorted"] = once(lambda: np.sort(ref))
                fits["merge"] = 0.0
                proto, fits["ref_proto"] = once(lambda: RefProto(ref))
                states = {"merge": 0.0, "merge_presorted": ref_sorted.nbytes / 2**20,
                          "ref_proto": proto.nbytes() / 2**20}
                if RefNative is not None:
                    native, fits["ref_native"] = once(lambda: RefNative(ref))
                    states["ref_native"] = native.nbytes / 2**20
                binned = {}
                for B, name in [(10_000, "binned_1e4"), (1_000_000, "binned_1e6")]:
                    if name == "binned_1e6" and a.no_binned_1e6:
                        continue
                    binned[name], fits[name] = once(lambda B=B: Binned(ref, B))
                    states[name] = binned[name].nbytes() / 2**20
                for m in map(int, a.chunk):
                    y = sample(dist, m, 0.3, rng)
                    for p in a.p:
                        truth = float(OT1D(ref, y, p=p))
                        V = {
                            "merge": lambda: float(OT1D(ref, y, p=p)),
                            "merge_presorted": lambda: float(OT1D(ref_sorted, np.sort(y), p=p, sorting=False)),
                            "ref_proto": lambda: proto.distance(y, p=p),
                        }
                        if RefNative is not None:
                            wref = ot1d.w1_reference if p == 1 else ot1d.w2_reference
                            V["ref_native"] = lambda: float(wref(native, y))
                        if p == 1:
                            for name, b in binned.items():
                                V[name] = lambda b=b: b.distance(y)
                        for name, fn in V.items():
                            sec = timeit(fn)
                            val = fn()
                            row = dict(dist=dist, p=p, n_ref=n, m=m, variant=name, seconds_per_chunk=sec,
                                       fit_seconds=fits[name], state_mb=states[name], value=val,
                                       relerr_vs_merge=abs(val - truth) / abs(truth) if truth else abs(val),
                                       host=host)
                            w.writerow(row)
                            f.flush()
                            print(f"{dist:9s} p={p} n_ref={n:>11,} m={m:>8,} {name:16s} "
                                  f"{sec*1e3:10.3f} ms/chunk  fit {fits[name]:7.3f} s  err {row['relerr_vs_merge']:.1e}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
