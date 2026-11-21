"""NumPy reference implementation (executable specification) of OT1D v0.5.0 `Reference`.

Exact W1 and W2 between a FIXED reference sample and many query samples ("chunks"), using the
quantile form of 1D optimal transport

    W_p^p = int_0^1 |F^{-1}(u) - G^{-1}(u)|^p du ,

instead of the north-west-corner merge. The reference is prepared once; each query then costs
O(m log m) to sort the chunk plus O(m) (W2) or O(m log n) (W1) — no pass over the n reference points.

Fit (once): sort r_1 <= ... <= r_n; c = mean(r); UNSCALED prefix sums P1[i] = sum_{l<i} z_l,
  P2[i] = sum_{l<i} z_l^2 of z = r - c (no division by n).
  Q1(u) = int_0^u (F^{-1}(v) - c) dv,  Q2(u) = int_0^u (F^{-1}(v) - c)^2 dv  are piecewise
  linear in u, with breakpoints at u = i/n.

Integer quantile axis (uniform case): scale u by N = n m. Reference point i owns [i m, (i+1) m),
  chunk point j owns [j n, (j+1) n); every breakpoint and overlap length is an exact integer <= N.
  At integer position num, with i = num // m:   N Qk(num / N) = m Pk[i] + (num - i m) z_i^k.
  All per-segment terms are accumulated in these N-scaled units and divided by N ONCE at the end.

Query, chunk sorted y_1 <= ... <= y_m: on u in [(j-1)/m, j/m) the chunk quantile is y_j, so
  W2:  N int (F^{-1} - y_j)^2 du = N dQ2 - 2 (y_j - c) N dQ1 + (y_j - c)^2 n
  W1:  F^{-1}(u) - y_j changes sign once, at u* = clip(F(y_j), (j-1)/m, j/m)  with F(y) = k/n,
       k = #{r_i <= y} (one binary search), so
       int |F^{-1} - y_j| du = (y_j - c)(u* - a) - [Q1(u*) - Q1(a)] + [Q1(b) - Q1(u*)] - (y_j - c)(b - u*)

PRECISION RULE (measured, see tests): every quantile position (j/m, k/n, and their clips) must be
represented EXACTLY as an integer in units of 1/(n m), like OT1D's existing uniform merge; no
intermediate may exceed N = n m (index = num // m, never num * n), so int64 suffices for N < 2^63.
Only lengths <= max(n, m) are converted to double (exact); N is used once, as the final divisor. With
floating-point positions (e.g. U = cumsum(1/n)) the drift ~ n*eps multiplies values of the
reference and costs up to 6e-7 relative error on heavy tails; with integer positions the error is
~1e-14, the same as the merge.

PRECISION RULE 2: prefix sums must be accumulated with compensation (Neumaier) during the fit.
A plain double running sum costs up to 2.6e-10 relative error at n = 10^6 (data with many ties);
compensated accumulation brings it to ~1e-14. Storing them as plain doubles is then enough.

KNOWN LIMIT (inherent to the expansion used for W2): W2^2 has an ABSOLUTE error of order
eps * mean((r - c)^2), so when the true W2 is below ~1e-7 * std(r) (e.g. a query identical to the
reference) the result is ~1e-8 * std(r) instead of 0. Irrelevant for drift monitoring; document it.
W1 has no such floor.

Weighted reference / chunk: the same formulas with cumulative weights instead of i/n, j/m, and a
binary search to find the segment of u. Positions are then floating point; use compensated
(Kahan/Neumaier) cumulative sums and normalize the last entry to exactly 1.
"""

from __future__ import annotations

import numpy as np


def _prefix(v, div=1):
    """Prefix sums [0, v1, v1+v2, ...] / div, accumulated in extended precision, stored as double.

    Stand-in for the C++ rule: accumulate with a Neumaier (compensated) sum in the single fit pass.
    Plain double cumsum costs up to 2.6e-10 relative error at n = 10^6 on data with many ties.
    """
    acc = np.cumsum(np.asarray(v, dtype=np.longdouble)) / div
    return np.concatenate(([0.0], acc.astype(np.float64)))


class Reference:
    """Fixed reference sample prepared once for many exact W1 / W2 queries."""

    def __init__(self, x, w=None):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or len(x) == 0:
            raise ValueError("reference must be a non-empty 1D array")
        if not np.isfinite(x).all():
            raise ValueError("reference contains NaN or inf")
        order = np.argsort(x, kind="stable")
        self.r = np.ascontiguousarray(x[order])
        self.n = n = len(x)
        if w is None:
            self.uniform = True
            self.c = float(np.mean(self.r))
            z = self.r - self.c
            self.P1 = _prefix(z)  # unscaled: sum of z, no / n
            self.P2 = _prefix(z * z)
        else:
            w = np.asarray(w, dtype=np.float64)[order]
            if w.shape != x.shape or (w < 0).any() or not np.isfinite(w).all() or w.sum() <= 0:
                raise ValueError("weights must be finite, non-negative, same length, positive sum")
            self.uniform = False
            ww = w / w.sum()
            self.c = float(np.dot(ww, self.r))
            z = self.r - self.c
            self.U = np.concatenate(([0.0], np.cumsum(ww)))
            self.U[-1] = 1.0
            self.S1 = _prefix(ww * z, 1)
            self.S2 = _prefix(ww * z * z, 1)

    # ------------------------------------------------------------------ uniform reference & chunk
    def _NQ_int(self, num, m):
        """N*Q1, N*Q2 (N = n m) at integer position num in [0, N]; no division anywhere.

        Position num lies in reference segment i = num // m with integer offset num - i m <= m,
        so N Qk = m Pk[i] + (num - i m) z_i^k. No intermediate exceeds N (an earlier version
        computed num * n, up to n^2 m, and overflowed int64 at n=1e7, m=1e5).
        """
        i = np.minimum(num // m, self.n - 1)
        off = (num - i * m).astype(np.float64)  # exact: off <= m
        z = self.r[i] - self.c
        return m * self.P1[i] + off * z, m * self.P2[i] + off * z * z

    def _uniform(self, y, p):
        n, m = self.n, len(y)
        if n * m >= 2**63:  # positions are int64 in [0, N]; C++ can switch to unsigned __int128
            raise OverflowError("n*m too large for int64 positions")
        N = float(n * m)
        j = np.arange(m, dtype=np.int64)
        a, b = j * n, (j + 1) * n  # chunk interval [a, b) on the integer axis
        yc = y - self.c
        q1a, q2a = self._NQ_int(a, m)
        q1b, q2b = self._NQ_int(b, m)
        if p == 2:  # everything in N-scaled units; one division at the end
            t = (q2b - q2a) - 2 * yc * (q1b - q1a) + (yc * yc) * n
            return float(np.sqrt(max(t.sum() / N, 0.0)))
        k = np.searchsorted(self.r, y, side="right").astype(np.int64)  # F(y_j) = k/n
        s = np.clip(k * m, a, b)  # sign change of F^{-1} - y_j, integer position
        q1s, _ = self._NQ_int(s, m)
        t = yc * (s - a) - (q1s - q1a) + (q1b - q1s) - yc * (b - s)
        return float(t.sum() / N)

    # ------------------------------------------------------------------ general (weighted) path
    def _Q_w(self, u):
        if self.uniform:  # uniform reference, weighted chunk: floating positions are unavoidable
            i = np.minimum((u * self.n).astype(np.int64), self.n - 1)
            Ui = i / self.n
            S1i, S2i = self.P1[i] / self.n, self.P2[i] / self.n
        else:
            i = np.minimum(np.searchsorted(self.U, u, side="right") - 1, self.n - 1)
            Ui = self.U[i]
            S1i, S2i = self.S1[i], self.S2[i]
        z = self.r[i] - self.c
        return S1i + (u - Ui) * z, S2i + (u - Ui) * z * z

    def _F(self, t):
        k = np.searchsorted(self.r, t, side="right")
        return k / self.n if self.uniform else self.U[k]

    def _weighted(self, y, bw, p):
        bw = bw / bw.sum()
        cb = np.cumsum(bw)
        cb[-1] = 1.0
        a = np.concatenate(([0.0], cb[:-1]))
        b = cb
        yc = y - self.c
        q1a, q2a = self._Q_w(a)
        q1b, q2b = self._Q_w(b)
        if p == 2:
            t = (q2b - q2a) - 2 * yc * (q1b - q1a) + yc * yc * (b - a)
            return float(np.sqrt(max(t.sum(), 0.0)))
        s = np.clip(self._F(y), a, b)
        q1s, _ = self._Q_w(s)
        t = yc * (s - a) - (q1s - q1a) + (q1b - q1s) - yc * (b - s)
        return float(t.sum())

    # ------------------------------------------------------------------ public API
    def distance(self, y, p=1, b=None, sorted=False) -> float:
        """Exact W_p (p = 1 or 2) between the reference and sample y (optional weights b)."""
        if p not in (1, 2):
            raise ValueError("p must be 1 or 2")
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1 or len(y) == 0:
            raise ValueError("query must be a non-empty 1D array")
        if not np.isfinite(y).all():
            raise ValueError("query contains NaN or inf")
        if b is None:
            if not sorted:
                y = np.sort(y)
            if self.uniform:
                return self._uniform(y, p)
            return self._weighted(y, np.full(len(y), 1.0), p)
        b = np.asarray(b, dtype=np.float64)
        if b.shape != y.shape or (b < 0).any() or b.sum() <= 0:
            raise ValueError("query weights must be non-negative, same length, positive sum")
        if not sorted:
            o = np.argsort(y, kind="stable")
            y, b = y[o], b[o]
        return self._weighted(y, b, p)

    def w1(self, y, b=None, sorted=False) -> float:
        return self.distance(y, 1, b, sorted)

    def w2(self, y, b=None, sorted=False) -> float:
        return self.distance(y, 2, b, sorted)

    def nbytes(self) -> int:
        arrs = (self.r, self.P1, self.P2) if self.uniform else (self.r, self.S1, self.S2, self.U)
        return sum(a.nbytes for a in arrs)


# ---------------------------------------------------------------------- v0.5.0 API names
# The same functions as the module ot1d, so that test_reference.py runs unchanged on the
# prototype and on the C++ build (method is ignored: the prototype implements only 'ecdf').

reference = Reference


def w1_reference(ref, Y, B=None, method="ecdf", sorting=True):
    return ref.distance(Y, 1, B, not sorting)


def w2_reference(ref, Y, B=None, method="ecdf", sorting=True):
    return ref.distance(Y, 2, B, not sorting)


def _batch(p, X, Ys, A=None, Bs=None):
    ref = X if isinstance(X, Reference) else Reference(X, A)
    Bs = [None] * len(Ys) if Bs is None else Bs
    return np.array([ref.distance(y, p, b) for y, b in zip(Ys, Bs)])


def w1_batch(X, Ys, A=None, Bs=None, method="ecdf", threads=1):
    return _batch(1, X, Ys, A, Bs)


def w2_batch(X, Ys, A=None, Bs=None, method="ecdf", threads=1):
    return _batch(2, X, Ys, A, Bs)
