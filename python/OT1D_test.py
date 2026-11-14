# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:57:37 2020

@author: Gualandi
"""

import numpy as np
from time import perf_counter

from scipy.stats import wasserstein_distance as W1
import ot


from OT1D import OT1D, parasort


def Test1(p, q, threads=8):
    sorting=True
    print('\n--------------- TEST 1: Unsorted input --------------------')

    print('For OT1D using', threads,'threads\n')

    print("Testing W1, samples of deltas, n=m")
    t0 = perf_counter()
    z = W1(p, q)
    print("Scipy: z =", round(z, 8), "time =", round(perf_counter() - t0, 4))

    t0 = perf_counter()
    z = ot.lp.wasserstein_1d(p, q, p=1)
    print("POT  : z =", round(z, 8), "time =", round(perf_counter() - t0, 4))

    t0 = perf_counter()
    z = OT1D(p, q, p=1, sorting=sorting, threads=threads)
    print("SG   : z =", round(z, 8), "time =", round(perf_counter() - t0, 4))

    print("\nTesting W2, samples of deltas, n=m")
    t0 = perf_counter()
    z = ot.lp.wasserstein_1d(p, q, p=2)
    print("POT  : z =", round(z, 8), "time =", round(perf_counter() - t0, 4))

    t0 = perf_counter()
    z = OT1D(p, q, p=2, sorting=sorting, threads=threads)
    print("SG   : z =", round(z, 8), "time =", round(perf_counter() - t0, 4))


    print("\nTesting W1, samples with weights")
    t0 = perf_counter()
    z = W1(p, q, a, b)
    print("Scipy: z =", round(z, 8), "time =", round(perf_counter() - t0, 4))

    t0 = perf_counter()
    z = ot.lp.wasserstein_1d(p, q, a, b, p=1)
    print("POT  : z =", round(z, 8), "time =", round(perf_counter() - t0, 4))

    t0 = perf_counter()
    z = OT1D(p, q, a, b, p=1, sorting=sorting, threads=threads)
    print("SG   : z =", round(z, 8), "time =", round(perf_counter() - t0, 4))

    print("\nTesting W2, samples with weights")
    t0 = perf_counter()
    z = ot.lp.wasserstein_1d(p, q, a, b, p=2)
    print("POT  : z =", round(z, 8), "time =", round(perf_counter() - t0, 4))

    t0 = perf_counter()
    z = OT1D(p, q, a, b, p=2, sorting=sorting, threads=threads)
    print("SG   : z =", round(z, 8), "time =", round(perf_counter() - t0, 4))
        
    
    
def Test2(p, q, a, b, threads=8):
    print('\n--------------- TEST 2: Sorted input --------------------')

    print('For OT1D using', threads,'threads\n')

    sorting=False

    t0 = perf_counter()
    # Sort input data once for all    
    parasort(p, a, threads)
    parasort(q, b, threads)
    
    print('Parallel sorting: time =', round(perf_counter() - t0, 3), 'sec\n')

    print("Testing W1, samples of deltas, n=m")
    t0 = perf_counter()
    z = W1(p, q)
    print("Scipy: z =", round(z, 8), "time =", round(perf_counter() - t0, 4))

    t0 = perf_counter()
    z = ot.lp.wasserstein_1d(p, q, p=1)
    print("POT  : z =", round(z, 8), "time =", round(perf_counter() - t0, 4))

    t0 = perf_counter()
    z = OT1D(p, q, p=1, sorting=sorting, threads=threads)
    print("SG   : z =", round(z, 8), "time =", round(perf_counter() - t0, 4))

    print("\nTesting W2, samples of deltas, n=m")
    t0 = perf_counter()
    z = ot.lp.wasserstein_1d(p, q, p=2)
    print("POT  : z =", round(z, 8), "time =", round(perf_counter() - t0, 4))

    t0 = perf_counter()
    z = OT1D(p, q, p=2, sorting=sorting, threads=threads)
    print("SG   : z =", round(z, 8), "time =", round(perf_counter() - t0, 4))


    print("\nTesting W1, samples with weights")
    t0 = perf_counter()
    z = W1(p, q, a, b)
    print("Scipy: z =", round(z, 8), "time =", round(perf_counter() - t0, 4))

    t0 = perf_counter()
    z = ot.lp.wasserstein_1d(p, q, a, b, p=1)
    print("POT  : z =", round(z, 8), "time =", round(perf_counter() - t0, 4))

    t0 = perf_counter()
    z = OT1D(p, q, a, b, p=1, sorting=sorting, threads=threads)
    print("SG   : z =", round(z, 8), "time =", round(perf_counter() - t0, 4))

    print("\nTesting W2, samples with weights")
    t0 = perf_counter()
    z = ot.lp.wasserstein_1d(p, q, a, b, p=2)
    print("POT  : z =", round(z, 8), "time =", round(perf_counter() - t0, 4))

    t0 = perf_counter()
    z = OT1D(p, q, a, b, p=2, sorting=sorting, threads=threads)
    print("SG   : z =", round(z, 8), "time =", round(perf_counter() - t0, 4))


def Test3(N = 500000, samples=30, threads=8):
    print('\n--------------- TEST 3: Unsorted input (average runtime) --------------------')

    print('For OT1D using', threads, 'threads\n')

    np.random.seed(13)

    sorting=True

    Ls = [0 for i in range(11)]
    print('\nrunning test', end=' ')
    for _ in range(samples):
        print('.', end=' ', sep='', flush=True)
        
        p = np.random.uniform(1, 10, N)
        q = np.random.uniform(3, 5, N)

        # Random weights for samples
        a = np.random.uniform(500, 1000, N)
        b = np.random.uniform(0, 1000, N)
        a = a/np.sum(a)
        b = b/np.sum(b)    
    
        t0 = perf_counter()
        z = W1(p, q)
        Ls[1] += perf_counter() - t0
    
        t0 = perf_counter()
        z = ot.lp.wasserstein_1d(p, q, p=1)
        Ls[2] += perf_counter() - t0

        t0 = perf_counter()
        z = OT1D(p, q, p=1, sorting=sorting, threads=threads)
        Ls[3] += perf_counter() - t0

        t0 = perf_counter()
        z = ot.lp.wasserstein_1d(p, q, p=2)
        Ls[4] += perf_counter() - t0

        t0 = perf_counter()
        z = OT1D(p, q, p=2, sorting=sorting, threads=threads)
        Ls[5] += perf_counter() - t0

        t0 = perf_counter()
        z = W1(p, q, a, b)
        Ls[6] += perf_counter() - t0
    
        t0 = perf_counter()
        z = ot.lp.wasserstein_1d(p, q, a, b, p=1)
        Ls[7] += perf_counter() - t0
    
        t0 = perf_counter()
        z = OT1D(p, q, a, b, p=1, sorting=sorting, threads=threads)
        Ls[8] += perf_counter() - t0
    
        t0 = perf_counter()
        z = ot.lp.wasserstein_1d(p, q, a, b, p=2)
        Ls[9] += perf_counter() - t0
    
        t0 = perf_counter()
        z = OT1D(p, q, a, b, p=2, sorting=sorting, threads=threads)
        Ls[10] += perf_counter() - t0

    print()

    print("Testing W1, samples of deltas, n=m")
    print("Scipy: average time =", round(Ls[1]/samples, 3),
          "speedup =", round(Ls[1]/Ls[3], 1))
    print("POT  : average time =", round(Ls[2]/samples, 3),
          "speedup =",round(Ls[2]/Ls[3], 1))
    print("SG   : average time =", round(Ls[3]/samples, 3),
          "speedup =",round(Ls[3]/Ls[3], 1))

    print("\nTesting W2, samples of deltas, n=m")
    print("POT  : average time =", round(Ls[4]/samples, 3),
          "speedup =", round(Ls[4]/Ls[5], 1))
    print("SG   : average time =", round(Ls[5]/samples, 3),
          "speedup =", round(Ls[5]/Ls[5], 1))

    print("\nTesting W1, samples with weights")
    print("Scipy: average time =", round(Ls[6]/samples, 3),
          "speedup =", round(Ls[6]/Ls[8], 1))
    print("POT  : average time =", round(Ls[7]/samples, 3),
          "speedup =", round(Ls[7]/Ls[8], 1))
    print("SG   : average time =", round(Ls[8]/samples, 3),
          "speedup =", round(Ls[8]/Ls[8], 1))

    print("\nTesting W2, samples with weights")
    print("POT  : average time =", round(Ls[9]/samples, 3),
          "speedup =", round(Ls[9]/Ls[10], 1))
    print("SG   : average time =", round(Ls[10]/samples, 3),
          "speedup =", round(Ls[10]/Ls[10], 1))


def Test4(N = 500000, samples=30, threads=8):
    print('\n--------------- TEST 4: Sorted input (average runtime) --------------------')

    print('For OT1D using', threads, 'threads\n')

    np.random.seed(13)

    sorting=False

    Ls = [0 for i in range(11)]
    print('running test', end=' ')
    for _ in range(samples):
        print('.', end=' ', sep='', flush=True)
        
        p = np.random.uniform(1, 10, N)
        q = np.random.uniform(3, 5, N)

        # Random weights for samples
        a = np.random.uniform(500, 1000, N)
        b = np.random.uniform(0, 1000, N)
        a = a/np.sum(a)
        b = b/np.sum(b)    

        t0 = perf_counter()
        # Sort input data once for all    
        parasort(p, a, threads)   
        parasort(q, b, threads)
        Ls[0] += perf_counter() - t0
    
        t0 = perf_counter()
        z = W1(p, q)
        Ls[1] += perf_counter() - t0
    
        t0 = perf_counter()
        z = ot.lp.wasserstein_1d(p, q, p=1)
        Ls[2] += perf_counter() - t0

        t0 = perf_counter()
        z = OT1D(p, q, p=1, sorting=sorting, threads=threads)
        Ls[3] += perf_counter() - t0

        t0 = perf_counter()
        z = ot.lp.wasserstein_1d(p, q, p=2)
        Ls[4] += perf_counter() - t0

        t0 = perf_counter()
        z = OT1D(p, q, p=2, sorting=sorting, threads=threads)
        Ls[5] += perf_counter() - t0

        t0 = perf_counter()
        z = W1(p, q, a, b)
        Ls[6] += perf_counter() - t0
    
        t0 = perf_counter()
        z = ot.lp.wasserstein_1d(p, q, a, b, p=1)
        Ls[7] += perf_counter() - t0
    
        t0 = perf_counter()
        z = OT1D(p, q, a, b, p=1, sorting=sorting, threads=threads)
        Ls[8] += perf_counter() - t0
    
        t0 = perf_counter()
        z = ot.lp.wasserstein_1d(p, q, a, b, p=2)
        Ls[9] += perf_counter() - t0
    
        t0 = perf_counter()
        z = OT1D(p, q, a, b, p=2, sorting=sorting, threads=threads)
        Ls[10] += perf_counter() - t0

    print()
    print('Parallel sorting: time =', round(Ls[0]/samples, 3), 'sec\n')

    print("Testing W1, samples of deltas, n=m")
    print("Scipy: average time =", round(Ls[1]/samples, 3),
          "speedup =", round(Ls[1]/Ls[3], 1))
    print("POT  : average time =", round(Ls[2]/samples, 3),
          "speedup =",round(Ls[2]/Ls[3], 1))
    print("SG   : average time =", round(Ls[3]/samples, 3),
          "speedup =",round(Ls[3]/Ls[3], 1))

    print("\nTesting W2, samples of deltas, n=m")
    print("POT  : average time =", round(Ls[4]/samples, 3),
          "speedup =", round(Ls[4]/Ls[5], 1))
    print("SG   : average time =", round(Ls[5]/samples, 3),
          "speedup =", round(Ls[5]/Ls[5], 1))

    print("\nTesting W1, samples with weights")
    print("Scipy: average time =", round(Ls[6]/samples, 3),
          "speedup =", round(Ls[6]/Ls[8], 1))
    print("POT  : average time =", round(Ls[7]/samples, 3),
          "speedup =", round(Ls[7]/Ls[8], 1))
    print("SG   : average time =", round(Ls[8]/samples, 3),
          "speedup =", round(Ls[8]/Ls[8], 1))

    print("\nTesting W2, samples with weights")
    print("POT  : average time =", round(Ls[9]/samples, 3),
          "speedup =", round(Ls[9]/Ls[10], 1))
    print("SG   : average time =", round(Ls[10]/samples, 3),
          "speedup =", round(Ls[10]/Ls[10], 1))

def Test5(threads=8, seed=13):
    print('\n--------------- TEST 5: W2 against POT (correctness) --------------------')

    print('Comparing OT1D with sqrt(POT), since POT returns W2^2\n')

    rng = np.random.default_rng(seed)
    cases = [(1, 1), (2, 3), (10, 10), (17, 5), (1000, 1000), (1000, 777),
             (100000, 100000), (100000, 65537), (5000000, 5000000)]
    worst = 0.0
    for m, n in cases:
        x = rng.normal(0, 1, m)
        y = rng.normal(3, 2, n)
        a = rng.uniform(0.1, 1, m)
        b = rng.uniform(0.1, 1, n)
        a, b = a/np.sum(a), b/np.sum(b)

        for weighted in (False, True):
            A, B = (a, b) if weighted else (None, None)

            t0 = perf_counter()
            if weighted:
                ref = np.sqrt(ot.lp.wasserstein_1d(x, y, a, b, p=2))
            else:
                ref = np.sqrt(ot.lp.wasserstein_1d(x, y, p=2))
            t_pot = perf_counter() - t0

            for th in sorted({1, threads}):
                t0 = perf_counter()
                z = OT1D(x, y, A, B, p=2, sorting=True, threads=th)
                t_sg = perf_counter() - t0

                err = abs(z - ref)/max(1.0, abs(ref))
                worst = max(worst, err)
                assert err < 1e-8, (m, n, weighted, th, z, ref)

                if th == threads:
                    print("m = {:>7}, n = {:>7}, weights = {:<5} | POT: z = {:.8f} time = {:.4f} | "
                          "SG: z = {:.8f} time = {:.4f}".format(m, n, str(weighted), ref, t_pot, z, t_sg))

            # Sorted input (sorting=False) and transportation plan
            if m <= 100000:
                ix, iy = np.argsort(x), np.argsort(y)
                xs, ys = x[ix], y[iy]
                As, Bs = (a[ix], b[iy]) if weighted else (None, None)
                z = OT1D(xs, ys, As, Bs, p=2, sorting=False, threads=threads)
                assert abs(z - ref)/max(1.0, abs(ref)) < 1e-8, ('sorted', m, n, weighted, z, ref)

                z, T, W = OT1D(x, y, A, B, p=2, sorting=True, threads=threads, plan=True)
                assert abs(z - ref)/max(1.0, abs(ref)) < 1e-8, ('plan', m, n, weighted, z, ref)
                cost = np.sqrt(np.sum(W * (x[T[:, 0]] - y[T[:, 1]])**2))
                assert abs(cost - ref)/max(1.0, abs(ref)) < 1e-8, ('plan cost', m, n, weighted, cost, ref)

    print('\nAll W2 values match POT: max relative error =', '{:.2e}'.format(worst))


# -----------------------------------------------
#   MAIN function
# -----------------------------------------------
if __name__ == "__main__":
    # Example 1 from here:
    #  https://www.kaggle.com/nhan1212/some-statistical-distances/comments
    
    np.random.seed(13)
    N = 5000000
    
    # Uniform samples
    if False:
        p = np.random.uniform(1, 2, N)
        q = np.random.uniform(0, 1, N)
    else:
        p = np.random.normal(0, 1, N)
        q = np.random.normal(10, 1, N)

    # Random weights for samples
    a = np.random.uniform(500, 1000, N)
    b = np.random.uniform(0, 1000, N)
    a = a/np.sum(a)
    b = b/np.sum(b)    

    Test1(p, q, threads=1)
    Test2(p, q, a, b, threads=1)

    Test3(threads=8)
    Test4(threads=8)
    Test5(threads=8)