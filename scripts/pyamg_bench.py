#!/usr/bin/env python3
# Standalone pyAMG (smoothed-aggregation) benchmark for ONE graph-Laplacian adjacency .mtx.
# Run as a subprocess (PyCall-free) so a pyAMG failure cannot crash the Julia harness.
# Reads a symmetric non-negative adjacency W, builds L = D - W, Dirichlet-pins one node for SPD,
# and times setup + solve to relative residual 1e-8. The reported times are measured INSIDE
# Python (perf_counter), so they exclude interpreter startup / import overhead.
# Output (one line):  "<setup_s> <solve_s> <ok:0|1>"   (or "nan nan 0" on failure).
import sys, time
import numpy as np, scipy.io, scipy.sparse as sp
TOL, MAXIT = 1e-8, 150
try:
    import pyamg
    W = scipy.io.mmread(sys.argv[1]).tocsr()
    n = W.shape[0]
    d = np.asarray(W.sum(axis=1)).ravel()
    L = (sp.diags(d) - W).tocsr()
    keep = np.arange(1, n)                       # Dirichlet pin node 0 -> SPD
    A = L[keep][:, keep].tocsr()
    rng = np.random.default_rng(1)
    b = A @ rng.standard_normal(n - 1)
    pyamg.smoothed_aggregation_solver(A).solve(b, tol=TOL, maxiter=MAXIT)   # warm-up (untimed)
    t0 = time.perf_counter(); ml = pyamg.smoothed_aggregation_solver(A); t1 = time.perf_counter()
    x = ml.solve(b, tol=TOL, maxiter=MAXIT); t2 = time.perf_counter()
    rel = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-30)
    print(f"{t1-t0:.6f} {t2-t1:.6f} {1 if rel <= TOL*100 else 0}")
except Exception:
    print("nan nan 0")
