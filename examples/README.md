# LAMG+ — running and timing the solver

LAMG+ is a lean, parameter-free, empirically *O(m)* algebraic-multigrid solver for
graph-Laplacian systems `L x = b` (and SDD systems). This directory shows how to run it
from the command line and time it. The solver itself lives in [`../src`](../src); the
test suite in [`../test`](../test).

## One-time setup

```bash
cd LAMG.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'      # main solver environment
```

The comparison example also needs the competitor environment (Laplacians.jl):

```bash
julia --project=scripts/competitor_env -e 'using Pkg; Pkg.instantiate()'
```

Graphs are MatrixMarket `.mtx` files under `data/` (e.g. SuiteSparse). Each example takes
a graph path and an optional relative-residual tolerance (default `1e-8`).

## 1. Solve one graph and time it

```bash
julia --project=. examples/solve_and_time.jl data/SNAP__web-Stanford.mtx 1e-8
```

```
graph: n = 213453 nodes, m = 1151741 edges
LAMG+  (tol = 1e-08)
  levels        : 17
  setup         :   0.41 s   (0.36 µs/edge)
  solve         :   0.31 s   (0.27 µs/edge, 3 cycles)
  total         :   0.72 s   (0.63 µs/edge)
  rel. residual : 4.1e-09   (CONVERGED)
```

The first solve is a warm-up, so the reported times exclude Julia's one-time JIT compilation.

## 2. Sweep a corpus and measure O(m) scaling

```bash
julia --project=. examples/run_corpus.jl data/lamg_full_list.txt corpus.csv 1e-8
```

Writes per-graph timings to `corpus.csv` and prints the empirical scaling exponent
`beta` in `total_time ~ m^beta` (β ≈ 1 is the O(m) statement) and the convergence rate.
On the full SuiteSparse test set this reports 100% convergence and β ≈ 1.02.

## 3. Compare against approximate-Cholesky (same process, language-fair)

```bash
julia --project=scripts/competitor_env examples/compare_approxchol.jl data/Gleich__flickr.mtx
```

Reports best-of-3 setup+solve time per nonzero (µs/nnz, the metric of Gao–Kyng–Spielman
2023) for LAMG+, AC (`approxchol_lap2`, the robust 2023 variant), and `approxchol_lap` (2016).

## Run the test suite (verify correctness)

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Or a single test file:

```bash
julia --project=. test/test_port_regression.jl     # faithful-port + refinement guards
julia --project=. test/test_caliber2.jl            # caliber-2 interpolation
```

## Using LAMG+ from your own code

```julia
using LAMG
L = laplacian(W)                                   # adjacency W -> Laplacian
x, info = solve(L, b; options = LAMGOptions(tol = 1e-8))
# reuse one hierarchy across many right-hand sides:
h = setup(L; options = LAMGOptions(tol = 1e-8))
x, info = solve(h, b; options = LAMGOptions(tol = 1e-8))
```

`LAMGOptions` is parameter-free by default; the two refinements over stock LAMG are
`caliber2_1d = true` (selective caliber-2 interpolation) and `agg_soc_τ = 0.05`
(strength-of-connection veto), both on by default. See the paper in
[`../doc/paper_program/lamg_plus.tex`](../doc/paper_program) for the algorithm.
