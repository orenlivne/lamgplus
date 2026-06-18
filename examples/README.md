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

## 4. Demo notebook — LAMG+ vs approxChol vs AC

A self-contained Jupyter notebook that times all three solvers on a synthetic grid and on a
graph downloaded from the SuiteSparse test set, illustrating the degree crossover (approxChol
wins the low-degree grid; LAMG+ wins the high-degree finite-element graph):

```bash
cd examples
julia --project=. -e 'using Pkg; Pkg.instantiate()'                       # one-time
julia --project=. -e 'using IJulia; jupyterlab()'                         # or open in any Jupyter
```

Then open [`lamgplus_demo.ipynb`](lamgplus_demo.ipynb). It has its own environment (LAMG+,
Laplacians.jl, and a SuiteSparse downloader, via `examples/Project.toml`) and ships with executed
output. The other `.jl` examples above use the main solver environment (`--project=.` from the repo
root), not this one.

## 5. Reproduce the comparison-table numbers

[`reproduce_comparison.ipynb`](reproduce_comparison.ipynb) re-runs **LAMG+ and approxChol live** on four
entries of the paper's multi-solver comparison table and checks the reproduced timing/accuracy against
the reported values. It uses the same code path as the full benchmark (the solver wrappers in
[`../scripts/repro_lib.jl`](../scripts/repro_lib.jl), extracted from `competitor_benchmark.jl`).

```bash
# one-time: lean env with LAMG + Laplacians.jl (run from the repo root)
julia --project=examples/repro_env -e 'using Pkg; Pkg.develop(path="."); Pkg.add("Laplacians"); Pkg.instantiate()'
# place SNAP__web-Stanford, SNAP__web-Google, GHS_psdef__bmwcra_1, Boeing__pwtk in ../data
# (or set $LAMGPLUS_DATA), then execute:
cd examples
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=julia-1.12 reproduce_comparison.ipynb
```

It prints a reproduced-vs-paper table, asserts the winner and 1e-8 convergence reproduce exactly, and
reports the wall-clock agreement (≤1.3× on the reference machine). Ships with executed output. For the
full 201-graph table and the other five solvers, run `../scripts/competitor_benchmark.jl`.

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
