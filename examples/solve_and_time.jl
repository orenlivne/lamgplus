#!/usr/bin/env julia
# examples/solve_and_time.jl — solve a graph-Laplacian system with LAMG+ and report timing.
#
#   Usage:  julia --project=. examples/solve_and_time.jl <graph.mtx> [tol]
#   E.g.:   julia --project=. examples/solve_and_time.jl data/SNAP__web-Stanford.mtx 1e-8
#
# Loads the .mtx as a Laplacian L, restricts to its largest connected component,
# builds a compatible random RHS b = L*x_true, and runs LAMG+ to the given
# relative-residual tolerance (default 1e-8). Reports setup time, solve time,
# cycle count, and the achieved residual. The first solve is a warm-up so the
# reported times are free of Julia's one-time JIT compilation.

using LAMG, LinearAlgebra, SparseArrays, Random, Printf
import LAMG: solve, setup, LAMGOptions
include(joinpath(@__DIR__, "mm_reader.jl"))

length(ARGS) >= 1 || error("usage: julia --project=. examples/solve_and_time.jl <graph.mtx> [tol]")
path = ARGS[1]
tol  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1e-8

println("loading $path ...")
W, L = reduce_to_lcc(read_mm_adj(path)...)
n = size(L, 1); m = (nnz(L) - n) ÷ 2
@printf("graph: n = %d nodes, m = %d edges\n", n, m)

rng = MersenneTwister(1)
xt = randn(rng, n); xt .-= sum(xt) / n          # zero-mean exact solution
b  = L * xt; bnorm = norm(b)
opts = LAMGOptions(tol = tol, max_cycles = 200)

# warm-up: compile every code path on this exact system, so the timed run is JIT-free
let; h0 = setup(L; options = opts); solve(h0, b; options = opts); end

t_setup = @elapsed h = setup(L; options = opts)
t_solve = @elapsed ((x, info) = solve(h, b; options = opts))
res = norm(b - L * x) / bnorm

@printf("\nLAMG+  (tol = %.0e)\n", tol)
@printf("  levels        : %d\n", length(h.levels))
@printf("  setup         : %7.3f s   (%.3f µs/edge)\n", t_setup, 1e6 * t_setup / m)
@printf("  solve         : %7.3f s   (%.3f µs/edge, %d cycles)\n", t_solve, 1e6 * t_solve / m, info.cycles)
@printf("  total         : %7.3f s   (%.3f µs/edge)\n", t_setup + t_solve, 1e6 * (t_setup + t_solve) / m)
@printf("  rel. residual : %.2e   (%s)\n", res, res <= tol ? "CONVERGED" : "NOT CONVERGED")
