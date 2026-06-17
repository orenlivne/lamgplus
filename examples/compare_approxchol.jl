#!/usr/bin/env julia
# examples/compare_approxchol.jl — time LAMG+ against Spielman's approximate-Cholesky solvers
# on one graph, in the same Julia process, and print LAMG+'s convergence history.
#
#   Requires the competitor environment (which has Laplacians.jl):
#   Usage:  julia --project=scripts/competitor_env examples/compare_approxchol.jl <graph.mtx> [tol]
#
#   A graph where LAMG+ beats AC (3.8x), so you can see both:
#     julia --project=scripts/competitor_env examples/compare_approxchol.jl data/Oberwolfach__bone010.mtx
#
# Reports best-of-3 setup+solve time per nonzero (µs/nnz, the metric of Gao-Kyng-Spielman
# 2023) for LAMG+, AC = approxchol_lap2 (robust 2023), and approxchol_lap (2016), each to the
# same relative-residual tolerance, plus LAMG+'s per-cycle residual history.

using LAMG, Laplacians, LinearAlgebra, SparseArrays, Random, Printf
import LAMG: solve, LAMGOptions
include(joinpath(@__DIR__, "mm_reader.jl"))

length(ARGS) >= 1 || error("usage: julia --project=scripts/competitor_env examples/compare_approxchol.jl <graph.mtx> [tol]")
path = ARGS[1]; tol = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1e-8
bestof(f) = minimum(@elapsed(f()) for _ in 1:3)

W, _ = reduce_to_lcc(read_mm_adj(path)...)
La = lap(W); n = size(La, 1); nz = nnz(W)
rng = MersenneTwister(7); xt = randn(rng, n); xt .-= sum(xt) / n; b = La * xt; bn = norm(b)
@printf("graph: %s\n  n = %d, nnz = %d (%d edges), tol = %.0e\n\n", basename(path), n, nz, nz ÷ 2, tol)

opts = LAMGOptions(tol = tol, max_cycles = 300)
xl, info = solve(La, b; options = opts); rl = norm(b - La * xl) / bn      # capture history (warm)
tl = bestof(() -> solve(La, b; options = opts))
ac_pc = Int[]; xa = Laplacians.approxchol_lap2(W; tol = tol, pcgIts = ac_pc)(b); ra = norm(b - La * xa) / bn
ta = bestof(() -> Laplacians.approxchol_lap2(W; tol = tol)(b))
af_pc = Int[]; xf = Laplacians.approxchol_lap(W; tol = tol, pcgIts = af_pc)(b); rf = norm(b - La * xf) / bn
tf = bestof(() -> Laplacians.approxchol_lap(W; tol = tol)(b))

@printf("%-24s %10s %12s %8s %10s\n", "solver", "time (s)", "µs/nnz", "iters", "rel.res")
@printf("%-24s %10.3f %12.3f %8d %10.1e\n", "LAMG+",                 tl, 1e6 * tl / nz, info.cycles, rl)
@printf("%-24s %10.3f %12.3f %8d %10.1e\n", "AC (approxchol_lap2)",  ta, 1e6 * ta / nz, isempty(ac_pc) ? -1 : ac_pc[1], ra)
@printf("%-24s %10.3f %12.3f %8d %10.1e\n", "approxchol_lap (2016)", tf, 1e6 * tf / nz, isempty(af_pc) ? -1 : af_pc[1], rf)
@printf("\nspeedup of LAMG+ over AC: %.2fx\n", ta / tl)

println("\nLAMG+ convergence history  (‖b − Ax‖ per cycle):")
for (k, r) in enumerate(info.residual_history)
    if k == 1
        @printf("  cycle %2d : %.3e   (initial)\n", 0, r)
    else
        @printf("  cycle %2d : %.3e   factor %.3f\n", k - 1, r, info.conv_factors[k - 1])
    end
end
