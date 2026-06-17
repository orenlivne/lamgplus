using Test
using Random
using LinearAlgebra
using SparseArrays
using LAMG

# Tier-2 (convergence-equivalence) oracle. Unlike test_solve_equivalence.jl (which
# pins the solve bit-for-bit), this certifies a NUMERIC kernel change — one that
# reorganizes floating-point arithmetic (e.g. caching 1/diag and multiplying instead
# of dividing in Gauss–Seidel) — does not REGRESS the solver: the cycle count is
# unchanged, the system is still solved, and the solution matches the reference to
# solve-accuracy (not machine precision). Same reference snapshots as the bit-identical
# oracle; only the tolerance on norm(x) is loosened from 1e-12 to 1e-6.
@testset "solve convergence-equivalence (numeric kernels)" begin
    refs = [
        ("grid2d", grid2d_laplacian(40, 40),          6, 40.10463194147957),
        ("grid3d", grid3d_laplacian(16, 16, 16),      6, 63.88991995235978),
        ("path",   path_laplacian(500),               1, 22.48752640804134),
        ("rand",   random_graph_laplacian(800, 0.02; rng = MersenneTwister(7)), 5, 28.22972920565193),
    ]
    opt = LAMGOptions(tol = 1e-10, max_cycles = 100, reorder = false)
    for (nm, L, cyc_ref, normx_ref) in refs
        n = size(L, 1)
        Random.seed!(42); xt = randn(n); xt .-= sum(xt) / n; b = L * xt
        Random.seed!(42); h = setup(L; options = opt)
        x, info = solve(h, b; options = opt)
        @test info.cycles == cyc_ref                       # convergence unchanged (key guard)
        @test isapprox(norm(x), normx_ref; rtol = 1e-6)    # solution matches to solve-accuracy
        @test norm(L * x - b) / norm(b) < 1e-9             # actually solves the system
        @test abs(sum(x)) < 1e-9                           # stays in the zero-mean (range) space
    end
end
