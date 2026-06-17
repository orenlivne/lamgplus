using Test
using Random
using LinearAlgebra
using SparseArrays
using LAMG

# Reference outputs captured from the pre-optimization (allocating) solve cycle. The
# allocation-free rewrite is a PURE mechanical change (same arithmetic, written in-place),
# so it MUST reproduce these bit-for-bit (cycle counts identical, norms to machine precision).
# If any of these move, the rewrite changed the numerics — that is a bug, not an optimization.
@testset "solve allocation-free equivalence (reference snapshots)" begin
    refs = [
        ("grid2d", grid2d_laplacian(40, 40),          6, 40.10463194147957),
        ("grid3d", grid3d_laplacian(16, 16, 16),      6, 63.88991995235978),
        ("path",   path_laplacian(500),               1, 22.48752640804134),
        ("rand",   random_graph_laplacian(800, 0.02; rng = MersenneTwister(7)), 5, 28.22972920565193),
    ]
    # Pin γ=1.5: this oracle pins the CYCLE KERNEL (GS, transfers, elim, recomb) bit-for-bit;
    # that is independent of the cycle index. The default γ moved to 1.25 (wall-clock optimum),
    # but the reference snapshots below were captured at γ=1.5, so we hold γ fixed here.
    opt = LAMGOptions(tol = 1e-10, max_cycles = 100, γ = 1.5, γ_coarse = 1.5, reorder = false)
    for (nm, L, cyc_ref, normx_ref) in refs
        n = size(L, 1)
        Random.seed!(42); xt = randn(n); xt .-= sum(xt) / n; b = L * xt
        Random.seed!(42); h = setup(L; options = opt)
        x, info = solve(h, b; options = opt)
        @test info.cycles == cyc_ref                       # identical iteration count
        @test isapprox(norm(x), normx_ref; rtol = 1e-12)   # identical solution (machine precision)
        @test norm(L * x - b) / norm(b) < 1e-9             # actually solves the system
        @test abs(sum(x)) < 1e-9                           # stays in the zero-mean (range) space
    end
end
