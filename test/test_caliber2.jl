using Test
using LinearAlgebra
using SparseArrays
using Random
using LAMG

# anisotropic PxP grid Laplacian: horizontal weight wx, vertical weight wy
function _c2_grid(P; wx = 1.0, wy = 1.0)
    idx(i, j) = (j - 1) * P + i
    I = Int[]; J = Int[]; V = Float64[]
    for j in 1:P, i in 1:P
        if i < P; push!(I, idx(i, j)); push!(J, idx(i + 1, j)); push!(V, wx); end
        if j < P; push!(I, idx(i, j)); push!(J, idx(i, j + 1)); push!(V, wy); end
    end
    W = sparse(I, J, V, P * P, P * P); W = W + W'
    d = vec(sum(W; dims = 2)); spdiagm(0 => d) - W
end

@testset "caliber2 interpolation" begin
    @testset "deterministic LS weight on a uniform path = 0.5" begin
        # path of 6 nodes, pair aggregates, single linear test vector: the interior
        # F-points (2 and 4) sit halfway between their two coarse seeds, so w = 0.5.
        L = path_laplacian(6)
        agg = [1, 1, 2, 2, 3, 3]
        X = reshape(collect(1.0:6.0), 6, 1)              # linear ramp, one TV
        P, R, Q, nup = caliber2_interpolation(agg, X, L; τ = 0.5)
        @test P * ones(3) ≈ ones(6)                      # constants preserved exactly
        @test P[2, 1] ≈ 0.5 atol = 1e-9                  # node 2 ← 0.5·agg1 + 0.5·agg2
        @test P[2, 2] ≈ 0.5 atol = 1e-9
        @test P[4, 2] ≈ 0.5 atol = 1e-9                  # node 4 ← 0.5·agg2 + 0.5·agg3
        @test P[4, 3] ≈ 0.5 atol = 1e-9
        @test nup == 2                                   # exactly the two interior F-points
        @test Q == P'
        @test R * ones(6) ≈ ones(3)                      # R preserves constants
        @test Matrix(R * P) ≈ Matrix(1.0I, 3, 3)         # R·P = I exactly (FAS-safe restriction)
    end

    @testset "guard rejects extrapolation (w ∉ [0,1]) → caliber-1 fallback" begin
        L = path_laplacian(6)
        agg = [1, 1, 2, 2, 3, 3]
        # node 2's TV value (5) lies outside [seed1=1, seed2=2] ⇒ w < 0 ⇒ reject.
        X = reshape([1.0, 5.0, 2.0, 2.0, 3.0, 3.0], 6, 1)
        P, _, _, _ = caliber2_interpolation(agg, X, L; τ = 0.5)
        @test count(!iszero, P[2, :]) == 1               # fell back to caliber-1
        @test P[2, agg[2]] ≈ 1.0
        @test P * ones(3) ≈ ones(6)                      # still preserves constants
    end

    @testset "constants & structure on an anisotropic grid (random TVs)" begin
        L = _c2_grid(16; wx = 1.0, wy = 1e-3); n = size(L, 1)
        ag = aggregate(L; K = 8, ν = 3, rng = MersenneTwister(0xca12))
        X = 2 .* rand(MersenneTwister(7), n, 8) .- 1.0
        P, R, Q, nup = caliber2_interpolation(ag.aggregate, X, L; τ = 0.5)
        @test size(P) == (n, ag.n_coarse)
        @test P * ones(ag.n_coarse) ≈ ones(n)            # constants preserved
        @test R * ones(n) ≈ ones(ag.n_coarse)            # R preserves constants
        @test Matrix(R * P) ≈ Matrix(1.0I, ag.n_coarse, ag.n_coarse)  # R·P = I (FAS-safe, non-diag PᵀP)
        @test Q == P'
        @test nup > 0                                    # gate fires on the 1-D pockets
        # every caliber-2 row is a convex combination (guard ⇒ weights in [0,1])
        @test all(0 - 1e-12 .<= nonzeros(P) .<= 1 + 1e-12)
    end

    @testset "Galerkin coarse op: symmetric, zero row sum (signed Laplacian)" begin
        # Caliber-2 can introduce small POSITIVE off-diagonals (M-matrix loss), so the
        # coarse operator is a signed Laplacian — symmetric with zero row sum, but not
        # necessarily a graph Laplacian. Both invariants the solver relies on still hold.
        L = _c2_grid(16; wx = 1.0, wy = 1e-3)
        ag = aggregate(L; K = 8, ν = 3, rng = MersenneTwister(0xca13))
        X = 2 .* rand(MersenneTwister(8), size(L, 1), 8) .- 1.0
        P, _, _, _ = caliber2_interpolation(ag.aggregate, X, L; τ = 0.5)
        Lc = galerkin_coarse_operator(L, P)
        @test Lc ≈ Lc'                                   # symmetric
        @test maximum(abs, vec(sum(Lc; dims = 2))) < 1e-9   # zero row sum (null = constants)
    end

    @testset "gate is self-targeting: fires far more on anisotropic than isotropic" begin
        La = _c2_grid(24; wx = 1.0, wy = 1e-3); na = size(La, 1)
        aga = aggregate(La; K = 8, ν = 3, rng = MersenneTwister(1))
        Xa = 2 .* rand(MersenneTwister(2), na, 8) .- 1.0
        _, _, _, nup_a = caliber2_interpolation(aga.aggregate, Xa, La; τ = 0.5)

        Li = _c2_grid(24; wx = 1.0, wy = 1.0); ni = size(Li, 1)
        agi = aggregate(Li; K = 8, ν = 3, rng = MersenneTwister(1))
        Xi = 2 .* rand(MersenneTwister(2), ni, 8) .- 1.0
        _, _, _, nup_i = caliber2_interpolation(agi.aggregate, Xi, Li; τ = 0.5)

        @test nup_a / na > 0.25                          # upgrades a large share of 1-D pockets
        @test nup_a > nup_i                              # strictly more than isotropic
        @test nup_i / ni < nup_a / na                    # isotropic stays comparatively caliber-1
    end

    @testset "end-to-end: flag converges and speeds up the anisotropic solve" begin
        L = _c2_grid(24; wx = 1.0, wy = 1e-3); n = size(L, 1)
        xt = randn(MersenneTwister(0), n); xt .-= sum(xt) / n; b = L * xt
        _, i0 = solve(L, b; options = LAMGOptions(caliber2_1d = false))
        _, i1 = solve(L, b; options = LAMGOptions(caliber2_1d = true))
        @test i0.final_residual <= 1e-7 * norm(b)        # baseline converges
        @test i1.final_residual <= 1e-7 * norm(b)        # feature converges (correctness)
        @test i1.cycles <= i0.cycles                     # and no slower
    end
end
