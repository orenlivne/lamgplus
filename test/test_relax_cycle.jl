using Test
using LinearAlgebra
using SparseArrays
using Random
using LAMG

@testset "relax_cycle" begin
    @testset "2-level relax cycle on path Laplacian reduces residual" begin
        # Hand-build a 2-level hierarchy and verify the cycle reduces residual.
        n = 16
        L = path_laplacian(n)
        rx = GaussSeidelRelaxer(L)
        finest = create_finest_level(L, rx)

        agg = [div(i - 1, 2) + 1 for i in 1:n]   # pair consecutive nodes
        P, R, Q = piecewise_constant_interpolation(agg)
        Lc = galerkin_coarse_operator(L, P)
        rxc = GaussSeidelRelaxer(Lc)
        coarse = Level(Lc, sparse(1.0I, n ÷ 2, n ÷ 2), rxc, R, P, Q)

        mlh = Multilevel(finest)
        push!(mlh, coarse)

        cyc = relax_cycle(mlh; γ = 1.0, ν_pre = 2, ν_post = 2, ν_coarsest = 4)

        rng = MersenneTwister(0xabba)
        x = randn(rng, n); x .-= sum(x) / n
        r0 = norm(L * x)
        run_cycle!(cyc, x)
        x = LAMG.result(cyc.processor, 1)
        x .-= sum(x) / n
        @test norm(L * x) < r0
    end
end
