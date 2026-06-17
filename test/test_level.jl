using Test
using LinearAlgebra
using SparseArrays
using LAMG

@testset "level" begin
    n = 16
    L = path_laplacian(n)
    rx = GaussSeidelRelaxer(L)

    @testset "create_finest_level (with default b = I)" begin
        lvl = create_finest_level(L, rx)
        @test size(lvl) == n
        @test lvl.r === nothing && lvl.p === nothing && lvl.q === nothing
        x = ones(n)
        # operator() with lam=0 returns A*x.
        @test operator(lvl, x) ≈ L * x
        # operator() with lam != 0 returns (A - lam B)*x; B=I here.
        @test operator(lvl, x; lam = 0.5) ≈ L * x .- 0.5 .* x
    end

    @testset "relax! delegates to the relaxer" begin
        lvl = create_finest_level(L, rx)
        x = randn(n); x .-= sum(x) / n
        b = zeros(n)
        r0 = norm(operator(lvl, x))
        relax!(lvl, x, b; sweeps = 5)
        x .-= sum(x) / n
        @test norm(operator(lvl, x)) < r0
    end

    @testset "coarse-level transfer operators are accessible" begin
        # 2-aggregate pairing on a length-4 path
        agg = [1, 1, 2, 2]
        P, R, Q = piecewise_constant_interpolation(agg)
        Lc = galerkin_coarse_operator(path_laplacian(4), P)
        rxc = GaussSeidelRelaxer(Lc)
        lvl_c = Level(Lc, sparse(1.0I, 2, 2), rxc, R, P, Q)
        @test size(lvl_c) == 2
        xc = [1.0, 2.0]
        @test interpolate_op(lvl_c, xc) ≈ P * xc
        x = [1.0, 1.0, 2.0, 2.0]
        @test coarsen_op(lvl_c, x) ≈ R * x
        @test restrict_op(lvl_c, x) ≈ Q * x
    end
end
