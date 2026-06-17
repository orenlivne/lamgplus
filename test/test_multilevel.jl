using Test
using LinearAlgebra
using SparseArrays
using LAMG

@testset "multilevel" begin
    L = path_laplacian(8)
    rx = GaussSeidelRelaxer(L)
    finest = create_finest_level(L, rx)

    @testset "create / length / iteration" begin
        mlh = Multilevel(finest)
        @test length(mlh) == 1
        @test num_levels(mlh) == 1
        @test finest_level(mlh) === finest
        levels = collect(mlh)
        @test levels == [finest]
    end

    @testset "push! appends coarser levels" begin
        mlh = Multilevel(finest)
        agg = [1, 1, 2, 2, 3, 3, 4, 4]
        P, R, Q = piecewise_constant_interpolation(agg)
        Lc = galerkin_coarse_operator(L, P)
        coarse_relax = GaussSeidelRelaxer(Lc)
        coarse = Level(Lc, sparse(1.0I, 4, 4), coarse_relax, R, P, Q)
        push!(mlh, coarse)
        @test length(mlh) == 2
        @test mlh[2] === coarse
        @test mlh[end] === coarse
    end
end
