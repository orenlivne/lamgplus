using Test
using Random
using LinearAlgebra
using SparseArrays
using LAMG

@testset "coarsen" begin
    @testset "affinity values are in [0, 1]" begin
        rng = MersenneTwister(0xfa14)
        n, K = 16, 4
        X = 2 .* rand(rng, n, K) .- 1.0
        edges = [(i, i + 1) for i in 1:(n - 1)]
        affs = affinity(X, edges)
        @test all(0 <= v <= 1 for v in values(affs))
    end

    @testset "affinity = 1 for parallel node profiles" begin
        # Rows 1 and 2 are proportional (row 2 = 2 * row 1); affinity = 1.
        X = [1.0 2.0 3.0;
             2.0 4.0 6.0;
             5.0 1.0 0.0;
             0.0 0.0 1.0]
        affs = affinity(X, [(1, 2), (1, 3), (1, 4)])
        @test affs[(1, 2)] ≈ 1.0 atol = 1e-12
        @test affs[(1, 3)] < 1.0
        @test affs[(1, 4)] < 1.0
    end

    @testset "affinity = 0 for orthogonal node profiles" begin
        X = [1.0 0.0;
             0.0 1.0]
        affs = affinity(X, [(1, 2)])
        @test affs[(1, 2)] ≈ 0.0 atol = 1e-12
    end

    @testset "aggregate on a path graph produces ~n/2 aggregates" begin
        L = path_laplacian(32)
        ag = aggregate(L; rng = MersenneTwister(0xfa11))
        @test ag isa Aggregation
        @test length(ag.aggregate) == 32
        @test 8 <= ag.n_coarse <= 17     # ideally 16; allow slack
    end

    @testset "aggregate maps every fine node to a valid coarse index" begin
        L = grid2d_laplacian(8, 8)
        ag = aggregate(L; rng = MersenneTwister(0xfa12))
        @test length(ag.aggregate) == 64
        @test all(1 <= a <= ag.n_coarse for a in ag.aggregate)
        # Every aggregate index is used exactly once in the sorted unique set.
        @test sort(unique(ag.aggregate)) == collect(1:ag.n_coarse)
    end

    @testset "aggregate respects max_aggregate_size" begin
        L = grid2d_laplacian(8, 8)
        ag = aggregate(L; max_aggregate_size = 2,
                      rng = MersenneTwister(0xfa13))
        sizes = zeros(Int, ag.n_coarse)
        for a in ag.aggregate
            sizes[a] += 1
        end
        @test maximum(sizes) <= 2
    end

    @testset "aggregation → caliber-1 P preserves graph-Laplacian structure" begin
        L = path_laplacian(16)
        ag = aggregate(L; rng = MersenneTwister(0xfa15))
        P, R, Q = piecewise_constant_interpolation(ag.aggregate)
        Lc = galerkin_coarse_operator(L, P)
        @test is_graph_laplacian(Lc)
    end

    @testset "jaccard_priority tie-break: valid partition, default unchanged" begin
        L = grid2d_laplacian(10, 10)
        ag0 = aggregate(L; rng = MersenneTwister(0xfa16))
        agj = aggregate(L; jaccard_priority = true, rng = MersenneTwister(0xfa16))
        for ag in (ag0, agj)
            @test length(ag.aggregate) == 100
            @test all(1 <= a <= ag.n_coarse for a in ag.aggregate)
            @test sort(unique(ag.aggregate)) == collect(1:ag.n_coarse)
        end
        # jaccard P must still yield a valid coarse graph Laplacian
        P, _, _ = piecewise_constant_interpolation(agj.aggregate)
        @test is_graph_laplacian(galerkin_coarse_operator(L, P))
        # flag default is off ⇒ identical result to the default tie-break
        ag_default = aggregate(L; jaccard_priority = false, rng = MersenneTwister(0xfa16))
        @test ag_default.aggregate == ag0.aggregate
    end
end
