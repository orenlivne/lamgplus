using Test
using LinearAlgebra
using SparseArrays
using LAMG

@testset "interpolation" begin
    @testset "caliber-1 P preserves constants" begin
        agg = [1, 1, 2, 2, 3, 3, 4, 4]
        P, _, _ = piecewise_constant_interpolation(agg)
        @test size(P) == (8, 4)
        # P · 1_c = 1_f
        @test P * ones(4) ≈ ones(8)
    end

    @testset "caliber-1 R averages within an aggregate" begin
        agg = [1, 1, 2, 2, 3, 3, 4, 4]
        _, R, _ = piecewise_constant_interpolation(agg)
        @test size(R) == (4, 8)
        # R applied to a constant fine vector returns the same constant.
        @test R * ones(8) ≈ ones(4)
        # R applied to (1,2,3,...,8) gives aggregate means.
        @test R * collect(1.0:8) ≈ [1.5, 3.5, 5.5, 7.5]
    end

    @testset "Q = Pᵀ" begin
        agg = [1, 1, 2, 2, 3, 3, 4, 4]
        P, _, Q = piecewise_constant_interpolation(agg)
        @test Q == P'
    end

    @testset "Galerkin coarse operator stays a graph Laplacian" begin
        # KEY structural property for max-flow: caliber-1 + nonneg fine
        # weights ⇒ caliber-1 coarse weights also nonneg, and zero row sum.
        L = path_laplacian(8)
        agg = [1, 1, 2, 2, 3, 3, 4, 4]
        P, _, _ = piecewise_constant_interpolation(agg)
        Lc = galerkin_coarse_operator(L, P)
        @test is_graph_laplacian(Lc)
        @test size(Lc) == (4, 4)
        # Concretely, after aggregating pairs on a path, the coarse Laplacian
        # is a path Laplacian on 4 nodes with weights = 1 on each coarse edge
        # (the shared "interface" fine edge of each adjacent pair).
        @test Lc ≈ path_laplacian(4)
    end

    @testset "non-trivial graph: graph-Laplacian structure preserved" begin
        # Build a small irregular graph and an arbitrary partition.
        L = sparse([2.0 -1.0 -1.0  0.0;
                   -1.0  3.0 -1.0 -1.0;
                   -1.0 -1.0  3.0 -1.0;
                    0.0 -1.0 -1.0  2.0])
        @test is_graph_laplacian(L)
        agg = [1, 1, 2, 2]
        P, _, _ = piecewise_constant_interpolation(agg)
        Lc = galerkin_coarse_operator(L, P)
        @test is_graph_laplacian(Lc)
    end

    @testset "aggregation_from_partition round-trips" begin
        agg = aggregation_from_partition([[1, 2], [3, 4, 5], [6]], 6)
        @test agg == [1, 1, 2, 2, 2, 3]
    end
end
