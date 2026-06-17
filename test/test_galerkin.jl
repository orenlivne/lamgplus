using Test
using LinearAlgebra
using SparseArrays
using Random
using LAMG

# Hand-computed Galerkin equivalence tests, inspired by `UTestGalerkinCaliber1Mex.m`.
# Verifies A_c = Pᵀ A P matches the explicit hand-computed result for small,
# tractable cases.

@testset "Galerkin caliber-1 coarse operator" begin
    @testset "path-4 → path-2: aggregate pairs, weights add along the shared edge" begin
        L = path_laplacian(4)
        # Aggregate: {1, 2} → 1, {3, 4} → 2.
        agg = [1, 1, 2, 2]
        P, _, _ = piecewise_constant_interpolation(agg)
        Lc = sparse(galerkin_coarse_operator(L, P))
        # The shared edge between aggregates is (2, 3) with weight 1.
        # Coarse Laplacian has a single edge between aggregate 1 and 2 with weight 1.
        @test Matrix(Lc) ≈ [1.0 -1.0; -1.0 1.0] atol = 1e-12
        @test is_graph_laplacian(Lc)
    end

    @testset "path-6 → path-3: each coarse edge has the original edge weight" begin
        # Path with weights = [1, 1, 1, 1, 1]. Aggregate: {1,2}, {3,4}, {5,6}.
        L = path_laplacian(6)
        agg = [1, 1, 2, 2, 3, 3]
        P, _, _ = piecewise_constant_interpolation(agg)
        Lc = sparse(galerkin_coarse_operator(L, P))
        @test Matrix(Lc) ≈ [1.0 -1.0  0.0;
                           -1.0  2.0 -1.0;
                            0.0 -1.0  1.0] atol = 1e-12
    end

    @testset "non-uniform weights survive the Galerkin sum" begin
        # 4-node path with edge weights [3, 5, 7]. Aggregate {1,2} and {3,4}.
        W = sparse([0.0  3.0  0.0  0.0;
                    3.0  0.0  5.0  0.0;
                    0.0  5.0  0.0  7.0;
                    0.0  0.0  7.0  0.0])
        L = laplacian(W)
        agg = [1, 1, 2, 2]
        P, _, _ = piecewise_constant_interpolation(agg)
        Lc = sparse(galerkin_coarse_operator(L, P))
        # Coarse edge (1, 2) weight = 5 (the original 2-3 edge).
        # Coarse degree row 1 = 5, row 2 = 5.
        @test Matrix(Lc) ≈ [5.0 -5.0; -5.0 5.0] atol = 1e-12
    end

    @testset "random graph: PᵀAP matches dense computation" begin
        rng = MersenneTwister(0xa1b2)
        L = random_graph_laplacian(20, 0.3; rng = rng)
        # Greedy random aggregation: pair nodes 1+2, 3+4, ...
        agg = [(i + 1) ÷ 2 for i in 1:20]
        P, _, _ = piecewise_constant_interpolation(agg)
        Lc_sparse = sparse(galerkin_coarse_operator(L, P))
        Lc_dense = Matrix(P)' * Matrix(L) * Matrix(P)
        @test Matrix(Lc_sparse) ≈ Lc_dense atol = 1e-12
        @test is_graph_laplacian(Lc_sparse)
    end

    @testset "structural preservation: zero row sum + nonneg off-diagonals" begin
        # Sweep a handful of random graphs and aggregations.
        rng = MersenneTwister(0xc0c0a)
        for _ in 1:5
            n = rand(rng, 10:40)
            p = rand(rng, [0.2, 0.3, 0.5])
            L = random_graph_laplacian(n, p; rng = rng)
            agg = aggregate(L; rng = rng)
            P, _, _ = piecewise_constant_interpolation(agg.aggregate)
            Lc = sparse(galerkin_coarse_operator(L, P))
            @test is_graph_laplacian(Lc)
        end
    end
end
