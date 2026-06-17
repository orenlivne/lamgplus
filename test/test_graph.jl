using Test
using LinearAlgebra
using SparseArrays
using LAMG

@testset "graph" begin
    @testset "path_laplacian" begin
        L = path_laplacian(5)
        @test is_laplacian(L)
        @test is_graph_laplacian(L)
        @test size(L) == (5, 5)
        @test L[1, 1] == 1.0
        @test L[3, 3] == 2.0
        @test L[1, 2] == -1.0
        u = null_vector(L)
        @test norm(L * u) < 1e-12
    end

    @testset "grid2d_laplacian" begin
        L = grid2d_laplacian(3, 3)
        @test is_laplacian(L)
        @test is_graph_laplacian(L)
        @test size(L) == (9, 9)
        @test L[1, 1] == 2.0
        @test L[2, 2] == 3.0
        @test L[5, 5] == 4.0
    end

    @testset "is_laplacian vs is_graph_laplacian" begin
        # A symmetric, zero-row-sum matrix with a positive off-diagonal:
        # NOT a graph Laplacian, but IS a "Laplacian-like" SPSD operator
        # (the kind LAMG admits — cf. Fig. 1 of Livne-Brandt 2012).
        L = sparse([2.4 -1.0 0.1 -1.0 -0.5;
                    -1.0  2.0  0.0 -1.0  0.0;
                    0.1  0.0  1.9 -2.0  0.0;
                   -1.0 -1.0 -2.0  4.0  0.0;
                   -0.5  0.0  0.0  0.0  0.5])
        @test is_laplacian(L)
        @test !is_graph_laplacian(L)   # the +0.1 off-diagonal disqualifies it
    end

    @testset "rejects bad inputs" begin
        A = sparse([1.0 -1.0; 0.0 1.0])
        @test !is_laplacian(A)
        B = sparse([1.0 -1.0; -1.0 2.0])
        @test !is_laplacian(B)
    end

    @testset "connected_components labels each node" begin
        # 5-node graph: edges (1,2), (3,4). Two CCs: {1,2}, {3,4}, plus 5 isolated.
        W = sparse([0.0 1.0 0.0 0.0 0.0;
                    1.0 0.0 0.0 0.0 0.0;
                    0.0 0.0 0.0 1.0 0.0;
                    0.0 0.0 1.0 0.0 0.0;
                    0.0 0.0 0.0 0.0 0.0])
        L = laplacian(W)
        cc = connected_components(L)
        @test cc[1] == cc[2]
        @test cc[3] == cc[4]
        @test cc[1] != cc[3]
        @test cc[5] != cc[1] && cc[5] != cc[3]
        @test maximum(cc) == 3
    end

    @testset "largest_component on a connected graph is the identity" begin
        L = path_laplacian(8)
        Lc, retained = largest_component(L)
        @test size(Lc) == size(L)
        @test retained == collect(1:8)
    end

    @testset "largest_component on a disconnected graph keeps the giant CC" begin
        # Two paths of size 5 and 3. Largest is the 5-node path.
        W_big = sparse([0.0 1.0 0.0 0.0 0.0;
                        1.0 0.0 1.0 0.0 0.0;
                        0.0 1.0 0.0 1.0 0.0;
                        0.0 0.0 1.0 0.0 1.0;
                        0.0 0.0 0.0 1.0 0.0])
        W_small = sparse([0.0 1.0 0.0;
                          1.0 0.0 1.0;
                          0.0 1.0 0.0])
        n_big, n_small = 5, 3
        nz = n_big + n_small
        # Block-diagonal combine.
        W = sparse(spzeros(nz, nz))
        W[1:n_big, 1:n_big] = W_big
        W[n_big + 1:end, n_big + 1:end] = W_small
        L = laplacian(W)
        Lc, retained = largest_component(L)
        @test size(Lc) == (n_big, n_big)
        @test is_graph_laplacian(Lc)
        @test retained == collect(1:n_big)
    end

    @testset "random_graph_laplacian" begin
        using Random
        rng = MersenneTwister(0xb0ba)
        L = random_graph_laplacian(20, 0.3; rng = rng)
        @test is_graph_laplacian(L)
        @test size(L) == (20, 20)
        u = null_vector(L)
        @test norm(L * u) < 1e-12
    end
end
