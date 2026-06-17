using Test
using LinearAlgebra
using SparseArrays
using Random
using LAMG

# Edge-case tests ported / inspired by `UTestElimination.m` and
# `UTestEliminationOperatorsMex.m` from the MATLAB LAMG suite.
#
# The key property we exercise: elimination is *exact* — when the F-set
# is correctly identified and Schur-eliminated, no residual is introduced.

@testset "elimination — edge cases" begin
    @testset "isolated nodes are detected as z (zero-degree)" begin
        # 5-node Laplacian with nodes 3 and 5 isolated.
        L = sparse([1.0 -1.0  0.0  0.0 0.0;
                   -1.0  1.0  0.0  0.0 0.0;
                    0.0  0.0  0.0  0.0 0.0;
                    0.0  0.0  0.0  0.0 0.0;
                    0.0  0.0  0.0  0.0 0.0])
        # Make L valid even though some rows are all zero.
        z, f, c = low_degree_nodes(L)
        @test 3 ∈ z && 4 ∈ z && 5 ∈ z
        # Non-isolated nodes 1, 2 are degree-1 each; they qualify for f.
        @test (1 ∈ f) || (2 ∈ f)   # one of them at least
    end

    @testset "1D grid: every other node ends up in f" begin
        for n in (8, 16, 33, 64)
            L = path_laplacian(n)
            z, f, c = low_degree_nodes(L; max_degree = 4)
            @test isempty(z)
            # On a path, |f| ≥ ⌊n/2⌋ (every other interior node + at least one boundary).
            @test length(f) >= div(n, 2) - 1
            @test length(f) + length(c) == n
            # f set must be mutually independent.
            for u in f, v in f
                u == v && continue
                @test L[u, v] == 0
            end
        end
    end

    @testset "f-set: every pair is non-adjacent (the independence invariant)" begin
        # Build a random connected graph and verify the invariant.
        rng = MersenneTwister(0xfaaf)
        L = random_graph_laplacian(40, 0.15; rng = rng)
        _, f, _ = low_degree_nodes(L)
        for u in f, v in f
            u == v && continue
            @test L[u, v] == 0
        end
    end

    @testset "elimination_operators on 5-node path: exact hand-computed Schur" begin
        # The 5-node path: nodes 1-2-3-4-5, unit weights.
        # Choose f = {2, 4}, c = {1, 3, 5} (interior odd nodes are nonadjacent).
        L = path_laplacian(5)
        f = [2, 4]; c = [1, 3, 5]
        @test L[2, 4] == 0
        P, R, q = elimination_operators(L, f, c)
        # A[f,f] = diag(2, 2), so q = [1/2, 1/2].
        @test q ≈ [0.5, 0.5]
        # P = -A[f,c] ./ diag(A[f,f]) per row.
        # Row 1 (f=2): A[2, c=1,3,5] = (-1, -1, 0). P row 1 = (.5, .5, 0).
        # Row 2 (f=4): A[4, c=1,3,5] = (0, -1, -1). P row 2 = (0, .5, .5).
        @test Matrix(P) ≈ [0.5 0.5 0.0; 0.0 0.5 0.5]
        @test Matrix(R) ≈ Matrix(P)'
        # Schur complement A[c,c] + A[c,f]·P:
        # A[c,c] = diag(1, 2, 1) for boundary/interior/boundary path-node degrees.
        # A[c,f] = [-1 0; -1 -1; 0 -1] (sym partner of A[f,c]).
        # A[c,f]·P = ([-1 0; -1 -1; 0 -1])·([.5 .5 0; 0 .5 .5])
        #         = [-0.5 -0.5 0; -0.5 -1.0 -0.5; 0 -0.5 -0.5]
        # Schur = A[c,c] + A[c,f]·P:
        #       = [1 0 0; 0 2 0; 0 0 1] + [-0.5 -0.5 0; -0.5 -1.0 -0.5; 0 -0.5 -0.5]
        #       = [0.5 -0.5 0.0; -0.5 1.0 -0.5; 0.0 -0.5 0.5]
        # which is the path-3 Laplacian with weight 1/2 on each edge (parallel resistors).
        Schur = L[c, c] + L[c, f] * P
        @test Matrix(Schur) ≈ [0.5 -0.5 0.0; -0.5 1.0 -0.5; 0.0 -0.5 0.5] atol = 1e-12
    end

    @testset "eliminate_once on a star graph collapses leaves into the hub" begin
        # A 5-pointed star: hub = 1, leaves = 2..6. All edges weight 1.
        n = 6
        rows = Int[]; cols = Int[]; vals = Float64[]
        for i in 2:n
            push!(rows, 1); push!(cols, i); push!(vals, 1.0)
            push!(rows, i); push!(cols, 1); push!(vals, 1.0)
        end
        W = sparse(rows, cols, vals, n, n)
        L = laplacian(W)
        @test is_graph_laplacian(L)
        # Leaves are degree-1 nodes; should be eliminated. Hub is degree-5,
        # not in f.
        z, f, c = low_degree_nodes(L)
        @test isempty(z)
        @test sort(f) == [2, 3, 4, 5, 6]
        @test c == [1]
        # After one stage: Anext is 1×1, and that's the "hub aggregated" node.
        stage, Anext, _ = eliminate_once(L)
        @test stage !== nothing
        @test size(Anext) == (1, 1)
    end

    @testset "elimination → solve: 1D Laplacian solves exactly via elimination" begin
        # Build a small path Laplacian, regularize, and verify the hierarchy
        # + solve produce machine-precision answers in one cycle.
        n = 64
        L = path_laplacian(n)
        rng = MersenneTwister(0x7)
        x_true = randn(rng, n); x_true .-= sum(x_true) / n
        b = L * x_true
        opts = LAMGOptions(tol = 1e-12, max_cycles = 10)
        x, info = solve(L, b; options = opts)
        @test info.final_residual <= 1e-12 * norm(b)
        @test info.cycles <= 3        # paths should solve in 1–2 cycles via elim
    end
end
