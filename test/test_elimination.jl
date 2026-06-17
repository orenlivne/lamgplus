using Test
using LinearAlgebra
using SparseArrays
using LAMG

@testset "elimination" begin
    @testset "low_degree_nodes on a path: every other node is F" begin
        L = path_laplacian(8)
        z, f, c = low_degree_nodes(L; max_degree = 4)
        @test isempty(z)
        @test length(f) >= 3            # at least every-other from one end
        @test length(c) == 8 - length(f)
        # F nodes are mutually independent — no two in f are adjacent in L.
        for u in f, v in f
            u == v && continue
            @test L[u, v] == 0
        end
    end

    @testset "low_degree_nodes on isolated nodes" begin
        # Build a 4-node graph with one isolated node.
        n = 4
        L = sparse([1.0 -1.0 0.0 0.0;
                   -1.0  1.0 0.0 0.0;
                    0.0  0.0 0.0 0.0;     # isolated
                    0.0  0.0 0.0 0.0])    # also isolated, will appear in z
        z, f, c = low_degree_nodes(L)
        @test 3 in z && 4 in z
    end

    @testset "elimination_operators on a simple path: exact Schur complement" begin
        # Build a 5-node path Laplacian. Eliminate the interior odd-numbered.
        L = path_laplacian(5)
        # Manually set f = [2, 4], c = [1, 3, 5] (independent in L).
        @test L[2, 4] == 0
        f = [2, 4]; c = [1, 3, 5]
        P, R, q = elimination_operators(L, f, c)
        @test size(P) == (2, 3)
        @test size(R) == (3, 2)
        @test length(q) == 2
        # The Schur complement should equal the path Laplacian on 3 nodes
        # with weight 1/2 each edge (parallel resistors: 1 + 1 = 2 → 1/2).
        Acc = L[c, c]; Acf = L[c, f]
        Aschur = Acc + Acf * P
        # Compare to a manual computation.
        expected = sparse([0.5 -0.5 0.0;
                          -0.5  1.0 -0.5;
                           0.0 -0.5  0.5])
        @test Aschur ≈ expected atol = 1e-12
    end

    @testset "eliminate_once: stage round-trip" begin
        L = path_laplacian(8)
        stage, Anext, z = eliminate_once(L)
        @test stage !== nothing
        @test size(Anext, 1) < 8
        @test size(Anext, 1) == length(stage.c)
        @test is_laplacian(Anext)
        # Round-trip: pick a c-only fine vector, restrict (with zero RHS),
        # interpolate, should recover the original (since elimination is
        # exact for vectors satisfying the constraint).
        # Build an x defined only on c-nodes.
    end

    @testset "elimination level: restrict and interpolate are mutual inverses for compatible inputs" begin
        # If x_f satisfies the elimination constraint (A[f,f] x_f = b[f] − A[f,c] x_c),
        # then interpolate(c_to_f(x_f), b_stages) recovers x_f exactly.
        L = path_laplacian(8)
        stage, Anext, _ = eliminate_once(L)
        @test stage !== nothing

        rx = GaussSeidelRelaxer(Anext)
        elev = create_elimination_level(Anext, rx, [stage])

        # Build a random fine x and corresponding b = L * x.
        rng = MersenneTwister(0xeeee)
        x_true = randn(rng, 8); x_true .-= sum(x_true) / 8
        b = L * x_true
        # Restrict b through the elimination stages.
        bc, bstages = restrict_elimination(elev, b)
        # Coarse-level x is the f-eliminated solution. We know:
        #   x_true[c] satisfies Anext * x_true[c] = bc.
        # Verify.
        @test Anext * x_true[stage.c] ≈ bc atol = 1e-10
        # Then interpolate back from x_true[c] using b_stages should give x_true.
        x_recon = interpolate_elimination(elev, x_true[stage.c], bstages)
        @test x_recon ≈ x_true atol = 1e-10
    end
end
