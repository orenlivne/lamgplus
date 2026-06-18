# Unit tests for the global δ vector (Napov–Notay eq. 15), against hand computations.
using Test, SparseArrays, LinearAlgebra

@testset "delta_vector" begin
    # Path 1-2-3 (unweighted): A = [1 -1 0; -1 2 -1; 0 -1 1].
    # δ = (U−D)D⁻¹(L−D)1, hand-computed = [0.5, 1, 0].
    A = sparse([1.0 -1 0; -1 2 -1; 0 -1 1])
    @test DRAQC.delta_vector(A) ≈ [0.5, 1.0, 0.0]

    # Single weighted edge 1=2 (weight 2): A = [2 -2; -2 2]. δ = [2, 0].
    A2 = sparse([2.0 -2; -2 2])
    @test DRAQC.delta_vector(A2) ≈ [2.0, 0.0]

    # Triangle (unweighted): δ = [1.5, 1, 0].
    A3 = sparse([2.0 -1 -1; -1 2 -1; -1 -1 2])
    @test DRAQC.delta_vector(A3) ≈ [1.5, 1.0, 0.0]

    # δ ≥ 0 for a graph Laplacian (random check).
    using Random
    rng = MersenneTwister(7)
    n = 20
    W = spzeros(n, n)
    for i in 1:n-1
        w = rand(rng) + 0.1; W[i, i+1] = w; W[i+1, i] = w
    end
    for i in 1:n, j in i+1:n
        if rand(rng) < 0.25
            w = rand(rng) + 0.1; W[i, j] = w; W[j, i] = w
        end
    end
    L = sparse(Diagonal(vec(sum(W; dims=2)))) - W
    @test all(DRAQC.delta_vector(L) .>= -1e-12)
end
