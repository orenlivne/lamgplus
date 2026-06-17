# Unit tests for weight-aware elimination (src/weight_aware_elimination.jl).
using Test, LAMG, SparseArrays, LinearAlgebra, Random

# anisotropic PxP grid Laplacian: horizontal weight wx, vertical weight wy
function _wae_grid(P; wx = 1.0, wy = 1.0)
    idx(i, j) = (j - 1) * P + i
    I = Int[]; J = Int[]; V = Float64[]
    for j in 1:P, i in 1:P
        if i < P; push!(I, idx(i, j)); push!(J, idx(i + 1, j)); push!(V, wx); end
        if j < P; push!(I, idx(i, j)); push!(J, idx(i, j + 1)); push!(V, wy); end
    end
    W = sparse(I, J, V, P * P, P * P); W = W + W'
    d = vec(sum(W, dims = 2))
    spdiagm(0 => d) - W
end

@testset "weight-aware elimination" begin

    @testset "exact Schur reconstruction (SPD)" begin
        # A x = b reconstructed exactly through one weight-aware elimination level.
        A = _wae_grid(8; wx = 1.0, wy = 1e-2) + 0.1 * I   # SPD (shifted)
        A = sparse(A)
        lvl = weight_aware_eliminate(A; τ = 0.1, dmax = 4)
        @test lvl !== nothing
        @test length(lvl.F) + length(lvl.C) == size(A, 1)
        rng = MersenneTwister(42)
        xtrue = randn(rng, size(A, 1))
        b = A * xtrue
        bC = wae_restrict(lvl, b)
        xC = Matrix(lvl.Ac) \ bC                # exact coarse solve (SPD)
        x = wae_interpolate(lvl, xC, b)
        @test x ≈ xtrue rtol = 1e-8
    end

    @testset "no-op on unweighted: identical to standard low-degree set" begin
        # equal weights ⇒ every edge is "strong" ⇒ selection == standard low_degree_nodes.
        L = _wae_grid(8; wx = 1.0, wy = 1.0)
        F, C = wae_select(L; τ = 0.1, dmax = 4)
        _, fstd, _ = low_degree_nodes(L; max_degree = 4)
        @test sort(F) == sort(fstd)
        # and the F-block is exactly diagonal (F is independent in the full graph)
        Aff = L[F, F]
        @test nnz(Aff - spdiagm(0 => diag(Aff))) == 0
    end

    @testset "strong-degree counts only strong edges" begin
        P = 6; L = _wae_grid(P; wx = 1.0, wy = 1e-4)
        u = (3 - 1) * P + 3                       # interior node (i=3, j=3)
        @test length(nzrange(L, u)) - 1 == 4      # full grid degree (4 neighbours)
        sdeg, _ = wae_strong_degree(L; τ = 0.1)
        @test sdeg[u] == 2                         # only the 2 strong (x) neighbours survive
    end

    @testset "anisotropic chains collapse far past red-black" begin
        n0 = 16 * 16
        A = sparse(_wae_grid(16; wx = 1.0, wy = 1e-4))
        levels = 0
        for _ in 1:8
            lvl = weight_aware_eliminate(A; τ = 0.1, dmax = 4)
            lvl === nothing && break
            A = lvl.Ac; levels += 1
        end
        # standard elimination stalls near n0/2; weight-aware collapses the chains
        @test size(A, 1) < 0.3 * n0
        @test levels ≥ 2
    end

    @testset "Schur complement preserves the Laplacian structure" begin
        L = _wae_grid(8; wx = 1.0, wy = 1e-2)
        lvl = weight_aware_eliminate(L; τ = 0.1, dmax = 4)
        @test lvl !== nothing
        @test norm(lvl.Ac - lvl.Ac') < 1e-10                       # symmetric
        @test maximum(abs, vec(sum(lvl.Ac, dims = 2))) < 1e-8      # zero row sums
    end

    @testset "no-op returns nothing when no node qualifies" begin
        n = 6
        W = sparse(ones(n, n) - I)                                  # complete graph
        A = spdiagm(0 => vec(sum(W, dims = 2))) - W
        @test weight_aware_eliminate(sparse(A); τ = 0.1, dmax = 2) === nothing
    end
end
