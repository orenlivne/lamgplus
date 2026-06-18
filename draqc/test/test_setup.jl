# Unit tests for the DRA-QC-CE setup: partition, Galerkin coarsening, hierarchy.
using Test, SparseArrays, LinearAlgebra, Random

function grid2d_lap(nx, ny)
    n = nx * ny; idx(i, j) = (j - 1) * nx + i
    I, J, V = Int[], Int[], Float64[]
    for j in 1:ny, i in 1:nx, (di, dj) in ((1,0),(-1,0),(0,1),(0,-1))
        ii, jj = i + di, j + dj
        if 1 <= ii <= nx && 1 <= jj <= ny
            push!(I, idx(i,j)); push!(J, idx(ii,jj)); push!(V, -1.0)
        end
    end
    W = sparse(I, J, V, n, n); sparse(Diagonal(-vec(sum(W; dims=2)))) + W
end

function rand_graph_lap(n; p=0.15, seed=1)
    rng = MersenneTwister(seed); W = spzeros(n, n)
    for i in 1:n-1
        w = rand(rng) + 0.2; W[i, i+1] = w; W[i+1, i] = w
    end
    for i in 1:n, j in i+1:n
        if rand(rng) < p
            w = rand(rng) + 0.2; W[i, j] = w; W[j, i] = w
        end
    end
    sparse(Diagonal(vec(sum(W; dims=2)))) - W
end

is_lap(A) = maximum(abs.(A * ones(size(A,1)))) < 1e-9 &&
            issymmetric(A) &&
            all(v -> v <= 1e-12, [A[i,j] for (i,j) in zip(findnz(A)[1:2]...) if i != j])

@testset "DRA-QC-CE setup" begin
    @testset "partition covers all vertices" begin
        for A in (grid2d_lap(12, 12), rand_graph_lap(200; seed=2), grid2d_lap(20, 8))
            agg, nc = DRAQC.draqc_partition(A)
            @test all(agg .>= 1)
            @test sort(unique(agg)) == collect(1:nc)
            @test nc < size(A, 1)                     # genuine coarsening
        end
    end

    @testset "Galerkin coarse operator is a graph Laplacian" begin
        A = grid2d_lap(16, 16)
        agg, nc = DRAQC.draqc_partition(A)
        Ac, P = DRAQC.galerkin(A, agg, nc)
        @test size(Ac) == (nc, nc)
        @test is_lap(Ac)                              # A_c·1=0, symmetric, nonpos offdiag
        @test size(P) == (size(A,1), nc)
        @test all(sum(P; dims=2) .== 1)              # each fine vertex in exactly one aggregate
    end

    @testset "hierarchy: valid Laplacians, bounded complexity, reaches coarsest" begin
        for A in (grid2d_lap(40, 40), rand_graph_lap(2000; seed=5))
            h = DRAQC.draqc_setup(A; maxcoarse=50)
            @test DRAQC.num_levels(h) >= 2
            @test size(h.A[end], 1) <= 50 || DRAQC.num_levels(h) == 40
            for Aℓ in h.A
                @test is_lap(Aℓ)
            end
            # coarsening ratio ≥ ~2 per level (DRA targets ≥3; allow slack)
            for ℓ in 1:length(h.P)
                @test size(h.A[ℓ], 1) / size(h.A[ℓ+1], 1) >= 2.0
            end
            # operator complexity well-bounded (paper: weighted complexity < 3)
            @test DRAQC.operator_complexity(h) < 2.5
        end
    end
end
