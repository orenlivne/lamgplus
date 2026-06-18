# DRAQCHybrid test suite — run: julia --project=. draqc_hybrid/test/runtests.jl
using Test, SparseArrays, LinearAlgebra, Random
include(joinpath(@__DIR__, "..", "src", "DRAQCHybrid.jl"))
using .DRAQCHybrid
const H = DRAQCHybrid

# --- graph builders ---
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
function aniso2d_lap(nx, ny, ε)
    idx(i, j) = (j - 1) * nx + i; I, J, V = Int[], Int[], Float64[]
    for j in 1:ny, i in 1:nx
        i < nx && (push!(I, idx(i,j)); push!(J, idx(i+1,j)); push!(V, -1.0))
        i > 1  && (push!(I, idx(i,j)); push!(J, idx(i-1,j)); push!(V, -1.0))
        j < ny && (push!(I, idx(i,j)); push!(J, idx(i,j+1)); push!(V, -ε))
        j > 1  && (push!(I, idx(i,j)); push!(J, idx(i,j-1)); push!(V, -ε))
    end
    W = sparse(I, J, V, nx*ny, nx*ny); sparse(Diagonal(-vec(sum(W; dims=2)))) + W
end
is_lap(A) = maximum(abs.(A * ones(size(A,1)))) < 1e-9 && issymmetric(A) &&
            all(v -> v <= 1e-12, [A[i,j] for (i,j) in zip(findnz(A)[1:2]...) if i != j])
zmrand(n, s) = (Random.seed!(s); v = randn(n); v .- sum(v)/n)

@testset "DRAQCHybrid" begin

    @testset "max_incident" begin
        # weights: edge(1,2)=3, edge(2,3)=1  ->  mw=[3,3,1]
        W = sparse([1,2,2,3],[2,1,3,2],[3.0,3,1,1],3,3)
        A = sparse(Diagonal(vec(sum(W;dims=2)))) - W
        @test H.max_incident(A) ≈ [3.0, 3.0, 1.0]
    end

    @testset "SoC veto: aggregates align with the strong direction" begin
        # On an anisotropic grid (strong = x, weak = ε·y), SoC aggregates must NOT
        # span the weak (y) direction — every aggregate sits in a single grid row.
        nx, ny, ε = 16, 16, 1e-4
        A = aniso2d_lap(nx, ny, ε)
        agg, nc = H.dra_aggregate_soc(A; τ = 0.05)
        rowof(v) = (v - 1) ÷ nx + 1
        for a in 1:nc
            members = findall(==(a), agg)
            @test length(unique(rowof.(members))) == 1     # one grid row ⇒ strong-direction chain
        end
        @test all(agg .>= 1)                               # full coverage
        # contrast: a plain (no-veto) DRA aggregate spans both directions on the same grid
        agg0, nc0 = H.DRAQC.dra_aggregate(A)
        spans = maximum(a -> length(unique(rowof.(findall(==(a), agg0)))), 1:nc0)
        @test spans >= 2                                   # plain DRA does span the weak direction
    end

    @testset "hybrid_partition covers all vertices; isotropic unaffected" begin
        for A in (grid2d_lap(20, 20), aniso2d_lap(20, 20, 1e-3))
            agg, nc = H.hybrid_partition(A)
            @test all(agg .>= 1)
            @test sort(unique(agg)) == collect(1:nc)
            @test nc < size(A, 1)
        end
    end

    @testset "hybrid hierarchy is a valid Laplacian stack" begin
        A = aniso2d_lap(48, 48, 1e-4)
        h = H.hybrid_setup(A; maxcoarse = 50)
        @test DRAQCHybrid.DRAQC.num_levels(h) >= 2
        for Aℓ in h.A; @test is_lap(Aℓ); end
    end

    @testset "TARGET: hybrid fixes anisotropy where DRA-QC stalls" begin
        # DRA-QC needs ~600 iters at ε=1e-4; the SoC graft must converge far faster.
        for ε in (1e-2, 1e-4)
            A = aniso2d_lap(128, 128, ε)
            b = zmrand(size(A,1), 1)
            xh, ih = H.hybrid(A, b; tol = 1e-8, maxcoarse = 80)
            @test ih.relres <= 1e-8
            @test ih.iters <= 60                            # vs DRA-QC's 90 (1e-2) / 604 (1e-4)
        end
    end

    @testset "low-degree elimination (deg ≤ d_max)" begin
        A = grid2d_lap(12, 12); n = size(A, 1)
        ed, Lc = H.eliminate_lowdeg(A; dmax = 4)
        @test isempty(intersect(Set(ed.F), Set(ed.C)))      # F, C disjoint
        @test sort(vcat(ed.F, ed.C)) == collect(1:n)        # partition
        # F is an independent set ⇒ L_FF strictly diagonal
        LFF = A[ed.F, ed.F]
        @test nnz(LFF - Diagonal(diag(LFF))) == 0
        @test is_lap(Lc)                                     # Schur complement is a Laplacian
        # dmax=4 eliminates strictly more than dmax=1 on a grid
        ed1, _ = H.eliminate_lowdeg(A; dmax = 1)
        @test length(ed.F) > length(ed1.F)

        # EXACTNESS: eliminate + exact coarse solve + back-substitute == direct solve
        b = zmrand(n, 5)
        bF = b[ed.F]; bc = b[ed.C] - ed.LFC' * (bF ./ ed.dff)
        φC = pinv(Matrix(Lc)) * bc
        φ = zeros(n); φ[ed.F] = H.backsub(ed, bF, φC); φ[ed.C] = φC; φ .-= sum(φ)/n
        φdir = pinv(Matrix(A)) * b; φdir .-= sum(φdir)/n
        @test φ ≈ φdir rtol = 1e-8
    end

    @testset "hybrid_elim (deg≤4) solves correctly" begin
        for A in (grid2d_lap(48, 48), aniso2d_lap(64, 64, 1e-4))
            b = zmrand(size(A,1), 4)
            φ, info, sz = H.hybrid_elim(A, b; dmax = 4, tol = 1e-8)
            @test norm(A * φ - b) / norm(b) <= 1e-7
            @test sz.nF > 0
        end
    end

    @testset "TARGET: deg≤4 elimination rescues high-contrast" begin
        # 7-decade random weights; SoC veto alone over-fragments and stalls, but
        # exact degree-≤4 elimination + SoC hybrid converges.
        rng = MersenneTwister(1); nx = ny = 96; idx(i,j) = (j-1)*nx + i
        I, J, V = Int[], Int[], Float64[]
        ae(a,b) = (w = 10.0^(7*rand(rng) - 3.5); push!(I,a);push!(J,b);push!(V,-w); push!(I,b);push!(J,a);push!(V,-w))
        for j in 1:ny, i in 1:nx; i<nx && ae(idx(i,j),idx(i+1,j)); j<ny && ae(idx(i,j),idx(i,j+1)); end
        A = sparse(I,J,V,nx*ny,nx*ny); A = sparse(Diagonal(-vec(sum(A;dims=2)))) + A
        b = zmrand(size(A,1), 6)
        φ, info, sz = H.hybrid_elim(A, b; dmax = 4, tol = 1e-8, maxiter = 400)
        @test info.relres <= 1e-8
        @test norm(A * φ - b) / norm(b) <= 1e-7
    end

    @testset "caliber-2 prolongation: constant-preserving, lean, valid" begin
        A = aniso2d_lap(24, 24, 1e-4)
        agg, nc = H.hybrid_partition(A)
        Ac, P = H.caliber2_prolongation(A, agg, nc)
        @test size(P) == (size(A,1), nc)
        @test all(sum(P; dims=2) .≈ 1)                    # rows sum to 1 (constants exact)
        @test all(v -> v <= 2, vec(sum(P .!= 0; dims=2))) # ≤ 2 parents per fine node (lean)
        @test is_lap(Ac)
    end

    @testset "caliber-2 lowers iterations vs caliber-1 on anisotropy" begin
        A = aniso2d_lap(128, 128, 1e-4); b = zmrand(size(A,1), 1)
        _, i1 = H.hybrid(A, b; tol = 1e-8, caliber2 = false)
        _, i2 = H.hybrid(A, b; tol = 1e-8, caliber2 = true)
        @test i2.relres <= 1e-8
        @test i2.iters <= i1.iters                        # caliber-2 no worse, expected fewer
    end

    @testset "no regression on isotropic / structured grids" begin
        for (nx, ny) in ((48, 48), (64, 64))
            A = grid2d_lap(nx, ny); b = zmrand(size(A,1), 2)
            xh, ih = H.hybrid(A, b; tol = 1e-8, maxcoarse = 80)
            @test ih.relres <= 1e-8
            @test ih.iters <= 60
        end
    end
end
