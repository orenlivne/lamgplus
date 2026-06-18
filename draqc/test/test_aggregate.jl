# Unit tests for DRA aggregation (Algorithm 2).
using Test, SparseArrays, LinearAlgebra

# 5-point 2D grid Laplacian (unweighted), nx×ny.
function grid2d_lap(nx, ny)
    n = nx * ny
    idx(i, j) = (j - 1) * nx + i
    I, J, V = Int[], Int[], Float64[]
    for j in 1:ny, i in 1:nx
        u = idx(i, j)
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ii, jj = i + di, j + dj
            if 1 <= ii <= nx && 1 <= jj <= ny
                push!(I, u); push!(J, idx(ii, jj)); push!(V, -1.0)
            end
        end
    end
    W = sparse(I, J, V, n, n)
    return sparse(Diagonal(-vec(sum(W; dims=2)))) + W
end

star_lap(k) = (W = spzeros(k + 1, k + 1);
    for j in 2:k+1; W[1, j] = 1.0; W[j, 1] = 1.0; end;
    sparse(Diagonal(vec(sum(W; dims=2)))) - W)

@testset "DRA aggregation" begin
    @testset "partition validity" begin
        for A in (grid2d_lap(8, 8), star_lap(20), grid2d_lap(12, 5))
            agg, nc = DRAQC.dra_aggregate(A)
            @test all(agg .>= 1)                 # every vertex aggregated
            @test maximum(agg) == nc
            @test sort(unique(agg)) == collect(1:nc)
            @test nc < size(A, 1)                # genuine coarsening
        end
    end

    @testset "star collapses to one aggregate" begin
        # The hub has the highest degree ⇒ chosen first as root ⇒ absorbs all leaves.
        agg, nc = DRAQC.dra_aggregate(star_lap(50))
        @test nc == 1
    end

    @testset "grid coarsening ratio ≥ 3 (aggregates big enough)" begin
        A = grid2d_lap(20, 20)
        _, nc = DRAQC.dra_aggregate(A)
        @test size(A, 1) / nc >= 3.0
    end

    @testset "active mask restricts aggregation" begin
        A = grid2d_lap(6, 6)
        active = trues(36); active[1:12] .= false
        agg, nc = DRAQC.dra_aggregate(A; active = active)
        @test all(agg[1:12] .== 0)               # inactive untouched
        @test all(agg[13:36] .>= 1)              # active all aggregated
        @test sort(unique(agg[13:36])) == collect(1:nc)
    end
end
