# Unit tests for FCG(1), the K-cycle, and the full DRA-QC solve.
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

# anisotropic 5-point grid: weak (ε) vertical coupling.
function aniso_grid_lap(nx, ny, ε)
    n = nx * ny; idx(i, j) = (j - 1) * nx + i
    I, J, V = Int[], Int[], Float64[]
    for j in 1:ny, i in 1:nx
        if i < nx; push!(I, idx(i,j)); push!(J, idx(i+1,j)); push!(V, -1.0); end
        if i > 1;  push!(I, idx(i,j)); push!(J, idx(i-1,j)); push!(V, -1.0); end
        if j < ny; push!(I, idx(i,j)); push!(J, idx(i,j+1)); push!(V, -ε);  end
        if j > 1;  push!(I, idx(i,j)); push!(J, idx(i,j-1)); push!(V, -ε);  end
    end
    W = sparse(I, J, V, n, n); sparse(Diagonal(-vec(sum(W; dims=2)))) + W
end

zmrand(n, seed) = (Random.seed!(seed); v = randn(n); v .- sum(v)/n)

@testset "FCG(1) + K-cycle + solve" begin

    @testset "FCG(1) on dense SPD equals direct solve (M = I)" begin
        rng = MersenneTwister(1)
        B = randn(rng, 12, 12); A = B'B + 12I        # SPD
        b = randn(rng, 12)
        x, iters, relres = DRAQC.fcg1(A, b, identity; tol=1e-12, maxiter=200)
        @test relres < 1e-10
        @test x ≈ A \ b rtol=1e-8
        @test iters <= 12                            # CG converges in ≤ n steps
    end

    @testset "Poisson grids converge to 1e-8" begin
        for (nx, ny, maxit) in ((32, 32, 60), (48, 48, 60), (64, 64, 70))
            A = grid2d_lap(nx, ny)
            b = zmrand(size(A,1), 42)
            h = DRAQC.draqc_setup(A; maxcoarse=80)
            x, info = DRAQC.draqc_solve(DRAQC.DRAQCSolver(h), b; tol=1e-8, maxiter=maxit)
            @test info.relres <= 1e-8
            @test norm(A * x - b) / norm(b) <= 1e-7
            @test info.iters <= maxit
        end
    end

    @testset "random graph Laplacian converges" begin
        rng = MersenneTwister(9); n = 800; W = spzeros(n, n)
        for i in 1:n-1
            w = rand(rng) + 0.2; W[i, i+1] = w; W[i+1, i] = w
        end
        for i in 1:n, j in i+1:n
            if rand(rng) < 8/n
                w = rand(rng) + 0.2; W[i, j] = w; W[j, i] = w
            end
        end
        A = sparse(Diagonal(vec(sum(W; dims=2)))) - W
        b = zmrand(n, 7)
        x, info = DRAQC.draqc(A, b; tol=1e-8)
        @test info.relres <= 1e-8
    end

    @testset "anisotropic grid converges (robustness)" begin
        A = aniso_grid_lap(48, 48, 1e-3)
        b = zmrand(size(A,1), 3)
        x, info = DRAQC.draqc(A, b; tol=1e-8)
        @test info.relres <= 1e-8
    end
end
