using Test
using LinearAlgebra
using SparseArrays
using Random
using Statistics
using LAMG

@testset "setup + solve (end-to-end)" begin
    @testset "setup on a path graph builds a multilevel hierarchy" begin
        L = path_laplacian(64)
        h = setup(L)
        @test length(h) >= 2
        # Coarsest must be small.
        @test size(h[end]) <= 20
    end

    @testset "solve on a path Laplacian converges to tol" begin
        n = 64
        L = path_laplacian(n)
        x_true = sin.(2π .* (1:n) ./ n); x_true .-= sum(x_true) / n
        b = L * x_true
        opts = LAMGOptions(tol = 1e-8, max_cycles = 50)
        x, info = solve(L, b; options = opts)
        # Compare against the true solution, modulo the null component.
        x .-= sum(x) / n
        x_true .-= sum(x_true) / n
        @test info.final_residual <= 1e-8 * norm(b)
        @test info.cycles < 50
    end

    @testset "solve on a 2D grid Laplacian converges to tol" begin
        L = grid2d_laplacian(8, 8)
        n = size(L, 1)
        rng = MersenneTwister(0xa5a5)
        x_true = randn(rng, n); x_true .-= sum(x_true) / n
        b = L * x_true
        opts = LAMGOptions(tol = 1e-8, max_cycles = 100)
        x, info = solve(L, b; options = opts)
        @test info.final_residual <= 1e-8 * norm(b)
    end

    @testset "convergence factors are bounded" begin
        # The per-cycle convergence factor should be well below 1 after the
        # first few warmup cycles. Asymptotic factor should be < 0.5 (LAMG
        # gets to ~0.3 with iterate recombination + γ=1.5).
        L = path_laplacian(128)
        n = size(L, 1)
        rng = MersenneTwister(0x77)
        x_true = randn(rng, n); x_true .-= sum(x_true) / n
        b = L * x_true
        opts = LAMGOptions(tol = 1e-10, max_cycles = 50)
        _, info = solve(L, b; options = opts)
        # Average per-cycle factor over the last few cycles.
        tail_factors = info.conv_factors[max(end - 4, 1):end]
        asymptotic = exp(mean(log.(max.(tail_factors, 1e-30))))
        @test asymptotic < 0.6
    end

    @testset "hierarchy can be reused across multiple RHSs" begin
        L = path_laplacian(64)
        h = setup(L)
        rng = MersenneTwister(0xbeef)
        b1 = randn(rng, 64); b1 .-= sum(b1) / 64
        b2 = randn(rng, 64); b2 .-= sum(b2) / 64
        x1, info1 = solve(h, b1)
        x2, info2 = solve(h, b2)
        @test info1.final_residual <= 1e-8 * norm(b1)
        @test info2.final_residual <= 1e-8 * norm(b2)
    end
end
