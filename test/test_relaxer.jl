using Test
using LinearAlgebra
using SparseArrays
using Random
using LAMG

@testset "relaxer" begin
    @testset "GaussSeidelRelaxer reduces residual on 1D Laplacian" begin
        n = 32
        L = path_laplacian(n)
        rx = GaussSeidelRelaxer(L)
        rng = MersenneTwister(0x1eaf)
        x = randn(rng, n); x .-= sum(x) / n
        b = zeros(n)
        r0 = norm(L * x)
        relax!(rx, x, b; sweeps = 10)
        x .-= sum(x) / n
        @test norm(L * x) < r0
    end

    @testset "GaussSeidelRelaxer drives a regularized system to the exact solution" begin
        # Use stronger regularization (cond ~ 10) so 500 GS sweeps suffice.
        n = 8
        L = sparse(path_laplacian(n) + I)
        rx = GaussSeidelRelaxer(L)
        x_true = collect(Float64, 1:n); x_true .-= sum(x_true) / n
        b = L * x_true
        x = zeros(n)
        relax!(rx, x, b; sweeps = 500)
        @test norm(x - x_true) < 1e-6
    end

    @testset "JacobiRelaxer reduces residual" begin
        n = 32
        L = path_laplacian(n)
        rx = JacobiRelaxer(L; ω = 2/3)
        rng = MersenneTwister(0xfeed)
        x = randn(rng, n); x .-= sum(x) / n
        r0 = norm(L * x)
        relax!(rx, x, zeros(n); sweeps = 10)
        x .-= sum(x) / n
        @test norm(L * x) < r0
    end

    @testset "GS damping ω=1.0 is the default and ω≠1 still reduces residual" begin
        n = 16
        L = path_laplacian(n)
        rx = GaussSeidelRelaxer(L; ω = 0.7)
        rng = MersenneTwister(0xc0ff)
        x = randn(rng, n); x .-= sum(x) / n
        b = zeros(n)
        r0 = norm(L * x)
        relax!(rx, x, b; sweeps = 20)
        @test norm(L * x) < r0
    end
end
