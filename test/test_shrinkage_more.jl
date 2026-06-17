using Test
using LinearAlgebra
using SparseArrays
using Random
using LAMG

# Edge-case tests for the shrinkage factor + PODR detection.
# Inspired by amgplus `test/solve/test_smoothing.py`.

@testset "shrinkage — edge cases" begin
    @testset "shrinkage on exact solver: factor → 0 quickly" begin
        # An "exact" solver replaces x with x - A^{-1} r (one Newton step on
        # the LS error). Should drive r to zero in one step → factor near 0.
        n = 32
        A = sparse(path_laplacian(n) + 0.1 * I)   # regularize for invertibility
        op = x -> A * x
        # A "method" that solves exactly in one step.
        F = lu(Matrix(A))
        exact_step! = function(x, b)
            r = b .- A * x
            δ = F \ r
            x .+= δ
        end
        res = shrinkage_factor(op, exact_step!, n; num_examples = 2,
                               max_sweeps = 5, rng = MersenneTwister(0xeaa1))
        @test res.factor < 1e-3
        @test res.podr_sweeps == 1
    end

    @testset "shrinkage on identity (no-op) method: factor ≈ 1" begin
        # A method that does nothing should give factor ≈ 1 (no reduction).
        n = 16
        A = sparse(path_laplacian(n) + 0.1 * I)
        noop! = function(x, b)
            return x
        end
        res = shrinkage_factor(x -> A * x, noop!, n; num_examples = 2,
                               max_sweeps = 3, rng = MersenneTwister(0xabcd))
        @test 0.95 <= res.factor <= 1.05
    end

    @testset "shrinkage handles random initial vectors" begin
        # Verify that multiple random starts give similar (but not identical)
        # shrinkage estimates.
        n = 32
        L = path_laplacian(n)
        rx = GaussSeidelRelaxer(L)
        op = x -> L * x
        meth! = (x, b) -> relax!(rx, x, b; sweeps = 1)
        factors = Float64[]
        for seed in (0x1, 0x2, 0x3, 0x4)
            res = shrinkage_factor(op, meth!, n; num_examples = 4,
                                   max_sweeps = 15,
                                   rng = MersenneTwister(seed))
            push!(factors, res.factor)
        end
        # Different seeds should give similar factors (within 2x).
        @test maximum(factors) / minimum(factors) < 2.5
    end

    @testset "shrinkage early-stops when conv_factor exceeds slow_conv_factor" begin
        # If the method diverges, shrinkage should stop early.
        n = 8
        A = sparse(path_laplacian(n) + 0.1 * I)
        # Build a diverging method by over-stepping.
        diverger! = function(x, b)
            r = b .- A * x
            x .+= 10.0 .* r        # huge overshoot
        end
        res = shrinkage_factor(x -> A * x, diverger!, n;
                               num_examples = 1, max_sweeps = 100,
                               slow_conv_factor = 1.3,
                               rng = MersenneTwister(0xd1ff))
        # Should stop before max_sweeps because conv_factor > 1.3.
        @test size(res.residual_history, 1) < 50
    end

    @testset "GS shrinkage factor for 2D Laplacian (n=64): theoretical ≈ 0.5" begin
        # For the 2D Laplacian, classical GS smoothing factor is ≈ 0.5.
        # The shrinkage at PODR should be a bit lower since it includes
        # initial transient progress.
        L = grid2d_laplacian(8, 8)
        n = size(L, 1)
        rx = GaussSeidelRelaxer(L)
        res = shrinkage_factor(x -> L * x,
                              (x, b) -> relax!(rx, x, b; sweeps = 1),
                              n; num_examples = 3, max_sweeps = 15,
                              rng = MersenneTwister(0x2d2d))
        # Asymptotic factor should be in the GS ballpark.
        @test 0.3 <= res.factor <= 0.95
        @test 0.3 <= res.conv_factor <= 0.999
    end
end
