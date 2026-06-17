using Test
using LinearAlgebra
using SparseArrays
using Random
using Statistics
using LAMG

# Ported / inspired by:
# - amgplus Python `test/solve/test_mock_cycle.py`
# - `amgplus.tex` §Mock Cycle convergence-factor table
#
# The mock cycle predicts the multilevel convergence factor *before* an
# interpolation operator is designed. It runs `num_steps` smoother sweeps
# + an idealized coarse-level correction (project out the coarse variables)
# and measures the per-cycle residual reduction. A small mock-cycle factor
# is necessary (though not sufficient) for good two-level convergence.

@testset "mock cycle — convergence factor table" begin
    """Run `cycles` mock cycles and return the geometric mean per-cycle factor."""
    function mock_cycle_factor(L::SparseMatrixCSC, R::SparseMatrixCSC;
                               num_steps::Int = 1,
                               cycles::Int = 8,
                               rng = MersenneTwister(0xface))
        rx = GaussSeidelRelaxer(L)
        proc = MockCycleProcessor(rx, R; num_steps = num_steps,
                                  num_corrector_steps = 1)
        n = size(L, 1)
        x = randn(rng, n); x .-= sum(x) / n
        norms = Float64[]
        push!(norms, norm(L * x))
        for _ in 1:cycles
            cyc = Cycle(proc, 1.0, 2)
            run_cycle!(cyc, x)
            x = LAMG.result(proc, 1)
            x .-= sum(x) / n
            push!(norms, norm(L * x))
        end
        # Geometric mean of last few factors.
        ratios = norms[2:end] ./ norms[1:end - 1]
        tail = ratios[max(end - 3, 1):end]
        return exp(mean(log.(max.(tail, 1e-30))))
    end

    @testset "1D path Laplacian — pointwise vs averaging coarsening" begin
        n = 32
        L = path_laplacian(n)
        # Pairing coarsening: aggregate {2k-1, 2k} → k.
        agg = [(i + 1) ÷ 2 for i in 1:n]
        # Caliber-1 averaging coarsening: R is the (1/2)-averaging operator.
        _, R_avg, _ = piecewise_constant_interpolation(agg)
        # Mock factor with averaging coarsening should be well below 1.
        f_avg_1 = mock_cycle_factor(L, R_avg; num_steps = 1)
        f_avg_3 = mock_cycle_factor(L, R_avg; num_steps = 3)
        # More relaxation = smaller mock factor.
        @test f_avg_1 < 0.9
        @test f_avg_3 < f_avg_1
        @test f_avg_3 < 0.5
    end

    @testset "more relaxation sweeps monotonically improve the mock factor" begin
        L = path_laplacian(32)
        agg = [(i + 1) ÷ 2 for i in 1:32]
        _, R, _ = piecewise_constant_interpolation(agg)
        factors = [mock_cycle_factor(L, R; num_steps = ν) for ν in 1:4]
        # Each successive ν should make the mock factor at most ~ as large.
        for k in 2:4
            @test factors[k] <= factors[k - 1] + 1e-3
        end
    end

    @testset "2D grid: mock factor stays bounded across sizes" begin
        # A 2D grid with pair-aggregation should yield a bounded mock factor
        # independent of n (the LAMG signature of optimal AMG).
        for k in (8, 16, 32)
            L = grid2d_laplacian(k, k)
            n = k * k
            # Simple pair-aggregation by node ordering.
            agg = [(i + 1) ÷ 2 for i in 1:n]
            _, R, _ = piecewise_constant_interpolation(agg)
            f = mock_cycle_factor(L, R; num_steps = 2,
                                  rng = MersenneTwister(0x111 + k))
            # Should be bounded well below 1 across all sizes.
            @test f < 0.8
        end
    end

    @testset "coarse projection: x is in null(R) after correction" begin
        # The corrector step exactly zeros the coarse-variable values.
        # After one corrector step: R * x = 0.
        n = 16
        L = path_laplacian(n)
        rx = GaussSeidelRelaxer(L)
        agg = [(i + 1) ÷ 2 for i in 1:n]
        _, R, _ = piecewise_constant_interpolation(agg)
        proc = MockCycleProcessor(rx, R; num_steps = 0, num_corrector_steps = 1)
        rng = MersenneTwister(0xc0)
        x = randn(rng, n)
        LAMG.initialize!(proc, 1, 2, x)
        LAMG.process_coarsest!(proc, 2)
        @test maximum(abs, R * LAMG.result(proc, 1)) < 1e-10
    end
end
