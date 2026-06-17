using Test
using LinearAlgebra
using Random
using LAMG

@testset "shrinkage" begin
    @testset "shrinkage_factor returns a ShrinkageResult" begin
        rng = MersenneTwister(0xface)
        n = 32
        L = path_laplacian(n)
        rx = GaussSeidelRelaxer(L)
        op = x -> L * x
        meth! = (x, b) -> relax!(rx, x, b; sweeps = 1)
        res = shrinkage_factor(op, meth!, n;
                               num_examples = 3, max_sweeps = 20, rng = rng)
        @test res isa ShrinkageResult
        @test 0.0 < res.factor < 1.0
        @test res.podr_sweeps >= 1
        @test size(res.residual_history, 1) == res.podr_sweeps + 1 ||
              size(res.residual_history, 1) >= 2
    end

    @testset "GS and Jacobi both shrink residuals on 1D Laplacian" begin
        # Note: for the 1D Laplacian, ω=2/3 Jacobi has smoothing factor 1/3,
        # beating GS's classical 1/2. So we do NOT assert GS < Jacobi here —
        # only that both are good smoothers (μ well below 1).
        n = 64
        L = path_laplacian(n)
        gs = GaussSeidelRelaxer(L)
        wj = JacobiRelaxer(L; ω = 2/3)
        op = x -> L * x
        gs_meth! = (x, b) -> relax!(gs, x, b; sweeps = 1)
        wj_meth! = (x, b) -> relax!(wj, x, b; sweeps = 1)
        rgs = shrinkage_factor(op, gs_meth!, n; num_examples = 3,
                               max_sweeps = 20, rng = MersenneTwister(0xc0de))
        rwj = shrinkage_factor(op, wj_meth!, n; num_examples = 3,
                               max_sweeps = 20, rng = MersenneTwister(0xc0de))
        @test rgs.factor < 0.8
        @test rwj.factor < 0.8
        # Asymptotic conv factor remains under 1 for both.
        @test rgs.conv_factor < 1.0
        @test rwj.conv_factor < 1.0
    end
end
