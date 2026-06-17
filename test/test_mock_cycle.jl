using Test
using LinearAlgebra
using SparseArrays
using Random
using LAMG

@testset "mock cycle" begin
    @testset "MockCycleProcessor projects out coarse-variable component" begin
        # Use a tiny problem where we can verify the projection by hand.
        n = 8
        L = path_laplacian(n)
        rx = GaussSeidelRelaxer(L)
        agg = [1, 1, 2, 2, 3, 3, 4, 4]
        _, R, _ = piecewise_constant_interpolation(agg)
        # Use the coarsening matrix R (n_c × n) as the projector Q.
        proc = MockCycleProcessor(rx, R; num_steps = 0, num_corrector_steps = 1)
        x = collect(Float64, 1:n)
        LAMG.initialize!(proc, 1, 2, x)
        LAMG.process_coarsest!(proc, 2)
        # After projection, R * x should be (approximately) zero.
        xnew = LAMG.result(proc, 1)
        @test maximum(abs, R * xnew) < 1e-10
    end

    @testset "mock_cycle convenience constructor builds a 2-level Cycle" begin
        n = 8
        L = path_laplacian(n)
        rx = GaussSeidelRelaxer(L)
        agg = [1, 1, 2, 2, 3, 3, 4, 4]
        _, R, _ = piecewise_constant_interpolation(agg)
        cyc = mock_cycle(rx, R; num_steps = 1, num_corrector_steps = 1)
        rng = MersenneTwister(0x77)
        x = randn(rng, n); x .-= sum(x) / n
        r0 = norm(L * x)
        run_cycle!(cyc, x)
        xnew = LAMG.result(cyc.processor, 1)
        # The mock cycle should reduce the residual norm (it's a contraction
        # for the path Laplacian with averaging coarsening — cf. amgplus
        # mock-cycle numerical table).
        @test norm(L * xnew) < r0
    end
end
