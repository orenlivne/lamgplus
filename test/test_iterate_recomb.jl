using Test
using LinearAlgebra
using LAMG

@testset "iterate recombination" begin
    @testset "IterateHistory circular buffer" begin
        h = IterateHistory(5, 3)
        @test h.capacity == 3
        @test h.n_active == 0
        @test h.latest == 0
        save_iterate!(h, [1.0, 0, 0, 0, 0], [0, 1, 0, 0, 0])
        @test h.n_active == 1
        @test h.latest == 1
        save_iterate!(h, [2.0, 0, 0, 0, 0], [0, 2, 0, 0, 0])
        save_iterate!(h, [3.0, 0, 0, 0, 0], [0, 3, 0, 0, 0])
        @test h.n_active == 3
        # Wrap-around.
        save_iterate!(h, [4.0, 0, 0, 0, 0], [0, 4, 0, 0, 0])
        @test h.n_active == 3
        @test h.latest == 1
        @test h.X[1, 1] == 4.0    # newest at position 1 after wrap
        clear_history!(h)
        @test h.n_active == 0
        @test h.latest == 0
    end

    @testset "min_res! reduces residual when iterates span the error" begin
        # Construct a tiny problem A x = b, with x_true known.
        n = 6
        A = LAMG.path_laplacian(n) + 0.1 * I
        x_true = collect(Float64, 1:n); x_true .-= sum(x_true) / n
        b = A * x_true
        # Two "iterates" along directions that bracket x_true.
        x1 = x_true .+ 0.5 * randn(LAMG.Random.MersenneTwister(0), n)
        x2 = x_true .+ 0.5 * randn(LAMG.Random.MersenneTwister(1), n)
        r1 = b .- A * x1
        r2 = b .- A * x2
        # Current iterate: a third with worse residual.
        x = x_true .+ 1.0 * randn(LAMG.Random.MersenneTwister(2), n)
        r = b .- A * x
        h = IterateHistory(n, 2)
        save_iterate!(h, x1, r1)
        save_iterate!(h, x2, r2)
        r_before = norm(r)
        min_res!(h, x, r)
        @test norm(r) <= r_before     # recombination can only help
    end

    @testset "min_res! is a no-op when history is empty" begin
        n = 4
        h = IterateHistory(n, 2)
        x = [1.0, 2.0, 3.0, 4.0]
        r = [0.5, 0.5, 0.5, 0.5]
        x_before = copy(x); r_before = copy(r)
        min_res!(h, x, r)
        @test x == x_before
        @test r == r_before
    end
end
