using Test
using LinearAlgebra
using SparseArrays
using LAMG

@testset "solve_cycle" begin
    @testset "2-level solve cycle reduces residual on Ax = b" begin
        n = 16
        # Use a strictly-pdef variant for testing convergence (a Laplacian
        # + small diagonal regularization breaks the null space).
        L = sparse(path_laplacian(n) + 1e-3 * I)
        rx = GaussSeidelRelaxer(L)
        finest = create_finest_level(L, rx)

        agg = [div(i - 1, 2) + 1 for i in 1:n]
        P, R, Q = piecewise_constant_interpolation(agg)
        Lc = sparse(galerkin_coarse_operator(L, P))
        rxc = GaussSeidelRelaxer(Lc)
        coarse = Level(Lc, sparse(1.0I, n ÷ 2, n ÷ 2), rxc, R, P, Q)

        mlh = Multilevel(finest)
        push!(mlh, coarse)

        x_true = sin.(2π .* (1:n) ./ n)
        b = L * x_true

        cyc = solve_cycle(mlh, b; γ = 1.0, ν_pre = 2, ν_post = 2,
                          ν_coarsest = -1, do_recomb = false,
                          use_direct_coarsest = false)
        x = zeros(n)
        run_cycle!(cyc, x)
        x = LAMG.result(cyc.processor, 1)
        @test norm(b .- L * x) < norm(b) * 0.5
    end
end
