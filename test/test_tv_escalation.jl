using Test
using LinearAlgebra
using SparseArrays
using Random
using Statistics
using LAMG

# Adaptive test-vector escalation: inherited test vectors + rebuild-on-slow-convergence.
@testset "test-vector escalation / inherited test vectors" begin
    L = grid2d_laplacian(40, 40); n = size(L, 1)
    rng = MersenneTwister(0xb007)
    xt = randn(rng, n); xt .-= mean(xt); b = L * xt
    opt = LAMGOptions(tol = 1e-10, max_cycles = 100, reorder = false)

    @testset "setup with supplied finest TVs inherits them down the hierarchy" begin
        X0 = LAMG._relax_test_vectors(L, 4, 3, MersenneTwister(1))
        h = setup(L; options = opt, tv0 = X0)
        h0 = setup(L; options = opt)
        @test length(h) >= 2
        @test abs(length(h) - length(h0)) <= 2                    # comparable depth
        for lv in h.levels                                        # structural invariants
            lv.p === nothing && continue
            @test lv.p * ones(size(lv.p, 2)) ≈ ones(size(lv.p, 1))   # constants preserved
            @test maximum(abs, vec(sum(lv.a; dims = 2))) < 1e-8      # zero row sums
        end
        x, info = solve(h, b; options = opt)
        @test norm(L * x - b) / norm(b) < 1e-9
        @test info.cycles <= 12
    end

    @testset "supplied TVs are permuted with the RCM reorder" begin
        # Random-numbered graph so the self-gated reorder actually fires; the hierarchy must still
        # be valid and solve the system when TVs are supplied in the caller's ordering.
        Lr = random_graph_laplacian(600, 0.02; rng = MersenneTwister(3)); nr = size(Lr, 1)
        o = LAMGOptions(tol = 1e-9, max_cycles = 100, reorder = true)
        X0 = LAMG._relax_test_vectors(Lr, 4, 3, MersenneTwister(2))
        h = setup(Lr; options = o, tv0 = X0)
        br = randn(MersenneTwister(4), nr); br .-= mean(br)
        x, info = solve(h, br; options = o)
        @test norm(Lr * x - br) / norm(br) < 1e-8
    end

    @testset "_escalation_tvs returns escalate_tv_K relaxed test vectors" begin
        h = setup(L; options = opt); A = LAMG.finest_level(h).a
        X = LAMG._escalation_tvs(h, opt, MersenneTwister(7))
        @test size(X) == (n, opt.escalate_tv_K)
        @test opt.escalate_tv_K > opt.agg_K                       # more than the lean default
        @test all(isfinite, X)
        @test all(norm(X[:, k]) > 0 for k in 1:size(X, 2))
        @test size(LAMG._relax_test_vectors(A, 4, 3, MersenneTwister(7)), 2) == 4
        R = randn(MersenneTwister(7), n, 1); R .-= mean(R); R ./= norm(R)
        ray(v) = dot(v, A * v) / dot(v, v)
        @test all(ray(X[:, k]) < ray(R[:, 1]) for k in 1:size(X, 2))   # relaxed ⇒ smoother than white
    end

    i_ref = solve(L, b; options = LAMGOptions(tol = 1e-12, max_cycles = 100, reorder = false))[2]
    @testset "healthy solve never triggers; forced trigger rebuilds and still converges" begin
        _, i0 = solve(L, b; options = opt)
        @test !i0.tv_escalated
        @test i0.rebuilt_hierarchy === nothing
        # threshold 0 ⇒ the trigger fires at cycle 4 regardless of the (good) factors
        of = LAMGOptions(tol = 1e-10, max_cycles = 100, reorder = false, escalate_tv_threshold = 0.0)
        xf, i1 = solve(L, b; options = of)
        @test i1.tv_escalated
        @test i1.rebuilt_hierarchy isa LAMG.Multilevel
        @test norm(L * xf - b) / norm(b) < 1e-9
        @test i1.cycles <= i0.cycles + 3            # continuing from the iterate: no restart penalty
        # escalate_tv_max is respected
        om = LAMGOptions(tol = 1e-10, max_cycles = 100, reorder = false, escalate_tv_threshold = 0.0,
                         escalate_tv_max = 0)
        _, i2 = solve(L, b; options = om)
        @test !i2.tv_escalated
    end

    @testset "guard: a rebuild that does not pay off is reverted; a good one is kept" begin
        # margin 0 ⇒ any post-rebuild rate counts as "worse": the solve must revert to the
        # original hierarchy one window after the (forced) rebuild and still converge.
        og = LAMGOptions(tol = 1e-12, max_cycles = 100, reorder = false, escalate_tv_threshold = 0.0,
                         escalate_tv_window = 1, escalate_tv_guard_margin = 0.0)
        xg, ig = solve(L, b; options = og)
        @test ig.tv_escalated
        @test ig.tv_reverted
        @test ig.rebuilt_hierarchy === nothing        # the hierarchy in use at the end is the original
        @test norm(L * xg - b) / norm(b) < 1e-11
        @test ig.cycles <= i_ref.cycles + 4
        # a huge margin ⇒ the rebuild is always kept
        ok = LAMGOptions(tol = 1e-12, max_cycles = 100, reorder = false, escalate_tv_threshold = 0.0,
                         escalate_tv_window = 1, escalate_tv_guard_margin = 1e6)
        _, ik = solve(L, b; options = ok)
        @test ik.tv_escalated && !ik.tv_reverted && ik.rebuilt_hierarchy isa LAMG.Multilevel
        # guard off ⇒ never reverts
        on = LAMGOptions(tol = 1e-12, max_cycles = 100, reorder = false, escalate_tv_threshold = 0.0,
                         escalate_tv_window = 1, escalate_tv_guard = false, escalate_tv_guard_margin = 0.0)
        _, inn = solve(L, b; options = on)
        @test inn.tv_escalated && !inn.tv_reverted
    end

    @testset "trigger uses the sliding geometric mean of the last window" begin
        # window 2, threshold just below the healthy solve's mean rate over cycles 3-4 ⇒ fires;
        # threshold 1.0 ⇒ never fires (factors < 1 on a converging solve).
        _, i0 = solve(L, b; options = opt)
        f = i0.conv_factors
        if length(f) >= 4
            gm = exp((log(f[3]) + log(f[4])) / 2)
            ot = LAMGOptions(tol = 1e-10, max_cycles = 100, reorder = false,
                             escalate_tv_window = 2, escalate_tv_threshold = 0.99 * gm)
            _, it = solve(L, b; options = ot)
            @test it.tv_escalated
        end
        o1 = LAMGOptions(tol = 1e-10, max_cycles = 100, reorder = false, escalate_tv_threshold = 1.0)
        _, i1 = solve(L, b; options = o1)
        @test !i1.tv_escalated
    end

    @testset "legacy restart escalation still available with escalate_tv=false" begin
        ol = LAMGOptions(tol = 1e-10, max_cycles = 100, reorder = false, escalate_tv = false)
        x, info = solve(L, b; options = ol)
        @test norm(L * x - b) / norm(b) < 1e-9
        @test !info.tv_escalated
    end
end
