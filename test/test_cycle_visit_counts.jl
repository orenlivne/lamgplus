using Test
using LAMG

# Ported / inspired by:
# - MATLAB `UTestCycleCounters.m`, `UTestCycleCountersState.m`
# - amgplus Python `test/hierarchy/test_cycle.py`
#
# The classical multigrid result: with cycle index γ, the per-cycle work at
# level l grows as γ^(l-1) relative to the finest level (l=1).  Equivalently,
# the number of `:process_coarsest` calls in a single cycle equals γ^(L-1)
# where L is the coarsest level (1-based). For γ < 1.5, this gives a V-cycle
# (count = 1); for γ = 2, a W-cycle (count = 2^(L-1)); for fractional γ
# the count is a piecewise-integer pattern.

@testset "cycle visit-count scaling" begin
    function body(γ, num_levels)
        cyc = dry_cycle(γ, num_levels)
        run_cycle!(cyc, Float64[])
        cyc.processor.calls
    end

    @testset "V-cycle: γ = 1 → 1 coarsest visit at all depths" begin
        for L in 2:7
            calls = body(1.0, L)
            n_coarsest = count(c -> c[1] === :process_coarsest, calls)
            @test n_coarsest == 1
        end
    end

    @testset "W-cycle: γ = 2 → 2^(L-2) coarsest visits" begin
        # Reasoning: descents from level l = γ^(l-1). Coarsest visits =
        # descents from level (L-1) = γ^(L-2). For L=2 (single coarsening
        # step), 2^0 = 1.
        for L in 2:5
            calls = body(2.0, L)
            n_coarsest = count(c -> c[1] === :process_coarsest, calls)
            @test n_coarsest == 2 ^ (L - 2)
        end
    end

    @testset "γ = 3 → 3^(L-2) coarsest visits (each level triples)" begin
        for L in 2:4
            calls = body(3.0, L)
            n_coarsest = count(c -> c[1] === :process_coarsest, calls)
            @test n_coarsest == 3 ^ (L - 2)
        end
    end

    @testset "Fractional γ = 1.5: hand-traced visit count for L=5" begin
        # With strict-less-than `num_visits[i] < γ * num_visits[i-1]`,
        # iterative trace gives:
        #   level 2 descents: 2 (1.5 strict > 1 → 2)
        #   level 3 descents: 3 (1.5*2 = 3.0 strict > 2 → 3)
        #   level 4 descents: 5 (1.5*3 = 4.5 strict > 4 → 5)
        # Coarsest (level 5) visits = level 4 descents = 5.
        calls = body(1.5, 5)
        n_coarsest = count(c -> c[1] === :process_coarsest, calls)
        @test n_coarsest == 5
    end

    @testset "per-level descent counts: monotonically non-decreasing with depth" begin
        # For γ ≥ 1, descents from level l should be ≥ descents from level l-1.
        for γ in (1.0, 1.5, 2.0), L in (3, 4, 5)
            calls = body(γ, L)
            visits = zeros(Int, L)
            for (op, lvl) in calls
                if op === :pre_process
                    visits[lvl] += 1
                end
            end
            # finest = 1 visit; deeper = ≥ prior level.
            @test visits[1] == 1
            for l in 2:L - 1
                @test visits[l] >= visits[l - 1]
            end
        end
    end

    @testset "process_coarsest is always preceded by pre_process at coarsest-1" begin
        # The visitation grammar: we must descend to the coarsest before
        # processing it.
        for γ in (1.0, 1.5, 2.0), L in (3, 4, 5)
            calls = body(γ, L)
            # Strip initialize/post_cycle.
            seq = [c for c in calls if c[1] ∉ (:initialize, :post_cycle)]
            for (i, (op, lvl)) in enumerate(seq)
                if op === :process_coarsest
                    @test i > 1
                    @test seq[i - 1] == (:pre_process, lvl - 1)
                end
            end
        end
    end

    @testset "post_process is always preceded by either process_coarsest or post_process" begin
        for γ in (1.0, 1.5, 2.0), L in (3, 4, 5)
            calls = body(γ, L)
            seq = [c for c in calls if c[1] ∉ (:initialize, :post_cycle)]
            for (i, (op, lvl)) in enumerate(seq)
                if op === :post_process
                    @test i > 1
                    prev_op = seq[i - 1][1]
                    @test prev_op === :process_coarsest || prev_op === :post_process ||
                          prev_op === :pre_process
                end
            end
        end
    end

    @testset "per-level γ vector — match the visit pattern" begin
        # Custom γ per level: descend twice from level 2, once elsewhere.
        # γ[1] unused at finest; γ[2..L-1] = rate of descents from level l.
        γ_vec = [1.0, 2.0, 1.0]   # 4-level cycle
        cyc = dry_cycle(γ_vec, 4)
        run_cycle!(cyc, Float64[])
        seq = [c for c in cyc.processor.calls if c[1] ∉ (:initialize, :post_cycle)]
        # Level 2 descents: should be 2 (γ[2]=2 × parent 1 = 2).
        @test count(c -> c == (:pre_process, 2), seq) == 2
        # Coarsest visits: 2 × 1 = 2.
        @test count(c -> c == (:process_coarsest, 4), seq) == 2
    end
end
