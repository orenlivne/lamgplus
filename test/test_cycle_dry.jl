using Test
using LAMG

# Dry-run cycle tests: verify the Cycle's level visitation order against
# expected sequences for various γ (cycle index) values. No numerical work
# is done — only the sequence of (hook_name, level) calls is checked.
#
# This guarantees that the visitation logic in Cycle.run_cycle! is correct
# independently of Level / Relaxer implementations.

# Helper: filter out the (:initialize, _) and (:post_cycle, _) entries
# which always wrap any cycle, leaving only the per-level sequence.
function _body(calls)
    [c for c in calls if c[1] ∉ (:initialize, :post_cycle)]
end

@testset "cycle dry run" begin
    @testset "2-level V cycle" begin
        cyc = dry_cycle(1.0, 2)
        run_cycle!(cyc, Float64[])
        @test _body(cyc.processor.calls) == [
            (:pre_process, 1),
            (:process_coarsest, 2),
            (:post_process, 1),
        ]
    end

    @testset "4-level V cycle (γ=1)" begin
        cyc = dry_cycle(1.0, 4)
        run_cycle!(cyc, Float64[])
        @test _body(cyc.processor.calls) == [
            (:pre_process, 1),
            (:pre_process, 2),
            (:pre_process, 3),
            (:process_coarsest, 4),
            (:post_process, 3),
            (:post_process, 2),
            (:post_process, 1),
        ]
    end

    @testset "3-level W cycle (γ=2)" begin
        cyc = dry_cycle(2.0, 3)
        run_cycle!(cyc, Float64[])
        @test _body(cyc.processor.calls) == [
            (:pre_process, 1),
            (:pre_process, 2),
            (:process_coarsest, 3),
            (:post_process, 2),
            (:pre_process, 2),
            (:process_coarsest, 3),
            (:post_process, 2),
            (:post_process, 1),
        ]
    end

    @testset "4-level W cycle (γ=2) - full pattern" begin
        cyc = dry_cycle(2.0, 4)
        run_cycle!(cyc, Float64[])
        seq = _body(cyc.processor.calls)
        # The W cycle on 4 levels visits the coarsest 4 times.
        n_coarsest = count(c -> c == (:process_coarsest, 4), seq)
        @test n_coarsest == 4
        # And every descent from level 1 is followed eventually by an ascent.
        n_pre1 = count(c -> c == (:pre_process, 1), seq)
        n_post1 = count(c -> c == (:post_process, 1), seq)
        @test n_pre1 == n_post1 == 1
    end

    @testset "4-level F-like cycle (γ_vec = [_, 1.5, 1.0]) — fractional γ" begin
        # γ[1] unused at finest (max=1 hard-coded). γ[2] = 1.5 means: per
        # descent into level 2, we descend twice from there to level 3
        # (floor(1.5) + 1 because of strict <), then ascend. γ[3] = 1.0
        # gives one coarsest visit per level-3 descent.
        cyc = dry_cycle([1.0, 1.5, 1.0], 4)
        run_cycle!(cyc, Float64[])
        seq = _body(cyc.processor.calls)
        # Expected sequence (hand-traced):
        # pre(1) pre(2) pre(3) coarsest(4) post(3) post(2)
        #        pre(2) pre(3) coarsest(4) post(3) post(2)
        # post(1)
        @test count(c -> c == (:pre_process, 1), seq) == 1
        @test count(c -> c == (:pre_process, 2), seq) == 2
        @test count(c -> c == (:pre_process, 3), seq) == 2
        @test count(c -> c == (:process_coarsest, 4), seq) == 2
        @test count(c -> c == (:post_process, 3), seq) == 2
        @test count(c -> c == (:post_process, 2), seq) == 2
        @test count(c -> c == (:post_process, 1), seq) == 1
    end

    @testset "1-level (no coarsening)" begin
        # Edge case: num_levels = 1 means coarsest == finest. Just a single
        # process_coarsest call should occur.
        cyc = dry_cycle(1.0, 1)
        run_cycle!(cyc, Float64[])
        @test _body(cyc.processor.calls) == [(:process_coarsest, 1)]
    end
end
