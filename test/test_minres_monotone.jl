"""
test_minres_monotone.jl — the min-res kernel invariant.

`min_res!` minimizes ‖r − AE·α‖ with AE = r − R_history. The identity
AE = A·E holds *iff* every stored residual is consistent with its stored
iterate (R_i = b − A·X_i). Under that condition the recombination CANNOT
increase the true residual ‖b − A·x‖ (α = 0 is feasible). This pins the
kernel as sound, so that any "recombination made convergence worse"
behavior is localized to the cycle feeding it inconsistent (x, r) — not
to this function.
"""

using Test, LAMG, LinearAlgebra, SparseArrays, Random
import LAMG: IterateHistory, save_iterate!, min_res!, GaussSeidelRelaxer, relax!,
             grid2d_laplacian

@testset "min-res monotonicity (consistent history ⇒ true residual non-increasing)" begin
    for (name, L) in (("grid2d/16", grid2d_laplacian(16, 16)),
                      ("grid2d/24", grid2d_laplacian(24, 24)))
        n = size(L, 1)
        b = randn(MersenneTwister(1), n); b .-= sum(b) / n
        rx = GaussSeidelRelaxer(L)
        hist = IterateHistory(n, 6)
        x = zeros(n)
        # Build a history of CONSISTENT (x_i, r_i = b − L x_i) snapshots.
        for _ in 1:6
            for _ in 1:3; relax!(rx, x, b; sweeps = 1); end
            x .-= sum(x) / n
            save_iterate!(hist, x, b .- L * x)
        end
        for _ in 1:3; relax!(rx, x, b; sweeps = 1); end
        x .-= sum(x) / n
        r = b .- L * x
        true_before = norm(r)
        xc = copy(x); rc = copy(r)
        min_res!(hist, xc, rc)
        true_after = norm(b .- L * xc)
        # (1) the true residual did not increase
        @test true_after <= true_before * (1 + 1e-9)
        # (2) the residual min_res! reports matches the true residual
        #     (this is exactly the AE = A·E identity)
        @test isapprox(norm(rc), true_after; rtol = 1e-8)
        # (3) it actually helped on this well-conditioned case
        @test true_after < true_before
    end
end
