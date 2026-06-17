"""
test_port_regression.jl — guards against the two LAMG port regressions found
by line-by-line comparison against MATLAB lamg-2.2.1 (octave) on identical
(L, b) systems:

  Bug #1 (aggregation): a hard `agg_max_size` cap (was 4) crippled scale-free /
     web graphs — the energy-ratio guard Q (LOP168 Eq. 3.12) is the sole,
     adaptive size regulator and must be allowed to form large hub aggregates.
     Also: leftover undecided nodes must become singleton seeds (not be
     force-absorbed without the guard), and the guard is asymmetric.

  Bug #2 (γ schedule): `γ_coarse_growth` < 1 DECAYED the per-level cycle index
     toward the V-cycle floor on coarse levels, so most levels got a single
     coarse visit and iterate recombination had only n_active=1 (a weak 1-D
     line search). It must stay ≥ 1 so γ ≈ γ_coarse at every level.

These caused web-Stanford ACF ≈ 0.97 (vs MATLAB 0.06) and grid ACF ≈ 0.55
(vs 0.21). With the fixes Julia matches/beats MATLAB across graph classes.
"""

using Test, LAMG, SparseArrays, LinearAlgebra, Random, Statistics
import LAMG: _build_gamma_vec, aggregate, grid2d_laplacian, setup, solve,
             LAMGOptions, laplacian

# Build a Barabási–Albert-style scale-free graph Laplacian (planted hubs).
function scale_free_laplacian(n::Int, m::Int; seed = 1)
    rng = MersenneTwister(seed)
    I = Int[]; J = Int[]; targets = collect(1:m)
    repeated = Int[]
    for v in (m + 1):n
        chosen = Set{Int}()
        while length(chosen) < m
            t = isempty(repeated) ? rand(rng, 1:(v - 1)) : repeated[rand(rng, 1:length(repeated))]
            t < v && push!(chosen, t)
        end
        for t in chosen
            push!(I, v); push!(J, t); push!(repeated, v); push!(repeated, t)
        end
    end
    W = sparse([I; J], [J; I], ones(2 * length(I)), n, n)
    foreach(j -> (for k in nzrange(W, j); W.rowval[k] == j && (W.nzval[k] = 0.0); end), 1:n)
    dropzeros!(W)
    laplacian(W)
end

# Anisotropic 2D grid: horizontal edges weight `wh`, vertical `wv`. The slow error modes run
# along the weak direction; caliber-1 piecewise-constant interpolation stalls on them (the bodyy5
# failure class), and the per-node caliber-2 upgrade is what recovers a healthy rate.
function aniso_grid(k::Int; wh = 1000.0, wv = 1.0)
    idx(i, j) = (j - 1) * k + i
    I = Int[]; J = Int[]; V = Float64[]
    for i in 1:k, j in 1:k
        if i < k; push!(I, idx(i, j)); push!(J, idx(i + 1, j)); push!(V, wh)
                  push!(I, idx(i + 1, j)); push!(J, idx(i, j)); push!(V, wh); end
        if j < k; push!(I, idx(i, j)); push!(J, idx(i, j + 1)); push!(V, wv)
                  push!(I, idx(i, j + 1)); push!(J, idx(i, j)); push!(V, wv); end
    end
    laplacian(sparse(I, J, V, k * k, k * k))
end

# Isotropic grid EXCEPT a p×p anisotropic patch (weak vertical `wv`) in a corner — mostly
# isotropic with a small anisotropic region. A GLOBAL anisotropy-fraction trigger reads ~(p/k)²
# (far below ½) and would leave the patch unrefined; the per-node caliber-2 gate upgrades exactly
# the patch. Guards that the refinement decision is LOCAL, not a global average.
function mixed_grid(k::Int, p::Int; wv = 1e-4)
    idx(i, j) = (j - 1) * k + i
    inpatch(i, j) = (i <= p && j <= p)
    I = Int[]; J = Int[]; V = Float64[]
    for i in 1:k, j in 1:k
        if i < k; push!(I, idx(i, j)); push!(J, idx(i + 1, j)); push!(V, 1.0)
                  push!(I, idx(i + 1, j)); push!(J, idx(i, j)); push!(V, 1.0); end
        if j < k; w = (inpatch(i, j) && inpatch(i, j + 1)) ? wv : 1.0
                  push!(I, idx(i, j)); push!(J, idx(i, j + 1)); push!(V, w)
                  push!(I, idx(i, j + 1)); push!(J, idx(i, j)); push!(V, w); end
    end
    laplacian(sparse(I, J, V, k * k, k * k))
end

acf(info) = (t = info.conv_factors[max(end - 4, 1):end];
             isempty(t) ? 0.0 : exp(mean(log.(max.(t, 1e-30)))))

@testset "LAMG port-regression guards" begin
    @testset "bug #2: γ schedule does not decay across depth" begin
        γ = _build_gamma_vec(1.5, 1.5, 1.0, 8)          # default growth = 1.0
        @test length(γ) == 7
        @test all(γ[2:end] .≈ 1.5)                       # constant, NOT decaying
        γ_old = _build_gamma_vec(1.5, 1.5, 0.7, 8)       # old buggy default
        @test γ_old[end] ≤ 1.0 + 1e-9                    # decayed to the floor
        @test γ_old[end] < γ[end] - 0.1                  # strictly worse
    end

    @testset "bug #1: default has no hard aggregate-size cap" begin
        @test LAMGOptions().agg_max_size > 1_000_000     # effectively unbounded
    end

    @testset "bug #1: energy guard forms large hub aggregates (no cap)" begin
        # Two dense cliques (size 30) joined by one edge. Each clique is
        # perfectly correlated under relaxed TVs ⇒ the guard admits a large
        # aggregate. A hard cap of 4 would forbid it.
        k = 30
        I = Int[]; J = Int[]
        for a in 1:k, b in (a + 1):k; push!(I, a); push!(J, b); end
        for a in 1:k, b in (a + 1):k; push!(I, k + a); push!(J, k + b); end
        push!(I, 1); push!(J, k + 1)                     # bridge
        W = sparse([I; J], [J; I], ones(2 * length(I)), 2k, 2k); dropzeros!(W)
        L = laplacian(W)
        ag = aggregate(L; rng = MersenneTwister(0))
        sizes = [count(==(a), ag.aggregate) for a in 1:ag.n_coarse]
        @test maximum(sizes) > 4                          # NOT capped at 4
    end

    @testset "end-to-end: grid converges fast (γ fix)" begin
        L = grid2d_laplacian(48, 48); n = size(L, 1)
        rng = MersenneTwister(3); xt = randn(rng, n); xt .-= sum(xt) / n; b = L * xt
        _, info = solve(L, b; options = LAMGOptions(tol = 1e-10, max_cycles = 40))
        @test acf(info) < 0.30                            # was ≈ 0.55 (decayed γ)
    end

    @testset "end-to-end: scale-free converges fast (size-cap fix)" begin
        L = scale_free_laplacian(4000, 3; seed = 7); n = size(L, 1)
        rng = MersenneTwister(5); xt = randn(rng, n); xt .-= sum(xt) / n; b = L * xt
        _, info = solve(L, b; options = LAMGOptions(tol = 1e-10, max_cycles = 40))
        @test acf(info) < 0.40                            # was ≈ 0.9+ (cap=4)
    end

    # Without iterate recombination the LAMG cycle is a FIXED linear iteration
    # x_{k+1} = M x_k + c. Its per-cycle residual factor must converge to a
    # stationary asymptotic value ρ(M): high-frequency transients decay first,
    # then the factor plateaus. Recombination is Krylov-like and intentionally
    # NON-stationary, so this property is checked with it OFF.
    @testset "cycle is stationary with recombination off" begin
        for k in (32, 48)
            L = grid2d_laplacian(k, k); n = size(L, 1)
            rng = MersenneTwister(1); xt = randn(rng, n); xt .-= sum(xt) / n
            b = L * xt
            # do_recomb=false: test the RAW fixed-iteration cycle (a stationary linear iteration),
            # not the Krylov recombination that deliberately makes it non-stationary.
            _, info = solve(L, b; options = LAMGOptions(tol = 1e-13, max_cycles = 22,
                                                        do_recomb = false))
            cf = info.conv_factors
            @test length(cf) >= 6
            # asymptotic factor stabilized: late per-cycle factors nearly equal.
            @test abs(cf[end] - cf[end - 1])     < 0.01
            @test abs(cf[end - 1] - cf[end - 2]) < 0.02
            @test 0.0 < cf[end] < 1.0                     # a genuine contraction
            @test cf[end] >= cf[end - 2] - 0.02           # monotone approach
        end
    end

    # Anisotropic graphs (extreme weight ratios, e.g. the bodyy5 FEM stiffness matrix) stall a
    # caliber-1 piecewise-constant interpolation (ACF ≈ 0.99). The always-on, self-targeting
    # caliber-2 upgrade (default) recovers a healthy rate at the cheap fixed K=4 — no test-vector
    # escalation. Guards that the default config converges on anisotropy.
    @testset "anisotropic grid converges (always-on caliber-2)" begin
        L = aniso_grid(48; wh = 1000.0, wv = 1.0); n = size(L, 1)
        rng = MersenneTwister(3); xt = randn(rng, n); xt .-= sum(xt) / n
        b = L * xt
        _, info = solve(L, b; options = LAMGOptions(tol = 1e-10, max_cycles = 40))
        @test acf(info) < 0.45                            # caliber-1 alone stalls ≈ 0.99
    end

    # Strength-of-connection veto (agg_soc_τ). On a large, strongly grid-aligned
    # anisotropic grid, TV sampling noise can spuriously rank a weight-ε edge above
    # the weight-1 edge, causing one cross-weak-direction merge that tanks the rate
    # (ACF ≈ 0.99). The matrix is not fooled; soc_τ>0 vetoes near-zero-conductance
    # merges. This guards both the default (soc_τ=0.05 converges) and that it is the
    # SoC veto doing it (soc_τ=0 still stalls), and that the veto is a no-op on an
    # isotropic grid (no near-zero edges ⇒ identical to soc_τ=0).
    @testset "strength-of-connection veto fixes anisotropic outliers" begin
        L = aniso_grid(256; wh = 1.0, wv = 1e-4); n = size(L, 1)
        rng = MersenneTwister(1); xt = randn(rng, n); xt .-= sum(xt) / n; b = L * xt
        # default (soc_τ = 0.05) converges; soc_τ = 0 stalls on this size/seed.
        _, i_on  = solve(L, b; options = LAMGOptions(tol = 1e-9, max_cycles = 80))
        _, i_off = solve(L, b; options = LAMGOptions(tol = 1e-9, max_cycles = 80, agg_soc_τ = 0.0))
        @test acf(i_on)  < 0.30          # SoC veto on (default): healthy
        @test acf(i_off) > 0.80          # SoC veto off: the documented stall it fixes
        # no-op on an isotropic grid (no near-zero-conductance edges to veto):
        Li = grid2d_laplacian(96, 96); ni = size(Li, 1)
        r2 = MersenneTwister(2); yt = randn(r2, ni); yt .-= sum(yt) / ni; bi = Li * yt
        _, j_on  = solve(Li, bi; options = LAMGOptions(tol = 1e-9, max_cycles = 40))
        _, j_off = solve(Li, bi; options = LAMGOptions(tol = 1e-9, max_cycles = 40, agg_soc_τ = 0.0))
        @test isapprox(acf(j_on), acf(j_off); atol = 0.03)   # veto is inert on isotropic
    end

    # LAMG+ closes the grid-aligned-anisotropy tail with two LOCAL, always-on refinements: the
    # per-edge strength-of-connection veto and the per-node selective caliber-2 interpolation (a
    # fine node whose strong edges reach exactly two coarse aggregates gets a fitted second
    # parent). Both are self-targeting — caliber-2 fires on ~0% of isotropic nodes — so there is
    # NO global trigger and NO test-vector escalation: a fixed, cheap K=4 plus always-on caliber-2
    # converges on pure anisotropy AND on a PARTIALLY anisotropic graph (a small anisotropic patch
    # in an isotropic sea), the case a global anisotropy-fraction threshold would miss.
    @testset "local always-on caliber-2 fixes anisotropy (no global gate)" begin
        function solveacf(L, opts)
            n = size(L, 1); r = MersenneTwister(1); xt = randn(r, n); xt .-= sum(xt) / n
            _, info = solve(L, L * xt; options = opts); acf(info)
        end
        cal2(maxc) = LAMGOptions(tol = 1e-9, max_cycles = maxc, agg_K = 4)               # default: on
        cal1(maxc) = LAMGOptions(tol = 1e-9, max_cycles = maxc, agg_K = 4, caliber2_1d = false)
        # pure grid-aligned anisotropy at two sizes: always-on caliber-2 @ K=4 is healthy, while
        # caliber-1 alone is the documented stall.
        for k in (256, 384)
            La = aniso_grid(k; wh = 1.0, wv = 1e-4)
            @test solveacf(La, cal2(80)) < 0.30
            @test solveacf(La, cal1(80)) > 0.45
        end
        # the decisive case: a 64×64 anisotropic patch in a 256² isotropic grid (~6% of nodes —
        # far below any majority threshold). A global gate would not escalate and the patch would
        # stall the whole solve; the LOCAL caliber-2 upgrade rescues exactly that patch.
        Lm = mixed_grid(256, 64; wv = 1e-4)
        @test solveacf(Lm, cal2(40)) < 0.10
        @test solveacf(Lm, cal1(40)) > 0.20
        # isotropic control (Brandt falsifier): caliber-2 is self-targeting ⇒ neutral.
        Li = grid2d_laplacian(96, 96)
        @test isapprox(solveacf(Li, cal2(40)), solveacf(Li, cal1(40)); atol = 0.03)
    end
end
