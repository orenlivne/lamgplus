"""
    LAMGOptions

User-tunable LAMG hyperparameters. Mirrors MATLAB's `amg.api.Options` with
just the options that affect the algorithm (no I/O / plotting flags).
"""
Base.@kwdef struct LAMGOptions
    # ───── Setup: hierarchy build ─────
    max_levels::Int = 20
    min_coarse_size::Int = 20             # stop coarsening at/below this.
                                          # NOTE: MATLAB lamg-2.2.1 uses
                                          # minCoarseSize=300; aligning to 300
                                          # makes Julia's level counts match
                                          # MATLAB exactly with identical ACF and
                                          # slightly less cycle work, but direct-
                                          # solves sub-300-node problems (changing
                                          # small-graph + maxflow test behavior).
                                          # Kept at 20 as the safe default.

    # ───── Setup: elimination (Schur) ─────
    elim_max_degree::Int = 4              # eliminate nodes of degree ≤ this
    elim_max_stages::Int = 100            # cap on elimination stages per level
    elim_min_fraction::Float64 = 0.01     # min |f|/n to keep eliminating
    # Fill-cap-controlled selective elimination (off by default; opt-in flag).
    # When > 0, treat as the upper bound on nnz/n inflation per elimination
    # stage: degree-≤(elim_max_degree-1) nodes are always candidates; the
    # remaining (degree == elim_max_degree) candidates are randomly sub-
    # sampled so that the projected nnz/n on survivors does not exceed
    # `elim_fill_cap` × (current nnz/n). 0.0 = feature disabled (use all
    # qualifying nodes regardless of fill).
    elim_fill_cap::Float64 = 0.0          # 0 = disabled; e.g. 1.5 = at most 50% growth
    elim_fill_cap_rng_seed::UInt = 0xe17  # rng seed for the sampling
    # Deterministic fill-GATED elimination: also eliminate higher-degree nodes
    # (up to elim_fill_hard_cap) when their neighbours are already a near-clique,
    # i.e. removing them adds ≤ elim_fill_tol coarse edges. Unlike the random
    # fill_cap sampling above, this is deterministic and only adds bounded fill,
    # so the coarse graph stays sparse while coarsening further. -1 = disabled.
    elim_fill_tol::Int = -1               # -1 = off; 1 or 2 = tolerated fill per node
    elim_fill_hard_cap::Int = 8           # max degree considered for the fill gate
    elim_fill_deg_budget::Int = 64        # skip the clique test on hub-adjacent nodes
    # Fill-aware degree-4 gate: a degree-elim_max_degree (=4) node is the ONLY exact-
    # elimination case that grows the operator (net +2 edges) — and only when its
    # neighbours aren't already connected. elim_fill_max caps the NEW coarse edges a
    # such node may add: deg≤3 always eliminated (net-non-increasing), deg==4 only if it
    # adds ≤ elim_fill_max edges. Directly trims the ~39% of OC that comes from elim fill,
    # cutting BOTH setup and solve. typemax = off (current behaviour). Convergence-tested.
    elim_fill_max::Int = typemax(Int)
    # AC-sampled APPROXIMATE elimination (UPDATE-6). When elim_sample_rho>0, the
    # elimination phase admits higher-degree independent nodes (up to
    # elim_sample_max_degree) and AC-samples their O(d²) Schur clique fill (Kyng–Sachdeva,
    # ρ=elim_sample_rho passes). Coarse operator is approximate but SPD + row-sums-exact;
    # F-recovery stays exact; the cycle + iterate recombination absorb the error. Frozen
    # at setup. Default 0 ⇒ exact elimination (path unchanged).
    elim_sample_rho::Float64 = 0.0
    elim_sample_hub_min_degree::Int = 16  # HUB-ONLY: sample-eliminate deg≥this; deg≤4 exact;
                                          # medium band aggregated. Keeps OC low.

    # ───── Setup: aggregation (caliber-1 PC P + LAMG §3.4 energy guard) ─────
    agg_ν::Int = 3                        # GS sweeps to relax test vectors
    agg_K::Int = 4                        # # test vectors (affinity samples). K=4
                                          # is the bootstrap-justified count for
                                          # caliber-2 (k≲2c with c=2; Brandt et al.,
                                          # Bootstrap AMG 2011, §2.1). Once the SoC
                                          # veto (agg_soc_τ) and selective caliber-2
                                          # (caliber2_1d) are on, they own the
                                          # affinity-variance and interpolation-order
                                          # defects that a larger K once masked, so
                                          # K=8 is redundant — and on a partially
                                          # anisotropic graph it is pure waste (it
                                          # doubles the dominant TV cost everywhere
                                          # to help a local minority). Measured: at
                                          # caliber-2, K=4 matches K=8 to <1 cycle on
                                          # pure anisotropy and BEATS it on mixed
                                          # graphs. A fixed K=4 halves the MATLAB
                                          # 4→8 TV cost, still O(Km).
    agg_max_size::Int = typemax(Int)      # max aggregate size. Per LOP168
                                          # §3.4.4 there is NO hard cap — the
                                          # energy-ratio guard Q (Eq. 3.12) is
                                          # the sole, ADAPTIVE size regulator
                                          # (small ~2 on meshes, 100+ on
                                          # scale-free hubs). The old hard cap
                                          # of 4 crippled scale-free/web graphs
                                          # (ACF 0.97 vs 0.06); unbounded +
                                          # guard matches MATLAB (Inf).
    # Affinity-threshold schedule per pairing stage. Lower = more permissive.
    # MATLAB default `deltaInitial = 0.0` (no threshold, rely on energy guard);
    # paper text uses (0.9, 0.54). We use a finer-grained descending sweep.
    agg_δ_stages::NTuple{4,Float64} = (0.9, 0.7, 0.5, 0.4)
    # Non-Galerkin sparsification of coarse operators: drop off-diagonal couplings with
    # |a_ij| < tol·sqrt(a_ii·a_jj), lumping to the diagonal (preserves the null space).
    # Reduces operator complexity on densifying (scale-free/social) coarse levels — cuts
    # BOTH setup (cheaper Galerkin/Schur to build downstream) and solve (cheaper cycle).
    # 0.0 = off (exact Galerkin). Convergence-equivalence-tested, not bit-identical.
    galerkin_sparsify_tol::Float64 = 0.0
    agg_Q::Float64 = 2.5                  # LAMG §3.4 energy-ratio bound on pairing
    # Strength-of-connection veto on aggregation. soc_τ>0: never aggregate
    # across an edge whose matrix weight is < soc_τ·(node's max incident weight).
    # Guards against the affinity (a noisy K-sample statistic) spuriously ranking
    # a near-zero-conductance edge above a strong one — the rare local defect that
    # tanks anisotropic grids (a weight-ε y-edge with affinity 0.97 > the weight-1
    # x-edge). The energy guard doesn't catch it (same TV noise makes q small too).
    # 0.0 = disabled (pure-affinity LAMG). 0.05 only vetoes truly negligible
    # edges (a no-op on graphs without near-zero couplings — verified neutral
    # across web/social/road/FE/circuit and giants to 15M edges), while removing
    # the rare anisotropic outliers (256²/384² ε=1e-4: ACF 0.99→0.11). Default on.
    agg_soc_τ::Float64 = 0.05
    agg_target_ratio::Float64 = 0.5       # early-stop coarsening ratio (per call)
    # Hub-isolation: nodes with degree ≥ this multiple of their neighbors'
    # MEDIAN degree are marked as forced singleton seeds before the main
    # aggregation pass. Port of lamg-2.2.1 aggregationDegreeThreshold = 8.
    # Prevents densification of the coarse Galerkin operator on scale-free
    # graphs (hub-dominated topologies). Set to 0.0 to disable.
    agg_hub_threshold::Float64 = 8.0
    # Stop coarsening when the aggregation ratio n_c/n is no better than this.
    # 0.7 = require we kept at most 70% of the nodes.
    agg_max_ratio::Float64 = 0.7

    # ───── Setup: selective caliber-2 interpolation on locally-1D nodes ─────
    # When true, a fine node whose STRONG edges reach exactly two coarse
    # aggregates (a locally one-dimensional / anisotropic neighborhood) gets a
    # second interpolation parent, with the weight fit by smoothness-weighted
    # least squares over the affinity test vectors (reused — no extra setup
    # cost). Fixes the caliber-1 energy-ratio ceiling (ρ≈0.5→0.12) on 1-D
    # pockets while staying caliber-1 (zero fill) elsewhere. See
    # `caliber2_interpolation`. ON by default: it rescues the anisotropic minority
    # (epb3/thermal/NACA: 100 cycles → 5–14) at ~neutral average cost (geomean work
    # slightly lower, ~+1% median time), and is self-targeting (fires only on locally-1D
    # nodes). Set false for bit-exact MATLAB-LAMG (caliber-1) comparison.
    caliber2_1d::Bool = true
    # ───── EXPERIMENTAL: structure-adaptive collapse coarsening ─────
    # When true, the aggregation step is replaced by `collapse_aggregate`: flat
    # (strongly-converging / expander-like) regions are contracted to one coarse
    # node each, instead of pairwise-matched. Targets the operator-complexity
    # blow-up on dense low-diameter (social/web) graphs. OFF by default.
    collapse_flat::Bool = false
    collapse_thr::Float64 = 0.3           # flat-edge threshold = thr × median algebraic distance
    # Jaccard-priority tie-break in aggregation: among affinity-admissible,
    # energy-feasible candidates, pick the highest neighbourhood-overlap partner
    # (the exact Galerkin-fill predictor) rather than the highest (saturated)
    # affinity. One parameter-free change; default off → path unchanged.
    agg_jaccard_priority::Bool = false
    cal2_τ::Float64 = 0.5                 # strength-of-connection threshold for
                                          # the locally-1D gate: an edge is
                                          # "strong" if |A_ij| ≥ τ·max(rowmax_i,
                                          # rowmax_j). 0.5 = at least half the
                                          # strongest incident coupling.


    # Cycle
    # γ = 1.25 is the WALL-CLOCK optimum (not the paper/MATLAB-port value 1.5): on the
    # benchmark suite it gives IDENTICAL cycle counts to 1.5 (same convergence/robustness:
    # web-Google 7, com-DBLP 4, apache2 9) but a cheaper cycle (fewer coarse-level visits)
    # → ~14–20% faster solve. γ=1.0 is faster still on easy/social graphs but loses
    # robustness on hard ones (web-Google 7→19 cycles), so it is NOT the default. Tests that
    # verify MATLAB-port fidelity pass γ=1.5 explicitly. See FINDINGS UPDATE-20.
    γ::Float64 = 1.25                     # cycle index at the finest level (wall-clock optimum)
    γ_coarse::Float64 = 1.25              # base for non-finest levels.
                                          # Per-level γ_l grows toward γ_coarse·g^l
                                          # where g is set so total cycle work
                                          # stays bounded — see paper §3.6 line 766.
    γ_coarse_growth::Float64 = 1.0        # per-level γ multiplier across depth.
                                          # MUST be ≥1: values <1 DECAY γ toward
                                          # the V-cycle floor on coarse levels
                                          # (0.7 gave γ→1.0 by level ~4), which
                                          # starves iterate recombination — most
                                          # levels then get a single coarse visit
                                          # so min-res has only n_active=1 (a weak
                                          # 1-D line search, not Krylov). 1.0 keeps
                                          # γ≈γ_coarse at every level; the hard
                                          # work-cap γ·τ<0.95 (see _build_gamma_vec)
                                          # still guarantees O(m) per cycle.
    rhs_correction::Float64 = 1.0         # Paper §3.6.1 default = 'none'/'minRes'
                                          # (MATLAB Options.m: energyCorrectionType
                                          # = 'none'; rhsCorrectionFactor = 4/3 is
                                          # the FLAT alternative, only used when
                                          # energyCorrectionType = 'flat'). Adaptive
                                          # correction is provided by iterate
                                          # recombination (do_recomb=true). Set to
                                          # 4/3 to reproduce the flat variant from
                                          # paper Table 4.4 (μ ≈ 0.279 vs adaptive
                                          # μ ≈ 0.136).
    ν_pre::Int = 1                        # Paper Fig 4.5 cycle: (1,2)
    ν_post::Int = 2
    ν_coarsest::Int = -1                  # -1 = use augmented direct solve

    # Cache-locality reordering (zero-tuning): renumber the finest input to RCM order
    # (a symmetric permutation — SAME operator/convergence math) so the relaxation/SpMV
    # gather streams AND the order-sensitive greedy aggregation forms coherent aggregates.
    # SELF-GATED: applied only if it reduces mean bandwidth, so already-local inputs (grids,
    # structured FE) are auto-skipped (no change), and bandwidth-poor inputs (social/web,
    # arbitrary node numbering) are reordered. Big win on web-Google (3.4→2.1× AC). The
    # finest permutation is stored on the hierarchy; solve permutes the RHS / un-permutes x.
    reorder::Bool = true

    # Solve
    tol::Float64 = 1e-8
    max_cycles::Int = 100
    do_recomb::Bool = true
    history_size::Int = 4

    # RNG
    seed::UInt = 0xfa11
end

"""
    LAMGHierarchy

Alias of `Multilevel`. Kept for naming consistency with the MATLAB code.
"""
const LAMGHierarchy = Multilevel

"""
    setup(A::SparseMatrixCSC; options::LAMGOptions=LAMGOptions()) -> Multilevel

Build the LAMG multilevel hierarchy for a symmetric graph Laplacian `A`.

The setup alternates:
1. **Elimination** of low-degree nodes (LAMG §4.1) if any qualify.
2. **Aggregation** via affinity (LAMG §4.2) into a caliber-1 piecewise-constant
   interpolation, when elimination saturates or wouldn't make progress.

Recursion stops when:
- coarse size ≤ `options.min_coarse_size`, OR
- a coarsening step produced ≥ `options.agg_min_coarsening` size ratio
  (aggregation refused to make progress), OR
- `options.max_levels` reached.
"""
function setup(A::SparseMatrixCSC; options::LAMGOptions = LAMGOptions())
    # Tolerance scales with size: 1e-10 is too strict for 1M-edge problems.
    laplacian_tol = max(1e-10, 1e-12 * size(A, 1))
    @assert is_laplacian(A; tol = laplacian_tol) "A must be symmetric with zero row sums"
    rng = MersenneTwister(options.seed)

    # Cache-locality reordering (self-gated, zero-tuning): renumber to RCM order iff it
    # reduces mean bandwidth. A symmetric permutation — same operator, same convergence —
    # but a local sparsity (cache-friendly gather + coherent aggregates). Grids/FE already
    # local ⇒ bandwidth not reduced ⇒ skipped (unchanged); social/web ⇒ reordered.
    perm = Int[]
    if options.reorder
        p = rcm_order(A)
        # Decide matrix-free (no A[p,p] materialization just to check bandwidth), and if it
        # helps, build the permuted operator ONCE via the fast symmetric-permutation kernel
        # (`permute` ≡ A[p,p] but ~2× faster than generic getindex; the prior code built A[p,p]
        # TWICE — once to decide, once for real — which was 30–40% of setup compute on big graphs).
        if mean_bandwidth(A, invperm(p)) < mean_bandwidth(A)
            A = permute(A, p, p)
            perm = p
        end
    end
    n = size(A, 1)

    # Finest level.
    rx = GaussSeidelRelaxer(A; symmetric = true)
    mlh = Multilevel(create_finest_level(A, rx))
    mlh.perm = perm

    Acur = A
    iter = 0
    while size(mlh[end]) > options.min_coarse_size && iter < options.max_levels
        iter += 1
        # Try elimination first; if any low-degree-node F set qualifies,
        # this stage is essentially free in convergence terms (Schur is exact).
        new_lvl, stages, used_elim = _try_elimination(Acur, options)
        if used_elim
            push!(mlh, create_elimination_level(new_lvl,
                                                GaussSeidelRelaxer(new_lvl; symmetric = true),
                                                stages))
            Acur = new_lvl
            continue
        end
        # Otherwise aggregate. Stop if aggregation didn't make enough progress.
        new_lvl, R, P, Q, used_agg = _try_aggregation(Acur, options, rng)
        used_agg || break
        push!(mlh, create_agg_level(new_lvl,
                                    GaussSeidelRelaxer(new_lvl; symmetric = true),
                                    R, P, Q))
        Acur = new_lvl
    end

    # NOTE: do NOT overwrite mlh[end].level_type — that would clobber
    # the :elimination tag and the cycle would treat the level as AGG.
    # The "coarsest" level is just `mlh[end]`; no tag needed.
    return mlh
end

function _try_elimination(A::SparseMatrixCSC, options::LAMGOptions)
    stages = EliminationStage[]
    Acur = A
    n0 = size(A, 1)
    rng = (options.elim_fill_cap > 0 || options.elim_sample_rho > 0) ?
          MersenneTwister(options.elim_fill_cap_rng_seed) : nothing
    for s in 1:options.elim_max_stages
        n = size(Acur, 1)
        n <= options.min_coarse_size && break
        stage, Anext, _ = eliminate_once(Acur;
                                         max_degree = options.elim_max_degree,
                                         min_elim_fraction = options.elim_min_fraction,
                                         fill_cap = options.elim_fill_cap,
                                         fill_tol = options.elim_fill_tol,
                                         fill_hard_cap = options.elim_fill_hard_cap,
                                         fill_deg_budget = options.elim_fill_deg_budget,
                                         fill_max_low = options.elim_fill_max,
                                         sample_rho = options.elim_sample_rho,
                                         sample_hub_min_degree = options.elim_sample_hub_min_degree,
                                         rng = rng)
        stage === nothing && break
        push!(stages, stage)
        Acur = Anext
    end
    used = !isempty(stages)
    return Acur, stages, used
end

function _build_gamma_vec(γ_fine::Float64, γ_coarse::Float64,
                          γ_coarse_growth::Float64, num_levels::Int,
                          mlh::Union{Nothing, Multilevel} = nothing)
    n = num_levels - 1
    n <= 0 && return Float64[]
    # Paper §3.6 line 766: γ_l grows on coarser levels to maximize error
    # reduction; total cycle work ≈ 3/(1 − g) relaxation sweeps so g = 0.7
    # gives ~10 sweeps. Concretely:
    #   γ_l ≤ γ_coarse · (g · m_{l-1}/m_l)
    # capped so work per cycle stays bounded by the geometric series.
    # When `mlh` is given, use the actual coarsening ratio per level;
    # otherwise fall back to a simple geometric schedule.
    γs = Vector{Float64}(undef, n)
    γs[1] = γ_fine
    if mlh === nothing
        @inbounds for i in 2:n
            γs[i] = γ_coarse * γ_coarse_growth ^ (i - 2)
        end
    else
        @inbounds for i in 2:n
            τ = size(mlh[i + 1]) / max(1, size(mlh[i]))
            # Bounded-work γ — leave a 5% safety margin under 1/τ. This cap is a
            # HARD work bound (γ·τ < 1 keeps per-cycle work O(m)); the floor below
            # must NOT exceed it, or poorly-coarsening graphs (τ→1, e.g. near-
            # disconnected web crawls) get γ·τ > 1 and per-cycle work explodes —
            # breaking O(m). Floor at 1.0 (at least a V-cycle).
            γ_cap = τ > 0 ? 0.95 / τ : 3.0
            γs[i] = min(γ_coarse * γ_coarse_growth ^ (i - 2), γ_cap, 3.0)
            γs[i] = max(γs[i], 1.0)
        end
    end
    return γs
end

function _try_aggregation(A::SparseMatrixCSC, options::LAMGOptions, rng)
    n = size(A, 1)
    # EXPERIMENTAL: structure-adaptive collapse coarsening (flag-gated).
    if options.collapse_flat
        Xc = _relax_test_vectors(A, options.agg_K, options.agg_ν, rng)
        agc = collapse_aggregate(A, Xc; thr_frac = options.collapse_thr)
        (agc.n_coarse / n > options.agg_max_ratio) &&
            return A, spzeros(0, 0), spzeros(0, 0), spzeros(0, 0), false
        Pc, Rc, Qc = piecewise_constant_interpolation(agc.aggregate)
        Ac = galerkin_coarse_operator(A, Pc, Qc)
        options.galerkin_sparsify_tol > 0 && (Ac = sparsify_lump(Ac, options.galerkin_sparsify_tol))
        return Ac, Rc, Pc, Qc, true
    end
    # Generate the affinity test vectors once; reuse them for the caliber-2 weight
    # fit (no extra setup cost) when the feature is enabled.
    X = options.caliber2_1d ?
        _relax_test_vectors(A, options.agg_K, options.agg_ν, rng) : nothing
    ag = aggregate(A;
                   ν = options.agg_ν, K = options.agg_K,
                   max_aggregate_size = options.agg_max_size,
                   δ_stages = options.agg_δ_stages,
                   target_coarsening_ratio = options.agg_target_ratio,
                   Q = options.agg_Q,
                   hub_threshold = options.agg_hub_threshold,
                   soc_τ = options.agg_soc_τ,
                   jaccard_priority = options.agg_jaccard_priority,
                   X_ext = X,
                   rng = rng)
    coarsening_ratio = ag.n_coarse / n
    if coarsening_ratio > options.agg_max_ratio
        # Aggregation didn't coarsen enough — likely the graph is already
        # close to its own "skeleton" (mostly degree-1 chains or singletons).
        return A, spzeros(0, 0), spzeros(0, 0), spzeros(0, 0), false
    end
    if options.caliber2_1d
        P, R, Q, _ = caliber2_interpolation(ag.aggregate, X, A; τ = options.cal2_τ)
    else
        P, R, Q = piecewise_constant_interpolation(ag.aggregate)
    end
    # P'·A·P already returns a canonical SparseMatrixCSC — no sparse() re-copy.
    # Reuse the already-built Q (= Pᵀ) so the product doesn't re-materialize P'
    # (bit-identical: Q === sparse(P')).
    Acoarse = galerkin_coarse_operator(A, P, Q)
    options.galerkin_sparsify_tol > 0 &&
        (Acoarse = sparsify_lump(Acoarse, options.galerkin_sparsify_tol))
    return Acoarse, R, P, Q, true
end

"""
    solve(A::SparseMatrixCSC, b::AbstractVector;
          options::LAMGOptions=LAMGOptions(),
          x0::Union{Nothing,AbstractVector}=nothing) -> (x, info)

End-to-end LAMG solve of `Ax = b` where A is a graph Laplacian. Builds the
hierarchy (if not provided) and iterates V/W/F-cycles until
`‖b − Ax‖ ≤ options.tol * ‖b‖`.

Returns `(x, info)` with `info` containing:
- `cycles`           :: number of cycles run
- `residual_history` :: Vector{Float64} of `‖b − Ax‖` per cycle (entry 1 is initial)
- `conv_factors`     :: per-cycle convergence factors
- `final_residual`   :: ‖b − Ax‖ at termination
- `setup_time`       :: seconds spent in `setup`
- `solve_time`       :: seconds spent in the cycle loop
- `hierarchy`        :: the `Multilevel` built (so it can be reused across RHSs)
"""
function solve(A::SparseMatrixCSC, b::AbstractVector;
               options::LAMGOptions = LAMGOptions(),
               x0::Union{Nothing,AbstractVector} = nothing)
    # Build the hierarchy once, at the fixed parameter-free configuration: a cheap K=4 affinity
    # with the always-on, self-targeting refinements (SoC veto + selective caliber-2). Every
    # refinement decision is LOCAL — caliber-2 upgrades only the locally-1-D nodes (a per-node
    # test inside `caliber2_interpolation`), so a partially anisotropic graph pays for the fix
    # only on its anisotropic part. There is no global trigger and no separate "full" build.
    t_setup = @elapsed h = setup(A; options = options)
    x, info = solve(h, b; options = options, x0 = x0)
    return x, merge(info, (setup_time = t_setup, hierarchy = h))
end

"""
    solve(h::Multilevel, b::AbstractVector;
          options::LAMGOptions=LAMGOptions(),
          x0::Union{Nothing,AbstractVector}=nothing) -> (x, info)

Solve `Ax = b` reusing an existing hierarchy `h`.
"""
function solve(h::Multilevel, b::AbstractVector;
               options::LAMGOptions = LAMGOptions(),
               x0::Union{Nothing,AbstractVector} = nothing)
    n = size(finest_level(h))
    @assert length(b) == n "b size mismatch"
    # If the hierarchy was built on a reordered (RCM) input, permute the RHS / initial guess
    # into that space; the solution is un-permuted back to the caller's order at the return.
    p = h.perm
    if !isempty(p)
        b = b[p]
        x0 = x0 === nothing ? nothing : collect(Float64.(x0))[p]
    end
    A = finest_level(h).a
    b_norm = norm(b)
    b_norm == 0 && (return zeros(n), (cycles = 0, residual_history = [0.0],
                                       conv_factors = Float64[],
                                       final_residual = 0.0,
                                       solve_time = 0.0,
                                       gamma_escalated = false))
    x = x0 === nothing ? zeros(Float64, n) : collect(Float64.(x0))
    x_init = copy(x)

    # Build a single processor we reuse across cycles — keeps the iterate
    # history across cycles at the finest level (Krylov-style acceleration).
    proc = SolveCycleProcessor(h, b;
                               ν_pre = options.ν_pre, ν_post = options.ν_post,
                               ν_coarsest = options.ν_coarsest,
                               do_recomb = options.do_recomb,
                               recomb_above_elim = options.elim_sample_rho > 0,
                               history_size = options.history_size,
                               use_direct_coarsest = (options.ν_coarsest == -1),
                               rhs_correction = options.rhs_correction)
    # Cycle-index vector: γ_fine at the finest, then γ_coarse * g^(l-2)
    # capped so work stays bounded.
    # Deep hierarchies (>= 18 levels: huge-diameter graphs) start directly on the grown
    # schedule -- the escalation of the cycle loop below would fire there anyway, and
    # starting grown removes its detection cost. No-op for typical (8-15 level) graphs.
    g_growth = (options.γ_coarse_growth <= 1.0 && length(h) >= 18) ? 1.15 :
               options.γ_coarse_growth
    γ_vec = _build_gamma_vec(options.γ, options.γ_coarse, g_growth, length(h), h)
    cyc = Cycle(proc, γ_vec, length(h))

    residual_history = Float64[b_norm]
    conv_factors = Float64[]
    escalated = false
    t_solve = @elapsed begin
        for k in 1:options.max_cycles
            # Seed state with current x as the cycle's initial guess.
            run_cycle!(cyc, x)
            x = copy(result(proc, 1))
            r = norm(b .- A * x)
            push!(residual_history, r)
            push!(conv_factors, r / residual_history[end - 1])
            r <= options.tol * b_norm && break
            # Deep-hierarchy escalation: a persistently slow cycle (factor > 0.6 twice in a
            # row) on the stock schedule signals an under-worked coarse hierarchy (very
            # large-diameter graphs, e.g. country-scale road networks, whose s–t dipole rhs
            # maximally excites the longest-wavelength mode). Regrow the cycle index by
            # 1.15x per level and continue — the hierarchy itself is untouched, and healthy
            # solves (factors well below 0.6) never trigger this.
            if !escalated && k >= 4 && g_growth <= 1.0 &&
               conv_factors[end] > 0.6 && conv_factors[end-1] > 0.6
                # ESCALATE BY RESTART: the slow phase leaves the error rotated into a
                # subspace no allowed schedule contracts (a measured fact: a fresh grown-γ
                # solve warm-started at this iterate also crawls at ~0.9, while the same
                # solve from scratch runs at 0.02–0.06 to the FP floor). So restart from
                # the initial guess under the grown schedule — the discarded progress is
                # regained within a few grown cycles — with a fresh processor.
                x = copy(x_init)
                proc = SolveCycleProcessor(h, b;
                                           ν_pre = options.ν_pre, ν_post = options.ν_post,
                                           ν_coarsest = options.ν_coarsest,
                                           do_recomb = options.do_recomb,
                                           recomb_above_elim = options.elim_sample_rho > 0,
                                           history_size = options.history_size,
                                           use_direct_coarsest = (options.ν_coarsest == -1),
                                           rhs_correction = options.rhs_correction)
                γ_vec = _build_gamma_vec(options.γ, options.γ_coarse, 1.15, length(h), h)
                cyc = Cycle(proc, γ_vec, length(h))
                escalated = true
            end
        end
    end
    info = (cycles = length(residual_history) - 1,
            residual_history = residual_history,
            conv_factors = conv_factors,
            final_residual = residual_history[end],
            solve_time = t_solve,
            gamma_escalated = escalated)
    if !isempty(p)
        xout = similar(x); xout[p] = x; x = xout   # un-permute to the caller's ordering
    end
    return x, info
end
