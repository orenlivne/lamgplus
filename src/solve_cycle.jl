# ── printState trace (MATLAB-compatible) ────────────────────────────────
# When ENV["LAMG_TRACE"] == "1", emit lines in the exact format of
# MATLAB ProcessorSolve.m's printState():
#   "%-5d %-25s %-13.3e (%-13.3e) %-13.3e\n"
# Columns: level, action, ‖r‖, ‖b - A x‖ (consistency check), ‖x‖.
# All norms are the SCALED L^2 (RMS) norm — matching MATLAB lpnorm.m:
#   lpnorm(x) = sqrt( sum(x.^2) / length(x) )
const _TRACE_REF = Ref{Union{Nothing,Bool}}(nothing)
function _trace_on()
    if _TRACE_REF[] === nothing
        _TRACE_REF[] = get(ENV, "LAMG_TRACE", "0") == "1"
    end
    return _TRACE_REF[]::Bool
end
_lpnorm(v::AbstractVector) = isempty(v) ? 0.0 : sqrt(sum(abs2, v) / length(v))

function _print_state(p, l::Int, action::AbstractString)
    _trace_on() || return
    x = p.x[l]; b = p.b[l]; A = p.mlh[l].a
    r_norm = _lpnorm(p.r[l])
    res_check = _lpnorm(b .- A * x)
    x_norm = _lpnorm(x)
    # Match MATLAB %-13.3e format: 13 chars wide, 3 decimal places, scientific.
    @printf("%-5d %-25s %-13.3e (%-13.3e) %-13.3e\n",
            l, action, r_norm, res_check, x_norm)
end

"""
    SolveCycleProcessor(multilevel, b; ν_pre, ν_post, ν_coarsest, do_recomb,
                        history_size, use_direct_coarsest)

Multilevel solution cycle for Ax = b with full LAMG features:
- Branches on level type (AGG vs ELIMINATION).
- Iterate recombination (min-res) at every AGG-coarse level and at the finest.
- Augmented direct solve at the coarsest (with null-space deflation).
- Residuals are tracked alongside iterates to avoid recomputing A x.

PROBLEM-DOMAIN POLYMORPHISM. The cycle is OBLIVIOUS to whether the levels
solve a linear Laplacian system or a constrained max-flow problem. Per-level
relaxation goes through `relax!(lv.relaxer, x, b; sweeps)`; the `Relaxer`
subtype decides what that means. Before each coarse recursion the cycle
calls `update_fas!(coarse.relaxer, fine.relaxer, fine_x, P, T)` so the
coarse-level relaxer can refresh any internal state from the current
fine iterate (e.g. FAS τ-correction of inequality bounds in max-flow);
linear relaxers default to a no-op for this hook.

Port of `+amg/+solve/ProcessorSolve.m`.
"""
mutable struct SolveCycleProcessor <: Processor
    mlh::Multilevel
    rhs::Vector{Float64}
    ν_pre::Int
    ν_post::Int
    ν_coarsest::Int
    do_recomb::Bool
    recomb_above_elim::Bool        # fire min-res ABOVE elimination levels too.
                                    # Default false (exact elim needs no correction
                                    # above it). Set true for APPROXIMATE (sampled)
                                    # elimination, whose error MUST be recombined.
    use_direct_coarsest::Bool
    rhs_correction::Float64        # μ for the flat RHS correction (paper §3.5),
                                    # applied as b_c ← μ·b_c after FAS restrict.
                                    # MATLAB default = 4/3. Set to 1.0 to disable.
    coarsest_factor::Any           # cached LU of augmented coarsest operator
    # Per-level state, PREALLOCATED once in the constructor and reused every cycle
    # (was reallocated per cycle in `initialize!` — the dominant solve allocation).
    x::Vector{Vector{Float64}}
    r::Vector{Vector{Float64}}
    b::Vector{Vector{Float64}}
    x_initial::Vector{Vector{Float64}}                 # FAS correction baseline
    sc::Vector{Vector{Float64}}                        # per-level matvec scratch
    caug::Vector{Float64}                              # augmented coarsest RHS scratch
    xaug::Vector{Float64}                              # augmented coarsest solution scratch
    elim::Vector{Union{Nothing,ElimScratch}}           # per-level allocation-free elim transfer scratch
    history::Vector{IterateHistory}
end

function SolveCycleProcessor(mlh::Multilevel, b::AbstractVector;
                             ν_pre::Int = 2, ν_post::Int = 2,
                             ν_coarsest::Int = -1,
                             do_recomb::Bool = true,
                             recomb_above_elim::Bool = false,
                             history_size::Int = 4,
                             use_direct_coarsest::Bool = true,
                             rhs_correction::Real = 4 / 3)
    n = num_levels(mlh)
    coarsest = mlh[end]
    coarsest_fact = nothing
    if use_direct_coarsest
        # Augmented system removes the null space (the constant vector).
        AL = _augmented_coarsest(coarsest.a)
        coarsest_fact = lu(AL)
    end
    mkbuf() = [zeros(size(mlh[l])) for l in 1:n]       # preallocated per-level buffers
    elim = Union{Nothing,ElimScratch}[is_elimination(mlh[l]) ? ElimScratch(mlh[l]) : nothing
                                      for l in 1:n]
    hist = [IterateHistory(size(mlh[l]), history_size) for l in 1:n]
    naug = size(coarsest.a, 1) + 1
    SolveCycleProcessor(mlh, collect(Float64.(b)), ν_pre, ν_post, ν_coarsest,
                        do_recomb, recomb_above_elim, use_direct_coarsest, Float64(rhs_correction),
                        coarsest_fact,
                        mkbuf(), mkbuf(), mkbuf(), mkbuf(), mkbuf(), zeros(naug), zeros(naug),
                        elim, hist)
end

# Augmented coarsest: append a row/column for the null component {1} to make
# the system non-singular for direct solve. b_aug = [b; 0]; solution returns
# x with zero sum.
function _augmented_coarsest(A::SparseMatrixCSC)
    n = size(A, 1)
    Y = ones(n, 1)
    [A Y; Y' spzeros(1, 1)]
end

function initialize!(p::SolveCycleProcessor, l::Int, _num_levels::Int, x)
    # Buffers are preallocated in the constructor and reused across cycles. Reset only
    # the finest level (coarse buffers are fully overwritten on descent). We do NOT clear
    # `p.history` — it persists across cycles for Krylov-style recombination (LAMG §6).
    p.x[l] .= x
    p.b[l] .= p.rhs
    residual!(p.r[l], p.mlh[l].a, p.x[l], p.b[l])                  # r = b − A x (fused, single pass)
    save_iterate!(p.history[l], p.x[l], p.r[l])
    if _trace_on()
        h = p.history[l]
        _print_state(p, l,
            "Save x to #$(h.latest), active $(h.n_active)")
    end
end

function process_coarsest!(p::SolveCycleProcessor, l::Int)
    lv = p.mlh[l]
    if p.use_direct_coarsest
        @views p.caug[1:end - 1] .= p.b[l]; p.caug[end] = 0.0    # b_aug = [b; 0]
        ldiv!(p.xaug, p.coarsest_factor, p.caug)                  # in-place augmented solve
        @views p.x[l] .= p.xaug[1:end - 1]
        residual!(p.r[l], lv.a, p.x[l], p.b[l])
        _print_state(p, l, "Direct solver")
    elseif p.ν_coarsest == -1
        # Generic dense fallback (regularized LU). Rare; left allocating.
        Adense = Matrix(lv.a) + 1e-12 * I
        p.x[l] .= Adense \ p.b[l]
        residual!(p.r[l], lv.a, p.x[l], p.b[l])
    else
        # Reduce residual by several orders of magnitude via relaxation.
        r_initial_norm = norm(p.r[l])
        target_norm = max(1e-13, 1e-5 * r_initial_norm)
        for _ in 1:p.ν_coarsest
            relax!(lv.relaxer, p.x[l], p.b[l]; sweeps = 1)
            residual!(p.r[l], lv.a, p.x[l], p.b[l])
            norm(p.r[l]) < target_norm && break
        end
    end
end

function pre_process!(p::SolveCycleProcessor, l::Int)
    lv = p.mlh[l]
    c = l + 1
    coarse = p.mlh[c]
    _print_state(p, l, "Initial")
    # Pre-relaxation. Per-level domain dispatch is hidden inside `lv.relaxer`.
    # Residual-maintaining form: r[l] is current on entry (set by initialize! at the
    # finest, or by the parent's restriction seed), and the sweep keeps r exact — so the
    # restriction below needs NO separate residual pass (Brandt §8.7 fused relax+residual).
    relax_resid!(lv.relaxer, p.x[l], p.r[l], lv.a, p.b[l]; sweeps = p.ν_pre)
    _print_state(p, l, "Pre-relax ($(p.ν_pre))")

    if is_elimination(coarse)
        # Elimination: full restriction (multi-stage), seed coarse with restricted x.
        sc = p.elim[c]::ElimScratch
        bc = restrict_elimination!(coarse, p.b[l], sc)       # allocation-free, bit-identical
        p.b[c] .= bc
        coarse_type!(coarse, p.x[l], sc, p.x[c])             # allocation-free, bit-identical
        residual!(p.r[c], coarse.a, p.x[c], p.b[c])
        # FAS coarse-state refresh hook for ELIM relaxers (default: no-op).
        # The max-flow relaxer overrides this to anchor its generalized-
        # constraint active bounds to the current fine edge gaps (PFAS
        # τ-shift): x_C0 = p.x[c] (coarse-initial), φ_fine = p.x[l].
        # Linear path: zero-cost.
        update_fas_elim!(coarse.relaxer, p.x[c], p.x[l])
    else
        # AGG (FAS). Paper §3.6.1 + Options.m default:
        #   energyCorrectionType = 'none', minRes = true
        # I.e., the ADAPTIVE correction is provided by iterate
        # recombination at every level (do_recomb=true), NOT by a
        # flat 4/3 RHS factor. The rhs_correction option remains
        # available for the 'flat' variant (Table 4.4 reports 0.279),
        # but the paper-default adaptive variant (Table 4.4 reports
        # 0.136) uses rhs_correction = 1.0.
        mul!(p.x_initial[c], coarse.r, p.x[l])               # Tx = coarsen_op(x[l]); stored once
        mul!(p.b[c], coarse.q, p.r[l]); p.b[c] .*= p.rhs_correction   # μ·restrict_op(r)
        mul!(p.sc[c], coarse.a, p.x_initial[c]); p.b[c] .+= p.sc[c]    # + coarse.a·Tx
        p.x[c] .= p.x_initial[c]
        residual!(p.r[c], coarse.a, p.x[c], p.b[c])
        # FAS coarse-state refresh hook (default: no-op). The max-flow
        # relaxer overrides this to τ-correct its internal low/high
        # bounds from the current fine iterate. Linear path: zero-cost.
        update_fas!(coarse.relaxer, lv.relaxer, p.x[l], coarse.p, coarse.r)
    end

    # Clear the coarse-level iterate history on descent.
    # Verbatim port of ProcessorSolve.m line 216:
    #     obj.clearHistory(c);
    # The MATLAB comment is "Clear coarse-level iterate history". The
    # rationale: each visit to the coarse level starts a NEW FAS
    # sub-problem whose RHS depends on the current fine-level iterate
    # and τ-correction. Iterates saved on a previous visit solved a
    # different problem, so recombining against them is invalid.
    # Finest-level history persists (cleared only via save_iterate
    # cycling), which gives the Krylov-style acceleration of §6.
    clear_history!(p.history[c])
    _print_state(p, c, "Clearing history")
end

function post_process!(p::SolveCycleProcessor, l::Int)
    lv = p.mlh[l]
    c = l + 1
    coarse = p.mlh[c]

    # Iterate recombination at the coarse level.
    # MATLAB ProcessorSolve.m line 237:
    #     if (obj.doMinRes && ~obj.isAboveElimination(c))
    #         [xc, rc] = obj.minRes(c, xc, rc);
    # `isAboveElimination(c)` is true iff level c+1 is an elimination level.
    # So skip if c+1 is elimination. Crucially: we DO fire min-res at a
    # level c that is itself elimination, provided c+1 is not — because
    # the Schur-coarse iterates accumulate in history(c) across γ-driven
    # multi-visits and can be recombined.
    # Skip min-res above an elimination level ONLY when that elimination is EXACT
    # (it introduces no error to correct). For APPROXIMATE/sampled elimination
    # (recomb_above_elim=true) we MUST fire min-res here to absorb the sampling error.
    above_elim_c = (c + 1 <= length(p.mlh)) && is_elimination(p.mlh[c + 1])
    if p.do_recomb && (!above_elim_c || p.recomb_above_elim) && !_MINRES_NO_COARSE[]
        if _MINRES_RECOMPUTE[]
            min_res!(p.history[c], p.x[c], p.r[c], p.mlh[c].a, p.b[c])
        else
            min_res!(p.history[c], p.x[c], p.r[c])
        end
        _print_state(p, c, "min-res ($(p.history[c].n_active))")
    end

    # Save the pre-correction iterate at this level (skip the finest; saved at end).
    if !is_finest(lv)
        save_iterate!(p.history[l], p.x[l], p.r[l])
        if _trace_on()
            h = p.history[l]
            _print_state(p, l,
                "Save x to #$(h.latest), active $(h.n_active)")
        end
    end

    # Apply coarse-level correction.
    if is_elimination(coarse)
        interpolate_elimination!(coarse, p.x[c], p.elim[c]::ElimScratch, p.x[l])  # allocation-free
    else
        # FAS: x ← x + P(x_c − x_c_initial), all in place
        @. p.sc[c] = p.x[c] - p.x_initial[c]
        mul!(p.sc[l], coarse.p, p.sc[c])
        p.x[l] .+= p.sc[l]
    end
    # The coarse-grid correction changed x[l], so r[l] is now stale. Refresh it once, then
    # residual-maintaining post-relaxation keeps r exact — no separate final residual pass.
    residual!(p.r[l], lv.a, p.x[l], p.b[l])
    _print_state(p, l, "Coarse-grid correction")

    # Post-relaxation (residual-maintaining; r[l] is the exact residual on return).
    relax_resid!(lv.relaxer, p.x[l], p.r[l], lv.a, p.b[l]; sweeps = p.ν_post)
    _print_state(p, l, "Post-relax ($(p.ν_post))")
end

function post_cycle!(p::SolveCycleProcessor, l::Int)
    lv = p.mlh[l]
    # Recombine at the finest, unless the next-coarser is elimination.
    if p.do_recomb && length(p.mlh) > 1 && !is_elimination(p.mlh[l + 1])
        if _MINRES_RECOMPUTE[]
            min_res!(p.history[l], p.x[l], p.r[l], p.mlh[l].a, p.b[l])
        else
            min_res!(p.history[l], p.x[l], p.r[l])
        end
        _print_state(p, l, "min-res ($(p.history[l].n_active))")
    end
    # Project out null space (constant vector for graph Laplacian).
    m = mean(p.x[l])
    p.x[l] .-= m
    p.r[l] = p.b[l] .- lv.a * p.x[l]
    _print_state(p, l, "Removed 0-modes")
end

result(p::SolveCycleProcessor, l::Int) = p.x[l]

"""
    solve_cycle(mlh, b; γ=1.5, ν_pre=2, ν_post=2, ν_coarsest=-1, do_recomb=true,
                history_size=4, use_direct_coarsest=true, num_levels=length(mlh)) -> Cycle

LAMG-style solve cycle. Default γ = 1.5 (fractional cycle index) per LAMG §6.
"""
function solve_cycle(mlh::Multilevel, b::AbstractVector;
                     γ::Real = 1.5, ν_pre::Int = 2, ν_post::Int = 2,
                     ν_coarsest::Int = -1, do_recomb::Bool = true,
                     history_size::Int = 4,
                     use_direct_coarsest::Bool = true,
                     num_levels::Int = length(mlh))
    proc = SolveCycleProcessor(mlh, b; ν_pre = ν_pre, ν_post = ν_post,
                               ν_coarsest = ν_coarsest,
                               do_recomb = do_recomb,
                               history_size = history_size,
                               use_direct_coarsest = use_direct_coarsest)
    Cycle(proc, γ, num_levels)
end
