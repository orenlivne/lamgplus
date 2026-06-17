"""
    Level

A single level in the multilevel hierarchy. Mirrors `helmholtz.hierarchy.multilevel.Level`
with the LAMG extension for elimination levels.

Fields:
- `a`           :: level operator (e.g. graph Laplacian on this level).
- `b`           :: mass matrix (the identity for plain Laplacian solves).
- `relaxer`     :: a `Relaxer` instance tied to `a`. THE polymorphic
                   point — the `Relaxer` subtype is what knows whether
                   this is a linear-LAMG level or a constrained max-flow
                   level. The cycle code is oblivious.
- `r`           :: coarsening operator `R : x → x^c` (size `n_c × n`).
                   `nothing` on the finest level.
- `p`           :: interpolation `P : x^c → x` (size `n × n_c`).
                   `nothing` on the finest level.
- `q`           :: restriction `Q : b → b^c` (usually `Pᵀ`).
                   `nothing` on the finest level.
- `elim_stages` :: `Vector{EliminationStage}`; non-empty marks this as an
                   ELIMINATION level. Each stage's (P, R, q, f, c, n) is
                   replayed sequentially during restrict / interpolate.
- `level_type`  :: `:finest | :agg | :elimination | :coarsest`.

For an AGG level, `r`, `p`, `q` are used directly. For an ELIMINATION level,
the `elim_stages` are used and `r`, `p`, `q` may be `nothing` (the multi-stage
transfer is *defined by* the stages).

`Level` is a pure STRUCTURAL carrier — it does not know about box constraints,
max-flow problems, or any problem-domain state. All such state lives inside
the `relaxer` subtype.
"""
mutable struct Level
    a::SparseMatrixCSC{Float64,Int}
    b::SparseMatrixCSC{Float64,Int}
    relaxer::Relaxer
    r::Union{Nothing, SparseMatrixCSC{Float64,Int}}
    p::Union{Nothing, SparseMatrixCSC{Float64,Int}}
    q::Union{Nothing, SparseMatrixCSC{Float64,Int}}
    elim_stages::Vector{EliminationStage}
    level_type::Symbol
end

"""
    Level(a, b, relaxer, r, p, q) -> Level

Convenience 6-arg constructor: creates an AGG (aggregation) level. To
construct an elimination level, use `create_elimination_level`.
"""
function Level(a::SparseMatrixCSC, b::SparseMatrixCSC, relaxer::Relaxer,
               r, p, q)
    Level(a, b, relaxer, r, p, q, EliminationStage[],
          (r === nothing && p === nothing && q === nothing) ? :finest : :agg)
end

"""
    create_finest_level(a, b, relaxer) -> Level
"""
function create_finest_level(a::SparseMatrixCSC, b::SparseMatrixCSC,
                             relaxer::Relaxer)
    Level(a, b, relaxer, nothing, nothing, nothing,
          EliminationStage[], :finest)
end

function create_finest_level(a::SparseMatrixCSC, relaxer::Relaxer)
    n = size(a, 1)
    b = sparse(1.0I, n, n)
    create_finest_level(a, b, relaxer)
end

"""
    create_agg_level(a, relaxer, r, p, q) -> Level

Construct an aggregation coarse level from precomputed transfer operators.
"""
function create_agg_level(a::SparseMatrixCSC, relaxer::Relaxer,
                          r::SparseMatrixCSC, p::SparseMatrixCSC,
                          q::SparseMatrixCSC)
    n = size(a, 1)
    Level(a, sparse(1.0I, n, n), relaxer, r, p, q,
          EliminationStage[], :agg)
end

"""
    create_elimination_level(a, relaxer, stages::Vector{EliminationStage}) -> Level
"""
function create_elimination_level(a::SparseMatrixCSC, relaxer::Relaxer,
                                  stages::Vector{EliminationStage})
    n = size(a, 1)
    Level(a, sparse(1.0I, n, n), relaxer, nothing, nothing, nothing,
          stages, :elimination)
end

Base.size(L::Level) = size(L.a, 1)

is_elimination(L::Level) = L.level_type === :elimination
is_finest(L::Level) = L.level_type === :finest

"""
    operator(L::Level, x; lam=0.0) -> Vector

Return `(A - λB) * x`. For Laplacian solves use `lam = 0.0` (default).
"""
function operator(L::Level, x::AbstractVecOrMat; lam::Real = 0.0)
    if iszero(lam)
        return L.a * x
    else
        return L.a * x .- lam .* (L.b * x)
    end
end

"""
    relax!(L::Level, x, b; sweeps=1) -> x
"""
function relax!(L::Level, x::AbstractVector, b::AbstractVector; sweeps::Int = 1)
    relax!(L.relaxer, x, b; sweeps = sweeps)
    return x
end

# For AGG levels: use precomputed R, P, Q matrices.
restrict_op(L::Level, x::AbstractVecOrMat)    = L.q * x
coarsen_op(L::Level, x::AbstractVecOrMat)     = L.r * x
interpolate_op(L::Level, xc::AbstractVecOrMat) = L.p * xc

"""
    restrict_elimination(L::Level, b::AbstractVector) -> (bc, bstages)

For an elimination level, perform the multi-stage restriction of the
fine-level RHS. Returns the final coarse-level RHS plus the per-stage
intermediate RHS arrays (needed during interpolation).

Stage q maps b → b_next where:
   b_next[c_q] = b[c_q] + (Pᵀ)_q * b[f_q]      (= b[c_q] + R_q * b[f_q])

Port of `LevelElimination.m::restrict`.
"""
function restrict_elimination(L::Level, b::AbstractVector)
    @assert is_elimination(L)
    bstages = Vector{Vector{Float64}}(undef, length(L.elim_stages) + 1)
    bstages[1] = collect(Float64.(b))
    b_cur = bstages[1]
    for (q, s) in enumerate(L.elim_stages)
        b_next = b_cur[s.c] .+ s.R * b_cur[s.f]
        bstages[q + 1] = b_next
        b_cur = b_next
    end
    return b_cur, bstages
end

"""
    interpolate_elimination(L::Level, xc::AbstractVector,
                            bstages::Vector{Vector{Float64}}) -> Vector

For an elimination level, expand the coarse solution back through all
stages. At each stage:
   x_next[f_q] = P_q * x_cur + q_q .* b_stage[f_q]
   x_next[c_q] = x_cur

Port of `LevelElimination.m::interpolate`.
"""
function interpolate_elimination(L::Level, xc::AbstractVector,
                                 bstages::Vector{Vector{Float64}})
    @assert is_elimination(L)
    @assert length(bstages) == length(L.elim_stages) + 1
    x_cur = collect(Float64.(xc))
    for q in length(L.elim_stages):-1:1
        s = L.elim_stages[q]
        x_next = zeros(Float64, s.n)
        x_next[s.f] .= s.P * x_cur .+ s.q .* bstages[q][s.f]
        x_next[s.c] .= x_cur
        x_cur = x_next
    end
    return x_cur
end

"""
    coarse_type(L::Level, x::AbstractVector) -> Vector

For an elimination level, return the restriction of fine-level x to the
final c-set (after all stages). This is the initial coarse guess for
elimination-level processing.
"""
function coarse_type(L::Level, x::AbstractVector)
    @assert is_elimination(L)
    x_cur = collect(Float64.(x))
    for s in L.elim_stages
        x_cur = x_cur[s.c]
    end
    return x_cur
end

# ─────────────────────────────────────────────────────────────────────────────
# Allocation-free elimination transfers.
#
# The functions above (restrict_elimination / interpolate_elimination /
# coarse_type) allocate full-length vectors and per-stage SpMV results on EVERY
# cycle. On elimination-heavy (FE/structural) graphs that is the dominant solve
# allocation (apache2: ~3.5 GB/solve). The `!` variants below do the identical
# arithmetic into preallocated `ElimScratch` buffers — bit-for-bit unchanged,
# zero allocation. Used by the solve cycle; the allocating versions are kept for
# other callers (tests, FAMG paths).

"""
    ElimScratch(L::Level)

Preallocated per-stage buffers for the in-place elimination transfers at one
elimination level. Sizes are fixed by the stage chain (independent of the RHS).
"""
struct ElimScratch
    bstages::Vector{Vector{Float64}}   # [1..Q+1]; stage-q RHS (len = stage_q.n); [Q+1] = coarse RHS
    xnext::Vector{Vector{Float64}}     # [1..Q];   interpolate x_next at stage q (len = stage_q.n)
    fbuf::Vector{Vector{Float64}}      # [1..Q];   f-gather / SpMV scratch (len = |f_q|)
    ctbuf::Vector{Vector{Float64}}     # [1..Q];   coarse_type gather chain  (len = |c_q|)
end

function ElimScratch(L::Level)
    @assert is_elimination(L)
    Q = length(L.elim_stages)
    bstages = Vector{Vector{Float64}}(undef, Q + 1)
    xnext   = Vector{Vector{Float64}}(undef, Q)
    fbuf    = Vector{Vector{Float64}}(undef, Q)
    ctbuf   = Vector{Vector{Float64}}(undef, Q)
    for (q, s) in enumerate(L.elim_stages)
        bstages[q] = zeros(s.n)
        xnext[q]   = zeros(s.n)
        fbuf[q]    = zeros(length(s.f))
        ctbuf[q]   = zeros(length(s.c))
    end
    bstages[Q + 1] = zeros(Q == 0 ? size(L) : length(L.elim_stages[Q].c))
    ElimScratch(bstages, xnext, fbuf, ctbuf)
end

"""
    restrict_elimination!(L, b, sc::ElimScratch) -> coarse RHS

In-place multi-stage RHS restriction. Returns `sc.bstages[end]`. The per-stage
RHS arrays `sc.bstages[1..Q]` are needed later by `interpolate_elimination!`.
Bit-identical to `restrict_elimination` (`b_next = b[c] .+ R*b[f]`, addition
commutes exactly).
"""
function restrict_elimination!(L::Level, b::AbstractVector, sc::ElimScratch)
    @assert is_elimination(L)
    bs = sc.bstages
    copyto!(bs[1], b)
    @inbounds for (q, s) in enumerate(L.elim_stages)
        bcur = bs[q]; bnext = bs[q + 1]; fg = sc.fbuf[q]
        f = s.f
        for t in eachindex(f)
            fg[t] = bcur[f[t]]
        end
        mul!(bnext, s.R, fg)              # bnext = R * b[f]
        c = s.c
        for t in eachindex(c)
            bnext[t] += bcur[c[t]]        # + b[c]  (R*b[f] + b[c] == b[c] + R*b[f])
        end
    end
    return bs[end]
end

"""
    interpolate_elimination!(L, xc, sc::ElimScratch, out) -> out

In-place multi-stage expansion of the coarse solution back to the fine size,
written into `out`. Bit-identical to `interpolate_elimination`.
"""
function interpolate_elimination!(L::Level, xc::AbstractVector, sc::ElimScratch,
                                  out::AbstractVector)
    @assert is_elimination(L)
    Q = length(L.elim_stages)
    xcur = xc
    @inbounds for q in Q:-1:1
        s = L.elim_stages[q]
        xn = sc.xnext[q]; Pf = sc.fbuf[q]; bsf = sc.bstages[q]; qv = s.q
        mul!(Pf, s.P, xcur)               # Pf = P * x_cur
        f = s.f
        for t in eachindex(f)
            xn[f[t]] = Pf[t] + qv[t] * bsf[f[t]]
        end
        c = s.c
        for t in eachindex(c)
            xn[c[t]] = xcur[t]
        end
        xcur = xn
    end
    copyto!(out, xcur)
    return out
end

"""
    coarse_type!(L, x, sc::ElimScratch, out) -> out

In-place restriction of fine `x` to the final c-set, written into `out`.
Bit-identical to `coarse_type`.
"""
function coarse_type!(L::Level, x::AbstractVector, sc::ElimScratch,
                      out::AbstractVector)
    @assert is_elimination(L)
    xcur = x
    @inbounds for (q, s) in enumerate(L.elim_stages)
        ct = sc.ctbuf[q]; c = s.c
        for t in eachindex(c)
            ct[t] = xcur[c[t]]
        end
        xcur = ct
    end
    copyto!(out, xcur)
    return out
end
