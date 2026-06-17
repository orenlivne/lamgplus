"""
    Relaxer

Abstract base type for relaxation methods. Concrete relaxers must implement

    relax!(rx::Relaxer, x::AbstractVector, b::AbstractVector; sweeps::Int=1)

which executes `sweeps` relaxation steps in place on `Ax = b` (or the nonlinear
analog).

Note: a relaxer wraps a fixed operator A. To relax with a different operator,
construct a new relaxer. This mirrors `helmholtz.solve.smoothing.{Gs,Kaczmarz}Relaxer`.

Subtypes are the *only* place that knows the problem domain (linear vs.
constrained max-flow vs. future). The cycle code (`solve_cycle.jl`) is
oblivious — it always calls `relax!(lv.relaxer, x, b; sweeps=…)`.

FAS extension point — coarse levels may need to update internal state
(e.g. τ-correct box bounds) before the coarse recursion. The cycle calls

    update_fas!(coarse_relaxer, fine_relaxer, fine_x, P, T)

with a default no-op for linear relaxers. Max-flow's relaxer overrides
it to refresh its `low`/`high` from the current fine iterate.
"""
abstract type Relaxer end

"""
    residual!(r, A, x, b) -> r

Compute `r = b - A*x` in a SINGLE pass over `A` (CSC column scatter), fusing the
mat-vec and the `b`-subtraction. Replaces the two-pass `mul!(r,A,x); @. r = b - r`
(which streams the length-n vector `r` twice) with one pass — a memory-bandwidth
reduction (Brandt 1984 Guide §8.7 "wave-like one pass"; Barkai–Brandt 1983 fused
residual). The result equals `b - A*x` up to floating-point REASSOCIATION (the
subtraction order differs from accumulate-then-subtract), so it is convergence-
equivalent but NOT bit-identical to the two-pass form — gated by the convergence
oracle, not the bit-identical one.
"""
# Parallelize kernels only above this size — below it, thread-spawn overhead dominates
# (the cycle's many tiny coarse levels stay serial). nthreads()>1 required.
const PARALLEL_FLOOR = 20_000
@inline _parallel(n::Int) = nthreads() > 1 && n >= PARALLEL_FLOOR

function residual!(r::AbstractVector, A::SparseMatrixCSC, x::AbstractVector,
                   b::AbstractVector)
    rows = A.rowval; vals = A.nzval; cp = A.colptr
    n = size(A, 2)
    if _parallel(n)
        # Row form (A symmetric ⇒ column i = row i): r[i] = b[i] − Σ_k A[i,k]x[k]. Each
        # thread writes a DISTINCT r[i] → race-free, no atomics/coloring. ~6× at 8 threads.
        @threads for i in 1:n
            s = b[i]
            @inbounds for k in cp[i]:(cp[i + 1] - 1)
                s -= vals[k] * x[rows[k]]
            end
            @inbounds r[i] = s
        end
        return r
    end
    @inbounds @simd for i in eachindex(r)
        r[i] = b[i]
    end
    @inbounds for j in 1:size(A, 2)
        xj = x[j]
        for k in cp[j]:(cp[j + 1] - 1)
            r[rows[k]] -= vals[k] * xj
        end
    end
    return r
end

"""
    relax_resid!(rx, x, r, A, b; sweeps=1) -> (x, r)

Relaxation that MAINTAINS the residual `r = b - A*x` in place, so the post-sweep
residual is produced for free — fusing the smoothing sweep and the residual into a
single pass over `A` (Brandt 1984 Guide §8.7; Barkai–Brandt 1983). Requires `r` to be
the current residual on entry.

Generic fallback (any relaxer): relax, then recompute `r` — same memory traffic as the
old two-step `relax!` + `residual!`, but a uniform call site for the cycle.
"""
function relax_resid!(rx::Relaxer, x::AbstractVector, r::AbstractVector,
                      A::SparseMatrixCSC, b::AbstractVector; sweeps::Int = 1)
    relax!(rx, x, b; sweeps = sweeps)
    residual!(r, A, x, b)
    return x, r
end
# (the GaussSeidelRelaxer-specialized relax_resid! is defined after the struct, below)

"""
    update_fas!(coarse::Relaxer, fine::Relaxer, fine_x, P, T) -> coarse

Cycle hook called by `pre_process!` BEFORE recursing into the coarse
level. Default: no-op. Override on relaxer subtypes that need to
recompute coarse state from the current fine iterate (e.g. FAS
τ-correction of inequality bounds in the max-flow path).
"""
update_fas!(coarse::Relaxer, fine::Relaxer, fine_x::AbstractVector,
            P, T) = nothing

"""
    GaussSeidelRelaxer(A; ω=1.0)

Forward Gauss-Seidel relaxation for `Ax = b`, with optional damping `ω`.

    x_i^{new} = x_i + ω * (b_i - sum_j A_ij x_j^{current}) / A_ii

Skips any row with `A_ii == 0` (isolated nodes / eliminated rows).
"""
struct GaussSeidelRelaxer{T<:Real} <: Relaxer
    A::SparseMatrixCSC{T,Int}
    At::SparseMatrixCSC{T,Int}      # transpose, for fast row access
    ω::T
    diagidx::Vector{Int}            # diagidx[i] = nzval index of A[i,i] in column i of At; 0 if absent
    colors::Vector{Vector{Int}}     # greedy graph coloring (lazy; for parallel multicolor GS). empty = uncomputed
end
function GaussSeidelRelaxer(A::SparseMatrixCSC; ω::Real = 1.0,
                           symmetric::Bool = false)
    # relax! walks Aᵀ's columns for sparse row access. For a SYMMETRIC matrix
    # (every LAMG level operator — Laplacian, Galerkin, Schur) Aᵀ==A, so share
    # A directly and skip the O(nnz) transpose copy (a top-3 setup cost).
    # diagidx caches the diagonal's position in each column so the relax! inner
    # loop is branch-free (no per-nonzero `j == i` test). (A reciprocal 1/diag was
    # measured and gave no speedup — GS is memory-latency-bound on the x[] gather,
    # not division-bound — so the bit-identical `s/d` division is kept.)
    At = symmetric ? A : copy(A')
    n = size(At, 2)
    diagidx = zeros(Int, n)
    rows = At.rowval
    @inbounds for i in 1:n
        for k in nzrange(At, i)
            if rows[k] == i
                diagidx[i] = k
                break
            end
        end
    end
    GaussSeidelRelaxer(A, At, convert(eltype(A), ω), diagidx, Vector{Int}[])
end

# Greedy graph coloring of A (symmetric): color[i] = smallest color not used by a neighbour
# (visiting in given order). Returns the partition into color classes (each a Vector of node
# ids). Same-color nodes are mutually NON-adjacent, so they can be relaxed in parallel without
# races. Cheap O(nnz). Cached on the relaxer (computed once, on first parallel relax).
function _color_classes(A::SparseMatrixCSC)
    n = size(A, 1); rows = rowvals(A)
    color = zeros(Int, n); used = Int[]
    @inbounds for i in 1:n
        empty!(used)
        for k in nzrange(A, i)
            j = rows[k]
            (j != i && color[j] != 0) && push!(used, color[j])
        end
        c = 1
        while c in used; c += 1; end
        color[i] = c
    end
    nc = maximum(color)
    classes = [Int[] for _ in 1:nc]
    @inbounds for i in 1:n; push!(classes[color[i]], i); end
    return classes
end

# Forward Gauss–Seidel. The diagonal index is precomputed (diagidx), so the inner
# loop is branch-free (no per-nonzero `j == i` test) and the diagonal value is a
# direct lookup. Arithmetic is IDENTICAL to the original (same traversal order,
# same `s/d` division, `ω·(s/d − x)` correction form) → bit-for-bit unchanged;
# the ω==1 path drops the `1.0*` (exact in IEEE) and is hoisted out of the i-loop.
function relax!(rx::GaussSeidelRelaxer{T}, x::AbstractVector,
                b::AbstractVector; sweeps::Int = 1) where {T<:Real}
    At = rx.At
    ω = rx.ω
    n = size(At, 1)
    rows = At.rowval; vals = At.nzval; cp = At.colptr
    diagidx = rx.diagidx
    @assert length(x) == n "x has wrong size"
    @assert length(b) == n "b has wrong size"
    for _ in 1:sweeps
        if isone(ω)
            @inbounds for i in 1:n
                di = diagidx[i]
                di == 0 && continue
                d = vals[di]
                iszero(d) && continue
                s = b[i]
                for k in cp[i]:di-1
                    s -= vals[k] * x[rows[k]]
                end
                for k in di+1:cp[i+1]-1
                    s -= vals[k] * x[rows[k]]
                end
                x[i] += (s / d - x[i])
            end
        else
            @inbounds for i in 1:n
                di = diagidx[i]
                di == 0 && continue
                d = vals[di]
                iszero(d) && continue
                s = b[i]
                for k in cp[i]:di-1
                    s -= vals[k] * x[rows[k]]
                end
                for k in di+1:cp[i+1]-1
                    s -= vals[k] * x[rows[k]]
                end
                x[i] += ω * (s / d - x[i])
            end
        end
    end
    return x
end

# Residual-form forward Gauss–Seidel (the MATLAB-port form): one column scatter per sweep,
# maintaining r. For column i (A symmetric ⇒ column i = row i):
#   δ = ω·r_i/A_ii ;  x_i += δ ;  r_k -= A_ki·δ  for all k in column i (k=i ⇒ r_i←(1-ω)r_i).
# Produces the IDENTICAL GS iterate to `relax!` (x_i += ω(s/d − x_i) with the same latest
# neighbour values) up to FP reassociation — convergence-equivalent, not bit-identical. The
# post-sweep `r` IS the exact residual, so the cycle needs no separate residual pass: the
# smoothing sweep and the residual are fused into ONE pass over A (the cache-cliff lever).
function relax_resid!(rx::GaussSeidelRelaxer{T}, x::AbstractVector, r::AbstractVector,
                      A::SparseMatrixCSC, b::AbstractVector; sweeps::Int = 1) where {T<:Real}
    M = rx.A; ω = rx.ω; n = size(M, 1)
    rows = M.rowval; vals = M.nzval; cp = M.colptr; diagidx = rx.diagidx
    @assert length(x) == n && length(r) == n
    if _parallel(n)
        return relax_multicolor!(rx, x, r, b; sweeps = sweeps)
    end
    for _ in 1:sweeps
        if isone(ω)
            @inbounds for i in 1:n
                di = diagidx[i]
                di == 0 && continue
                d = vals[di]
                iszero(d) && continue
                δ = r[i] / d
                x[i] += δ
                for k in cp[i]:(cp[i + 1] - 1)
                    r[rows[k]] -= vals[k] * δ
                end
            end
        else
            @inbounds for i in 1:n
                di = diagidx[i]
                di == 0 && continue
                d = vals[di]
                iszero(d) && continue
                δ = ω * (r[i] / d)
                x[i] += δ
                for k in cp[i]:(cp[i + 1] - 1)
                    r[rows[k]] -= vals[k] * δ
                end
            end
        end
    end
    return x, r
end

# Parallel MULTICOLOR Gauss–Seidel (shared-memory): relax color-by-color; within each color
# the nodes are mutually NON-ADJACENT, so they update in parallel with NO race (each thread
# writes a distinct x[i]; neighbours are other colors, not written in this color sweep). This
# is GS with a color ordering → it PRESERVES the GS smoothing factor (Brandt's preferred
# parallel smoother; Guide §3.6, Barkai–Brandt RB-GS), unlike block-Jacobi which freezes
# couplings and degrades μ on scale-free graphs. r resynced by one parallel residual pass.
# Coloring cached on first call. Convergence ≈ serial GS (cycle counts essentially unchanged).
function relax_multicolor!(rx::GaussSeidelRelaxer{T}, x::AbstractVector, r::AbstractVector,
                           b::AbstractVector; sweeps::Int = 1) where {T<:Real}
    A = rx.A; ω = rx.ω
    rows = A.rowval; vals = A.nzval; cp = A.colptr; diagidx = rx.diagidx
    isempty(rx.colors) && append!(rx.colors, _color_classes(A))   # lazy, cached
    for _ in 1:sweeps
        for cls in rx.colors
            @threads for ci in 1:length(cls)
                @inbounds begin
                    i = cls[ci]
                    di = diagidx[i]
                    di == 0 && continue
                    d = vals[di]
                    iszero(d) && continue
                    s = b[i]
                    for k in cp[i]:(cp[i + 1] - 1)
                        k == di && continue
                        s -= vals[k] * x[rows[k]]
                    end
                    x[i] += ω * (s / d - x[i])
                end
            end
        end
    end
    residual!(r, A, x, b)                                 # resync r (parallel row-form)
    return x, r
end

"""
    JacobiRelaxer(A; ω=2/3)

Damped Jacobi relaxation. `ω=2/3` is the optimal smoothing weight for the
1D Laplacian; `ω=4/5` is closer to optimal in 2D.
"""
struct JacobiRelaxer{T<:Real} <: Relaxer
    A::SparseMatrixCSC{T,Int}
    d::Vector{T}        # cached diagonal
    ω::T
end
function JacobiRelaxer(A::SparseMatrixCSC; ω::Real = 2/3)
    JacobiRelaxer(A, collect(diag(A)), convert(eltype(A), ω))
end

function relax!(rx::JacobiRelaxer{T}, x::AbstractVector,
                b::AbstractVector; sweeps::Int = 1) where {T<:Real}
    A = rx.A
    d = rx.d
    ω = rx.ω
    n = size(A, 1)
    r = similar(x)
    for _ in 1:sweeps
        mul!(r, A, x)
        @. r = b - r
        @inbounds for i in 1:n
            if d[i] != 0
                x[i] += ω * r[i] / d[i]
            end
        end
    end
    return x
end

# No-op FAS-elimination hook for linear relaxers. (The max-flow companion defines a
# specialised method; on the linear LAMG+ path this is a no-op.)
update_fas_elim!(::Relaxer, ::AbstractVector, ::AbstractVector) = nothing
