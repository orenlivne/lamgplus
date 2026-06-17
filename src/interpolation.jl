"""
    piecewise_constant_interpolation(aggregate::AbstractVector{Int}) -> P, R, Q

Build the LAMG caliber-1 transfer operators from a coarse-aggregate
assignment.

`aggregate[i]` is the coarse index (1..n_c) of fine node `i`.

Returns three sparse matrices:
- `P :: n × n_c`  — interpolation, piecewise-constant. `P[i, J] = 1` if fine
                    node `i` belongs to aggregate `J`, else 0.
- `R :: n_c × n`  — coarsening, `R[J, i] = 1 / |aggregate J|` averaging.
                    Mathematically `R = (PᵀP)⁻¹ Pᵀ`, the Moore-Penrose
                    pseudo-inverse of `P`.
- `Q :: n_c × n`  — restriction, `Q = Pᵀ` (sum over aggregate).

This is the structure-preserving choice for a graph Laplacian:
- `P · 1_c = 1_f` (preserves constants) ⇒ coarse Laplacian retains zero row sum.
- `P` has nonneg entries (in fact 0/1) ⇒ Galerkin `PᵀAP` has nonneg off-diagonal sums of fine-Laplacian off-diagonals (which are ≤ 0) ⇒ coarse weights ≥ 0.

See `doc/architecture.md` §"Why piecewise-constant P".
"""
function piecewise_constant_interpolation(aggregate::AbstractVector{Int})
    n = length(aggregate)
    n_c = maximum(aggregate)
    @assert minimum(aggregate) >= 1 "aggregate indices must be ≥ 1"
    @assert n_c >= 1 "must have at least one aggregate"

    # P: column J holds the indicator of aggregate J.
    p_rows = collect(1:n)
    p_cols = collect(aggregate)
    p_vals = ones(Float64, n)
    P = sparse(p_rows, p_cols, p_vals, n, n_c)

    # Q = Pᵀ
    Q = sparse(P')

    # R = (PᵀP)^{-1} Pᵀ. PᵀP is diagonal with entry J = |aggregate J|.
    sizes = zeros(Int, n_c)
    for a in aggregate
        sizes[a] += 1
    end
    @assert all(sizes .> 0) "every aggregate must contain at least one fine node"
    R_vals = [1.0 / sizes[aggregate[i]] for i in 1:n]
    R_rows = collect(aggregate)
    R_cols = collect(1:n)
    R = sparse(R_rows, R_cols, R_vals, n_c, n)

    return P, R, Q
end

"""
    galerkin_coarse_operator(A, P) -> SparseMatrixCSC
    galerkin_coarse_operator(A, P, Q) -> SparseMatrixCSC

Returns the Galerkin coarse operator `A^c = Pᵀ A P` as a canonical `SparseMatrixCSC`
(sorted rows, no stored zeros) — callers must NOT re-wrap it in `sparse(...)`.

The 3-argument form reuses an already-materialized restriction `Q = sparse(Pᵀ)`
(both interpolation builders return it) instead of re-materializing the lazy
adjoint `P'` inside the product. Since `Q === sparse(P')` entry-for-entry,
`Q * A * P` is bit-identical to `P' * A * P` (same nonzero pattern, same float
accumulation order) while saving the transient transpose allocation.
"""
galerkin_coarse_operator(A::SparseMatrixCSC, P::SparseMatrixCSC) = P' * A * P
galerkin_coarse_operator(A::SparseMatrixCSC, P::SparseMatrixCSC, Q::SparseMatrixCSC) =
    Q * A * P

"""
    sparsify_lump(A, tol) -> SparseMatrixCSC

Non-Galerkin sparsification: drop weak off-diagonal couplings of the (symmetric,
Laplacian-like) coarse operator and LUMP each dropped weight onto both incident
diagonals. An edge (i,j) is dropped when `|A[i,j]| < tol·sqrt(|A[i,i]|·|A[j,j]|)`.
Lumping `A[i,i]+=A[i,j]`, `A[j,j]+=A[i,j]` preserves every row sum, so the constant
stays in the null space (the result is still a graph Laplacian — the coarse graph with
its weakest edges removed). Reduces operator complexity on scale-free / densifying
coarse levels at the cost of a slightly weaker coarse correction. `tol≤0` is a no-op.
"""
function sparsify_lump(A::SparseMatrixCSC, tol::Real)
    tol <= 0 && return A
    n = size(A, 1)
    d = diag(A)
    rows = rowvals(A); vals = nonzeros(A)
    dlump = collect(float.(d))
    cap = nnz(A)
    I = Int[]; J = Int[]; V = Float64[]; sizehint!(I, cap); sizehint!(J, cap); sizehint!(V, cap)
    @inbounds for j in 1:n
        for k in nzrange(A, j)
            i = rows[k]
            i < j || continue                       # process each off-diagonal edge once
            v = vals[k]
            if abs(v) < tol * sqrt(abs(d[i]) * abs(d[j]))
                dlump[i] += v; dlump[j] += v          # drop & lump (row sums preserved)
            else
                push!(I, i); push!(J, j); push!(V, v)
                push!(I, j); push!(J, i); push!(V, v) # keep, symmetric
            end
        end
    end
    @inbounds for i in 1:n
        push!(I, i); push!(J, i); push!(V, dlump[i])
    end
    return sparse(I, J, V, n, n)
end

"""
    caliber2_interpolation(aggregate, X, A; τ=0.5, δ=1e-3) -> P, R, Q, n_upgraded

Caliber-1 piecewise-constant interpolation with a SELECTIVE caliber-2 upgrade on
locally one-dimensional (anisotropic) fine nodes.

Inexact elimination of near-diagonally-dominant `A_FF` blocks either fills in the
coarse operator or, as an M/N splitting, fails to converge (the LAMG paper's
near-DD `A_FF` route). Instead we raise the interpolation caliber where — and only
where — it is cheap and safe: a fine node whose STRONG edges reach exactly **two**
coarse aggregates sits on a 1-D strong "line", so it gets a second parent. This
fixes the caliber-1 energy-ratio ceiling (ρ≈0.5 → ρ≈0.12 on the 1-D pockets) while
staying caliber-1 — zero extra fill — everywhere the neighborhood is higher-dimensional.

Per fine node `i` (non-seed): collect the distinct aggregates reached through strong
edges `|A_ij| ≥ τ·max(rowmax_i, rowmax_j)`. If exactly two, `{Aᵍ, Bᵍ}`, fit the single
weight `w` (parents `w, 1−w`, summing to 1 so constants interpolate exactly) by
smoothness-weighted least squares over the test vectors `X`
(`w = Σ ω_k (a_k−b_k)(x_i^k−b_k) / Σ ω_k (a_k−b_k)²`, `ω_k = ‖x_k‖²/‖A x_k‖²`).
Guard: reject `w ∉ [0,1]` (extrapolation off the line → wrong-sign coarse coupling /
M-matrix loss) and fall back to caliber-1; snap `w≈0/1` to caliber-1 to keep `P` sparse.

`X` are the SAME test vectors used for affinity (reuse, no extra setup cost). Returns
`(P, R, Q, n_upgraded)`; `R` is injection at the aggregate seeds, so `R·P = I` exactly
without forming `(PᵀP)⁻¹` (which is dense for caliber-2's non-diagonal `PᵀP`). This keeps
the restriction sparse and FAS-safe — a coarse function restricts to itself — which is
what lets caliber-2 generalize to the nonlinear max-flow FMG-FAS cycle, where `R` (unlike
the linear cycle) does not cancel.
"""
function caliber2_interpolation(aggregate::AbstractVector{Int}, X::AbstractMatrix,
                                A::SparseMatrixCSC; τ::Real = 0.5, δ::Real = 1e-3)
    n = length(aggregate)
    n_c = maximum(aggregate)
    @assert size(X, 1) == n "test-vector rows must match number of fine nodes"
    rows = rowvals(A); vals = nonzeros(A)

    # representative (seed) per aggregate = first (min-index) member
    seed = zeros(Int, n_c)
    @inbounds for i in 1:n
        a = aggregate[i]; seed[a] == 0 && (seed[a] = i)
    end
    isseed = falses(n); @inbounds for a in 1:n_c; isseed[seed[a]] = true; end

    # row-max off-diagonal magnitude (strength-of-connection denominator)
    rmax = zeros(n)
    @inbounds for j in 1:n, k in nzrange(A, j)
        i = rows[k]; i != j && (rmax[i] = max(rmax[i], abs(vals[k])))
    end

    # smoothness weights ω_k = ‖x_k‖² / ‖A x_k‖²  (bias LS toward smooth TVs)
    K = size(X, 2); ω = ones(Float64, K)
    for k in 1:K
        x = @view X[:, k]; Ax = A * x; nx = dot(x, x); nAx = dot(Ax, Ax)
        ω[k] = nAx > 1e-30 ? nx / nAx : 1.0
    end
    ωm = sum(ω) / K; ωm > 0 && (ω ./= ωm)

    Ip = Int[]; Jp = Int[]; Vp = Float64[]; n_up = 0
    S = Int[]
    @inbounds for i in 1:n
        if isseed[i]
            push!(Ip, i); push!(Jp, aggregate[i]); push!(Vp, 1.0); continue
        end
        empty!(S)
        for k in nzrange(A, i)
            r = rows[k]; r == i && continue
            if abs(vals[k]) >= τ * max(rmax[i], rmax[r])
                a = aggregate[r]; (a in S) || push!(S, a)
            end
        end
        if length(S) == 2                                   # locally 1-D: upgrade
            Ag, Bg = S[1], S[2]
            sa = seed[Ag]; sb = seed[Bg]
            den = 0.0; num = 0.0
            for k in 1:K
                d = X[sa, k] - X[sb, k]
                den += ω[k] * d * d; num += ω[k] * d * (X[i, k] - X[sb, k])
            end
            w = den > 1e-30 ? num / den : 1.0
            if w < 0 || w > 1                               # extrapolation → caliber-1
                push!(Ip, i); push!(Jp, aggregate[i]); push!(Vp, 1.0)
            elseif w <= δ
                push!(Ip, i); push!(Jp, Bg); push!(Vp, 1.0)
            elseif w >= 1 - δ
                push!(Ip, i); push!(Jp, Ag); push!(Vp, 1.0)
            else
                push!(Ip, i); push!(Jp, Ag); push!(Vp, w)
                push!(Ip, i); push!(Jp, Bg); push!(Vp, 1 - w); n_up += 1
            end
        else                                                # higher-dim or isolated → caliber-1
            push!(Ip, i); push!(Jp, aggregate[i]); push!(Vp, 1.0)
        end
    end

    P = sparse(Ip, Jp, Vp, n, n_c)
    Q = sparse(P')
    # R = injection at the aggregate seeds: R[J, seed_J] = 1. Seeds are caliber-1
    # (P[seed_J,·] = e_J), so R·P = I EXACTLY even though PᵀP is non-diagonal for
    # caliber-2 — no (PᵀP)⁻¹ is formed. This keeps R sparse and deterministic and is
    # FAS-safe: a coarse function restricts to itself, so the FAS τ-correction is exact.
    # (In the linear cycle R cancels; in the nonlinear max-flow FAS cycle it does not,
    # so R·P = I is what lets caliber-2 generalize to FMG-FAS.)
    @assert all(seed .> 0) "every aggregate must have a seed"
    R = sparse(collect(1:n_c), seed, ones(n_c), n_c, n)
    return P, R, Q, n_up
end

"""
    aggregation_from_partition(partition::Vector{Vector{Int}}, n::Int) -> Vector{Int}

Convert a list-of-lists partition (`partition[J]` = list of fine indices in
aggregate `J`) into the flat `aggregate[i] = J` representation. `n` is the
total number of fine nodes (used to detect missing assignments).
"""
function aggregation_from_partition(partition::Vector{Vector{Int}}, n::Int)
    agg = zeros(Int, n)
    for (J, members) in enumerate(partition)
        for i in members
            @assert agg[i] == 0 "node $i is assigned to multiple aggregates"
            agg[i] = J
        end
    end
    @assert all(agg .> 0) "some fine nodes are unassigned to any aggregate"
    return agg
end
