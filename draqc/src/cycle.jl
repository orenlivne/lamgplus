# K-cycle with symmetrized Gauss–Seidel smoothing — Algorithm 1 applied
# recursively with a 2-iteration FCG(1) coarse solve (Napov–Notay 2017, §2–3, §5.2).

"""
    DRAQCSolver(h)

Precompute the per-level smoother factors (lower/upper triangles of each `A_ℓ`)
and a dense pseudo-inverse of the coarsest operator, from a `DRAQCHierarchy`.
"""
struct DRAQCSolver
    h::DRAQCHierarchy
    L::Vector{SparseMatrixCSC{Float64,Int}}   # tril(A_ℓ) = D+strictL, levels 1..end-1
    U::Vector{SparseMatrixCSC{Float64,Int}}   # triu(A_ℓ) = D+strictU
    coarse::Matrix{Float64}                    # pinv of the coarsest A
end

function DRAQCSolver(h::DRAQCHierarchy)
    nlev = num_levels(h)
    L = [tril(h.A[ℓ]) for ℓ in 1:nlev-1]
    U = [triu(h.A[ℓ]) for ℓ in 1:nlev-1]
    coarse = pinv(Matrix(h.A[end]))            # coarsest is small; min-norm solve
    return DRAQCSolver(h, L, U, coarse)
end

"""
    precond_apply(s, ℓ, r) -> v

One application of the multigrid preconditioner at level `ℓ` (≈ `A_ℓ⁻¹ r`),
Algorithm 1: forward-GS presmoothing, restriction, coarse correction, prolongation,
backward-GS postsmoothing. The coarse correction approximately solves `A_{ℓ+1}`
with 2 FCG(1) iterations (the K-cycle) when coarsening is strong enough
(`n_{ℓ+1} > n_ℓ^{1/3}`), else a single recursive application; the coarsest level is
solved directly.
"""
function precond_apply(s::DRAQCSolver, ℓ::Int, r::AbstractVector)
    nlev = num_levels(s.h)
    ℓ == nlev && return s.coarse * r
    A = s.h.A[ℓ]; P = s.h.P[ℓ]
    v1 = LowerTriangular(s.L[ℓ]) \ r                  # forward GS presmoothing
    rt = r - A * v1                                    # residual update
    rc = P' * rt                                       # restriction (sum over aggregate)
    if ℓ + 1 == nlev
        ec = precond_apply(s, ℓ + 1, rc)              # direct coarsest solve
    else
        nfine = size(A, 1); ncoarse = size(s.h.A[ℓ+1], 1)
        if ncoarse > nfine^(1/3)
            ec, _, _ = fcg1(s.h.A[ℓ+1], rc, w -> precond_apply(s, ℓ+1, w); tol=0.0, maxiter=2)
        else
            ec = precond_apply(s, ℓ + 1, rc)
        end
    end
    v2 = P * ec                                        # prolongation
    rb = rt - A * v2                                    # residual update
    v3 = UpperTriangular(s.U[ℓ]) \ rb                  # backward GS postsmoothing
    return v1 + v2 + v3
end
