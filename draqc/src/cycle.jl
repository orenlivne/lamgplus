# K-cycle with symmetrized Gauss–Seidel smoothing — Algorithm 1 applied recursively
# with a 2-iteration FCG(1) coarse solve (Napov–Notay 2017, §2–3, §5.2).
#
# Fully preallocated / in-place: every level owns scratch buffers, so a solve does
# no heap allocation in its inner loops (mul!/ldiv!/axpy only).

"""
    DRAQCSolver(h)

Precompute per-level smoother factors (lower/upper triangles of each `A_ℓ`), a dense
pseudo-inverse of the coarsest operator, and all scratch buffers for an allocation-free
in-place solve.
"""
struct DRAQCSolver
    h::DRAQCHierarchy
    L::Vector{LowerTriangular{Float64,SparseMatrixCSC{Float64,Int}}}
    U::Vector{UpperTriangular{Float64,SparseMatrixCSC{Float64,Int}}}
    coarse::Matrix{Float64}
    # precond scratch (per level, size n_ℓ)
    s1::Vector{Vector{Float64}}; s2::Vector{Vector{Float64}}; s3::Vector{Vector{Float64}}
    # FCG scratch (per level): r,z,p,Ap and the previous p,Ap; plus parent interface rc/ec
    cr::Vector{Vector{Float64}}; cz::Vector{Vector{Float64}}
    cp::Vector{Vector{Float64}}; cAp::Vector{Vector{Float64}}
    cpp::Vector{Vector{Float64}}; cApp::Vector{Vector{Float64}}
    fr::Vector{Vector{Float64}}; fx::Vector{Vector{Float64}}
end

function DRAQCSolver(h::DRAQCHierarchy)
    nlev = num_levels(h)
    L = [LowerTriangular(tril(h.A[ℓ])) for ℓ in 1:nlev-1]
    U = [UpperTriangular(triu(h.A[ℓ])) for ℓ in 1:nlev-1]
    coarse = pinv(Matrix(h.A[end]))
    mk() = [zeros(size(h.A[ℓ], 1)) for ℓ in 1:nlev]
    return DRAQCSolver(h, L, U, coarse,
                       mk(), mk(), mk(),
                       mk(), mk(), mk(), mk(), mk(), mk(),
                       mk(), mk())
end

"""
    precond_apply!(s, ℓ, r, out) -> out

One multigrid preconditioner application at level `ℓ` (`out ≈ A_ℓ⁻¹ r`), Algorithm 1:
forward-GS presmoothing, restriction, coarse correction (2 FCG(1) iterations — the
K-cycle — when `n_{ℓ+1} > n_ℓ^{1/3}`, else one recursive application; direct at the
coarsest), prolongation, backward-GS postsmoothing. Fully in-place.
"""
function precond_apply!(s::DRAQCSolver, ℓ::Int, r::AbstractVector, out::AbstractVector)
    nlev = num_levels(s.h)
    if ℓ == nlev
        mul!(out, s.coarse, r)
        return out
    end
    A = s.h.A[ℓ]; P = s.h.P[ℓ]
    v1 = s.s1[ℓ]; rt = s.s2[ℓ]; v2 = s.s3[ℓ]
    ldiv!(v1, s.L[ℓ], r)                      # v1 = tril(A)⁻¹ r   (forward GS)
    copyto!(rt, r); mul!(rt, A, v1, -1.0, 1.0) # rt = r − A v1
    rc = s.fr[ℓ+1]; ec = s.fx[ℓ+1]
    mul!(rc, transpose(P), rt)                # restrict: rc = Pᵀ rt
    if ℓ + 1 == nlev
        precond_apply!(s, ℓ + 1, rc, ec)
    elseif size(s.h.A[ℓ+1], 1) > size(A, 1)^(1/3)
        fcg_solve!(s, ℓ + 1, rc, ec, 2, 0.0) # K-cycle: 2 FCG(1) iterations
    else
        precond_apply!(s, ℓ + 1, rc, ec)
    end
    mul!(v2, P, ec)                           # prolong
    copyto!(out, v1); out .+= v2              # out = v1 + v2
    mul!(rt, A, v2, -1.0, 1.0)                # rt = rt − A v2  (= r̄)
    ldiv!(v1, s.U[ℓ], rt)                     # v3 = triu(A)⁻¹ r̄ (backward GS), reuse v1
    out .+= v1                                # out += v3
    return out
end

"""
    fcg_solve!(s, ℓ, b, x, maxiter, tol) -> iters

FCG(1) solving `A_ℓ x = b` into `x`, preconditioned by `precond_apply!(s,ℓ,·,·)`,
using level-ℓ scratch buffers (no allocation). With `maxiter=2, tol=0` this is the
K-cycle inner solve; at `ℓ=1` with a real `tol` it is the outer solver. Returns the
number of iterations performed.
"""
function fcg_solve!(s::DRAQCSolver, ℓ::Int, b::AbstractVector, x::AbstractVector,
                    maxiter::Int, tol::Real)
    A = s.h.A[ℓ]
    r = s.cr[ℓ]; z = s.cz[ℓ]; p = s.cp[ℓ]; Ap = s.cAp[ℓ]; pp = s.cpp[ℓ]; App = s.cApp[ℓ]
    fill!(x, 0.0); copyto!(r, b)              # x=0 ⇒ r = b
    nb = norm(b); nb == 0 && (nb = 1.0)
    βden = 0.0; have_prev = false; iters = 0
    for i in 1:maxiter
        tol > 0 && norm(r) / nb <= tol && break
        precond_apply!(s, ℓ, r, z)
        if !have_prev
            copyto!(p, z)
        else
            β = dot(z, App) / βden
            @. p = z - β * pp
        end
        mul!(Ap, A, p)
        pAp = dot(p, Ap); pAp == 0 && break
        α = dot(p, r) / pAp
        @. x += α * p
        @. r -= α * Ap
        p, pp = pp, p; Ap, App = App, Ap     # rotate current → previous
        βden = dot(pp, App)
        have_prev = true
        iters = i
    end
    return iters
end
