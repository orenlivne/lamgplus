# Outer solve and convenience entry points (Napov–Notay 2017 use FCG(1) with a
# zero initial guess and a relative-residual stopping criterion).

"""
    draqc_solve(s, b; tol=1e-8, maxiter=2000) -> (x, info)

Solve `A x = b` (`A = s.h.A[1]`, a graph Laplacian) with FCG(1) preconditioned by
the DRA-QC K-cycle. The RHS is projected to the range (zero mean) and the returned
`x` is zero-mean. `info` has fields `iters` and `relres`.
"""
function draqc_solve(s::DRAQCSolver, b::AbstractVector; tol::Real=1e-8, maxiter::Int=2000)
    n = length(b)
    bz = b .- sum(b) / n                       # project RHS to the range (zero mean)
    x = zeros(n)
    iters = fcg_solve!(s, 1, bz, x, maxiter, tol)
    x .-= sum(x) / n
    relres = norm(s.h.A[1] * x - bz) / max(norm(bz), 1e-30)
    return x, (iters = iters, relres = relres)
end

draqc_solve(h::DRAQCHierarchy, b::AbstractVector; kw...) = draqc_solve(DRAQCSolver(h), b; kw...)

"""
    draqc(A, b; tol=1e-8, κbar=10.0, maxcoarse=100) -> (x, info)

Convenience: build the DRA-QC hierarchy for `A` and solve `A x = b`.
"""
function draqc(A::SparseMatrixCSC, b::AbstractVector; tol::Real=1e-8, κbar::Real=10.0, maxcoarse::Int=100)
    h = draqc_setup(A; κbar=κbar, maxcoarse=maxcoarse)
    return draqc_solve(DRAQCSolver(h), b; tol=tol)
end
