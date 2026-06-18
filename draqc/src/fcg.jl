# Flexible Conjugate Gradient with truncation 1 — FCG(1) (Notay 2000; the outer
# solver and the K-cycle inner iteration of Napov–Notay 2017, §3 & §5.2).

"""
    fcg1(A, b, M; tol=1e-8, maxiter=1000, x0=nothing) -> (x, iters, relres)

FCG(1): preconditioned CG that tolerates a *variable* preconditioner `M`
(a callable `r -> M⁻¹r`) by A-orthogonalizing each search direction against the
previous one. Used both as the outer solver and, with `maxiter=2, tol=0`, as the
2-iteration inner coarse solve that makes the cycle a K-cycle.

`iters` counts iterations actually performed; `relres = ‖b − Ax‖/‖b‖`.
"""
function fcg1(A, b::AbstractVector, M; tol::Real=1e-8, maxiter::Int=1000, x0=nothing)
    n = length(b)
    x = x0 === nothing ? zeros(n) : copy(x0)
    r = b - A * x
    nb = norm(b); nb == 0 && (nb = 1.0)
    pprev = zeros(n); Apprev = zeros(n); have_prev = false
    iters = 0
    for i in 1:maxiter
        norm(r) / nb <= tol && break
        z = M(r)
        if !have_prev
            p = copy(z)
        else
            β = dot(z, Apprev) / dot(pprev, Apprev)
            p = z - β * pprev
        end
        Ap = A * p
        pAp = dot(p, Ap)
        pAp == 0 && break
        α = dot(p, r) / pAp
        @. x += α * p
        @. r -= α * Ap
        pprev, Apprev = p, Ap
        have_prev = true
        iters = i
    end
    return x, iters, norm(b - A * x) / nb
end
