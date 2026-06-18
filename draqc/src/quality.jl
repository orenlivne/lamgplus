# Aggregate quality measure μ(G) and the cheap quality tests
# (Napov–Notay 2017, §3 Thm 3.2/3.3 and §4.2 eqs. 14–16).

"""
    aggregate_quality_matrices(A, G, δ) -> (A_G, γ, X_G)

Assemble the quality matrices for aggregate `G` (Napov–Notay Thm 3.2):
`A_G` induced-subgraph Laplacian, `γ_p = δ_{G[p]} + 2·ext_p` the diagonal of
`Γ_G`, and `X_G = A_G + diag(γ)`.
"""
function aggregate_quality_matrices(A::SparseMatrixCSC, G::AbstractVector{<:Integer}, δ::AbstractVector)
    AG, ext = subgraph_laplacian(A, G)
    γ = [δ[Int(G[p])] + 2 * ext[p] for p in 1:length(G)]
    XG = AG + Diagonal(γ)
    return AG, γ, XG
end

"""
    mu_exact(A_G, X_G) -> μ

Exact aggregate quality μ(G) = 1/λ₂, where λ₂ is the second-smallest
generalized eigenvalue of `A_G z = λ X_G z` (Napov–Notay Thm 3.3; the smallest
is 0 with eigenvector 1). Returns 1.0 for a singleton.
"""
function mu_exact(AG::AbstractMatrix, XG::AbstractMatrix)
    ng = size(AG, 1)
    ng == 1 && return 1.0
    λ = eigvals(Symmetric(Matrix(AG)), Symmetric(Matrix(XG)))
    λ2 = sort(real.(λ))[2]
    return 1.0 / λ2
end

"""
    quality_ok_factor(A_G, X_G, κ̄) -> Bool

Test `μ(G) ≤ κ̄` via nonneg-definiteness of
`Z_G = κ̄ A_G − X_G(I − 1(1ᵀX_G1)⁻¹ 1ᵀX_G)`  (Napov–Notay eq. 14).
`Z_G` is symmetric and singular with null vector 1; it is nonneg-definite iff its
leading `(|G|−1)` principal block is positive definite, checked by Cholesky.
"""
function quality_ok_factor(AG::AbstractMatrix, XG::AbstractMatrix, κbar::Real)
    ng = size(AG, 1)
    ng == 1 && return true
    o = ones(ng)
    XGo = XG * o                          # X_G 1  (= (1ᵀX_G)ᵀ, X_G symmetric)
    s = dot(o, XGo)                       # 1ᵀ X_G 1
    # X_G(I − 1 s⁻¹ 1ᵀ X_G) = X_G − (X_G1)(X_G1)ᵀ / s
    ZG = κbar .* Matrix(AG) .- (Matrix(XG) .- (XGo * XGo') ./ s)
    ZG = (ZG + ZG') ./ 2                  # symmetrize away round-off
    block = ZG[1:ng-1, 1:ng-1]
    c = cholesky(Symmetric(block); check = false)
    return issuccess(c)
end

"""
    criterion16(A, G, root, δ, κ̄) -> Bool

Cheap sufficient condition `μ(G) < κ̄` of Napov–Notay eq. 16:
`(2 Σ_{k∉G}|a_jk| + δ_j) / |a_{jr}| ≤ κ̄ − 1` for every non-root `j ∈ G`,
with `r` the root. Returns false if any non-root vertex is not connected to the
root (`a_{jr} = 0`), since then the bound does not apply.
"""
function criterion16(A::SparseMatrixCSC, G::AbstractVector{<:Integer}, root::Integer,
                     δ::AbstractVector, κbar::Real)
    AG, ext = subgraph_laplacian(A, G)
    pos = Dict{Int,Int}(Int(G[p]) => p for p in 1:length(G))
    r = pos[Int(root)]
    for p in 1:length(G)
        G[p] == root && continue
        ajr = -AG[p, r]                   # |a_{jr}|  (root-to-j internal weight)
        ajr <= 0 && return false          # not connected to root ⇒ bound void
        lhs = (2 * ext[p] + δ[Int(G[p])]) / ajr
        lhs > κbar - 1 && return false
    end
    return true
end
