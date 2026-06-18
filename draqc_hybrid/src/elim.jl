# LAMG-style low-degree elimination (degree ≤ d_max, vs DRA-QC's degree-1 only).
#
# A maximal independent set F of nodes with degree ≤ d_max is eliminated exactly:
# because F is independent, L_FF is diagonal and the Schur complement onto C is exact
# (no approximation). On regular grids this removes ~half the nodes (interior degree-4
# nodes) for free before aggregation — Livne–Brandt LAMG §3.2.

"""
    ElimData

Back-substitution data for one low-degree elimination: eliminated set `F`, kept set
`C`, the diagonal `dff` of `L_FF`, and `LFC = L[F,C]`.
"""
struct ElimData
    F::Vector{Int}
    C::Vector{Int}
    dff::Vector{Float64}
    LFC::SparseMatrixCSC{Float64,Int}
    n::Int
end

"""
    eliminate_lowdeg(L; dmax=4) -> (ed::ElimData, Lc)

Pick a maximal independent set `F` of degree-≤`dmax` nodes and return the exact
Schur complement `Lc = L_CC − L_CF L_FF⁻¹ L_FC` (a graph Laplacian) plus the
back-substitution data.
"""
function eliminate_lowdeg(L::SparseMatrixCSC; dmax::Int=4)
    n = size(L, 1); rv = rowvals(L)
    deg = [count(k -> rv[k] != j, nzrange(L, j)) for j in 1:n]
    inF = falses(n); blocked = falses(n)        # blocked = in F or adjacent to an F node
    @inbounds for j in 1:n
        (deg[j] <= dmax && !blocked[j]) || continue
        inF[j] = true; blocked[j] = true
        for idx in nzrange(L, j)
            k = rv[idx]; k != j && (blocked[k] = true)
        end
    end
    F = findall(inF); C = findall(!, inF)
    dff = diag(L)[F]                            # L_FF is diagonal (F independent)
    LFC = L[F, C]
    Lc = L[C, C] - LFC' * Diagonal(1.0 ./ dff) * LFC
    Lc = (Lc + Lc') / 2
    return ElimData(F, C, dff, LFC, n), Lc
end

"""
    backsub(ed, bF, φC) -> φF

Recover the eliminated unknowns: `φ_F = L_FF⁻¹ (b_F − L_FC φ_C)`.
"""
backsub(ed::ElimData, bF::AbstractVector, φC::AbstractVector) = (bF .- ed.LFC * φC) ./ ed.dff

"""
    hybrid_elim(L, b; dmax=4, tol=1e-8, κbar=10.0, τ=0.05, maxcoarse=100, maxiter=2000)
        -> (φ, info, sizes)

Eliminate degree-≤`dmax` nodes, solve the Schur complement with the SoC hybrid, and
back-substitute. `dmax=1` reproduces DRA-QC's elimination; `dmax=4` is the LAMG variant.
"""
function hybrid_elim(L::SparseMatrixCSC, b::AbstractVector; dmax::Int=4, tol::Real=1e-8,
                     κbar::Real=10.0, τ::Real=0.05, maxcoarse::Int=100, maxiter::Int=2000,
                     caliber2::Bool=false)
    ed, Lc = eliminate_lowdeg(L; dmax = dmax)
    bF = b[ed.F]; bC = b[ed.C]
    bc = bC - ed.LFC' * (bF ./ ed.dff)         # reduced RHS (zero-mean, consistent)
    h = hybrid_setup(Lc; κbar = κbar, τ = τ, maxcoarse = maxcoarse, caliber2 = caliber2)
    φC, info = hybrid_solve(h, bc; tol = tol, maxiter = maxiter)
    φF = backsub(ed, bF, φC)
    φ = zeros(ed.n); φ[ed.F] = φF; φ[ed.C] = φC
    φ .-= sum(φ) / ed.n
    return φ, info, (nF = length(ed.F), nC = length(ed.C))
end
