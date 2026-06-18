# Caliber-2 interpolation for the aggregation framework (the LAMG+ idea adapted).
#
# DRA-QC uses caliber-1 (piecewise-constant, 0/1) prolongation, which recovers only
# half the amplitude of a smooth-along-a-chain mode and pins the two-level factor at
# ρ≈0.5. Here a fine node whose *strong* edges reach exactly two coarse aggregates
# {a,b} interpolates from both, P[u,a]=c_a/(c_a+c_b), P[u,b]=c_b/(c_a+c_b), with c_g
# its total strong coupling to aggregate g — operator-dependent linear interpolation
# that recovers the smooth mode along a semicoarsened chain. Stays lean (≤2 nz/row),
# parameter-free, and constant-preserving (rows sum to 1). Nodes reaching one (interior)
# or >2 aggregates stay caliber-1.

"""
    lump_positive_offdiag!(Ac) -> Ac

Move positive off-diagonal entries into the diagonal (symmetrically), restoring a
graph Laplacian. A weighted (caliber-2) Galerkin operator can have positive
off-diagonals; lumping preserves symmetry and zero row-sums so the coarse-level
quality control stays valid.
"""
function lump_positive_offdiag!(Ac::SparseMatrixCSC)
    n = size(Ac, 1); rv = rowvals(Ac); nz = nonzeros(Ac); add = zeros(n)
    @inbounds for j in 1:n, idx in nzrange(Ac, j)
        i = rv[idx]; v = nz[idx]
        if i != j && v > 0
            add[i] += v; nz[idx] = 0.0
        end
    end
    @inbounds for j in 1:n, idx in nzrange(Ac, j)
        rv[idx] == j && (nz[idx] += add[j])
    end
    dropzeros!(Ac)
    # a node whose off-diagonals were all positive becomes isolated with a zero
    # diagonal after lumping; give it a unit diagonal so the GS smoother stays
    # nonsingular (the coarse variable is then harmlessly decoupled).
    d = diag(Ac)
    @inbounds for i in 1:n
        d[i] <= 1e-300 && (Ac[i, i] = 1.0)
    end
    return Ac
end

"""
    caliber2_prolongation(A, agg, nc; τ=0.05) -> (A_c, P)

Build the caliber-2 prolongation `P` and Galerkin coarse operator `A_c = Pᵀ A P`,
with positive off-diagonals lumped so `A_c` is a graph Laplacian.
"""
function caliber2_prolongation(A::SparseMatrixCSC, agg::AbstractVector{<:Integer}, nc::Integer; τ::Real=0.05)
    n = size(A, 1); mw = max_incident(A); rv = rowvals(A); nz = nonzeros(A)
    I = Int[]; J = Int[]; V = Float64[]
    coup = Dict{Int,Float64}()
    for u in 1:n
        a = Int(agg[u]); thr = τ * mw[u]
        empty!(coup)
        @inbounds for idx in nzrange(A, u)
            k = rv[idx]; k == u && continue
            w = -nz[idx]
            w >= thr || continue                       # strong edges only
            g = Int(agg[k]); coup[g] = get(coup, g, 0.0) + w
        end
        haskey(coup, a) || (coup[a] = 0.0)             # u always belongs to its own aggregate
        if length(coup) == 2
            b = first(k for k in keys(coup) if k != a)
            ca = coup[a]; cb = coup[b]; s = ca + cb
            if s > 0
                push!(I, u); push!(J, a); push!(V, ca / s)
                push!(I, u); push!(J, b); push!(V, cb / s)
            else
                push!(I, u); push!(J, a); push!(V, 1.0)
            end
        else
            push!(I, u); push!(J, a); push!(V, 1.0)     # caliber-1 (interior or >2 aggregates)
        end
    end
    P = sparse(I, J, V, n, nc)
    Ac = P' * A * P; Ac = (Ac + Ac') / 2
    lump_positive_offdiag!(Ac)
    return Ac, P
end
