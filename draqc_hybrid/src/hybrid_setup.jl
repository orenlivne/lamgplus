# Hybrid setup: DRA-QC quality control + complexity enhancement + K-cycle, but with
# SoC-vetoed aggregation. Reuses DRAQC's δ, refine_aggregate!, galerkin, hierarchy,
# solver, and solve unchanged — only the tentative-aggregate formation differs.

"""
    hybrid_partition(A; κbar=10.0, τ=0.05, maxdepth=8) -> (agg, nc)

Greedy DRA-QC partition (Napov–Notay Algorithm 3) with the LAMG strength-of-connection
veto applied at tentative-aggregate formation. Quality control (bad-vertex removal,
subgroup extraction) is DRAQC's, unchanged.
"""
function hybrid_partition(A::SparseMatrixCSC; κbar::Real=10.0, τ::Real=0.05, maxdepth::Int=8)
    n = size(A, 1)
    δ = DRAQC.delta_vector(A)
    mw = max_incident(A)
    rv = rowvals(A)
    deg = [count(k -> rv[k] != j, nzrange(A, j)) for j in 1:n]
    key = [deg[j] >= 1 ? floor(Int, log2(deg[j])) : -1 for j in 1:n]
    order = sortperm(key; rev = true)

    aggregated = falses(n); agg = zeros(Int, n)
    inS = falses(n); scratch = Int[]
    nc = 0; nleft = n
    while nleft > 0
        for r in order
            aggregated[r] && continue
            G = form_tentative_soc(A, r, aggregated, mw, τ)
            G = DRAQC.refine_aggregate!(A, G, r, δ, inS, scratch; κbar = κbar, maxdepth = maxdepth)
            nc += 1
            for v in G; aggregated[v] = true; agg[v] = nc; end
            nleft -= length(G)
        end
    end

    # complexity enhancement (steps 21–24): dissolve aggregates of size ≤3 and
    # re-aggregate the freed vertices greedily across strong edges (no QC).
    if nc > n / 4
        nz = nonzeros(A)
        sizes = zeros(Int, nc); for v in 1:n; sizes[agg[v]] += 1; end
        keep = sizes .>= 4
        newid = zeros(Int, nc); nk = 0
        for a in 1:nc; keep[a] && (nk += 1; newid[a] = nk); end
        freed = falses(n)
        for v in 1:n
            keep[agg[v]] ? (agg[v] = newid[agg[v]]) : (freed[v] = true; agg[v] = 0)
        end
        nc2 = 0
        for r in order
            (freed[r] && agg[r] == 0) || continue
            nc2 += 1; agg[r] = nk + nc2; thr = τ * mw[r]
            @inbounds for idx in nzrange(A, r)
                k = rv[idx]; k == r && continue
                (freed[k] && agg[k] == 0 && -nz[idx] >= thr) && (agg[k] = nk + nc2)
            end
        end
        nc = nk + nc2
    end
    return agg, nc
end

"""
    hybrid_setup(A; κbar=10.0, τ=0.05, maxcoarse=100, maxlevels=40) -> DRAQC.DRAQCHierarchy

Build the multilevel hierarchy with SoC-vetoed DRA-QC coarsening and DRAQC's
Galerkin operator. The returned hierarchy plugs directly into DRAQC's solver.
"""
function hybrid_setup(A::SparseMatrixCSC; κbar::Real=10.0, τ::Real=0.05,
                      maxcoarse::Int=100, maxlevels::Int=40)
    As = SparseMatrixCSC{Float64,Int}[A]
    Ps = SparseMatrixCSC{Float64,Int}[]
    while size(As[end], 1) > maxcoarse && length(As) < maxlevels
        agg, nc = hybrid_partition(As[end]; κbar = κbar, τ = τ)
        nc >= size(As[end], 1) && break
        Ac, P = DRAQC.galerkin(As[end], agg, nc)
        push!(Ps, P); push!(As, Ac)
    end
    return DRAQC.DRAQCHierarchy(As, Ps)
end

"""
    hybrid_solve(h, b; tol=1e-8, maxiter=2000) -> (x, info)

Solve with DRAQC's K-cycle/FCG on a SoC-built hierarchy.
"""
hybrid_solve(h::DRAQC.DRAQCHierarchy, b::AbstractVector; kw...) =
    DRAQC.draqc_solve(DRAQC.DRAQCSolver(h), b; kw...)

"""
    hybrid(A, b; tol=1e-8, κbar=10.0, τ=0.05, maxcoarse=100) -> (x, info)

Convenience: build the hybrid hierarchy and solve.
"""
function hybrid(A::SparseMatrixCSC, b::AbstractVector; tol::Real=1e-8, κbar::Real=10.0,
                τ::Real=0.05, maxcoarse::Int=100)
    h = hybrid_setup(A; κbar = κbar, τ = τ, maxcoarse = maxcoarse)
    return hybrid_solve(h, b; tol = tol)
end
