# Strength-of-connection (SoC) vetoed aggregation — the LAMG idea grafted onto
# DRA. The DRA root selection and quality control are unchanged; the only change
# is that a tentative aggregate may grow only across *strong* edges
# (|w_uv| ≥ τ·max_{t∼u}|w_ut|, the Ruge–Stüben/LAMG+ criterion). This supplies the
# directional information DRA lacks, so on grid-aligned anisotropy the aggregates
# align with the strong direction (semicoarsening) instead of spanning the weak one.

"""
    max_incident(A) -> mw

`mw[v]` = largest incident edge weight |a_{vk}| of vertex `v`.
"""
function max_incident(A::SparseMatrixCSC)
    n = size(A, 1); rv = rowvals(A); nz = nonzeros(A); mw = zeros(n)
    @inbounds for j in 1:n, idx in nzrange(A, j)
        k = rv[idx]; k != j && (w = -nz[idx]; w > mw[j] && (mw[j] = w))
    end
    return mw
end

"""
    form_tentative_soc(A, root, aggregated, mw, τ; expand_max=6) -> Vector{Int}

DRA tentative aggregate (Algorithm 2 steps 5–6) restricted to strong edges: a
neighbor `j` of `m` is admitted only if `|a_{mj}| ≥ τ·mw[m]`.
"""
function form_tentative_soc(A::SparseMatrixCSC, root::Int, aggregated::AbstractVector{Bool},
                            mw::AbstractVector, τ::Real; expand_max::Int=6)
    rv = rowvals(A); nz = nonzeros(A)
    G = Int[root]
    thr = τ * mw[root]
    @inbounds for idx in nzrange(A, root)
        k = rv[idx]; k == root && continue
        (!aggregated[k] && -nz[idx] >= thr) && push!(G, k)
    end
    if length(G) <= expand_max
        base = copy(G); seen = Set(G)
        for m in base
            thm = τ * mw[m]
            @inbounds for idx in nzrange(A, m)
                k = rv[idx]; k == m && continue
                (!aggregated[k] && !(k in seen) && -nz[idx] >= thm) && (push!(G, k); push!(seen, k))
            end
        end
    end
    return G
end

"""
    dra_aggregate_soc(A; τ=0.05, expand_max=6) -> (agg, nc)

Full DRA aggregation with the SoC veto and *no* quality control (for inspecting
aggregate shape). Leftover vertices with no strong unaggregated neighbor become
singletons.
"""
function dra_aggregate_soc(A::SparseMatrixCSC; τ::Real=0.05, expand_max::Int=6)
    n = size(A, 1); mw = max_incident(A); rv = rowvals(A)
    deg = [count(k -> rv[k] != j, nzrange(A, j)) for j in 1:n]
    key = [deg[j] >= 1 ? floor(Int, log2(deg[j])) : -1 for j in 1:n]
    order = sortperm(key; rev = true)
    agg = zeros(Int, n); aggregated = falses(n); nc = 0
    for r in order
        aggregated[r] && continue
        G = form_tentative_soc(A, r, aggregated, mw, τ; expand_max = expand_max)
        nc += 1
        for v in G; aggregated[v] = true; agg[v] = nc; end
    end
    for v in 1:n
        agg[v] == 0 && (nc += 1; agg[v] = nc)
    end
    return agg, nc
end
