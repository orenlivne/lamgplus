# Degree-aware Rooted Aggregation — Algorithm 2 (Napov–Notay 2017, §4.1; from [14]).
#
# Vertices are processed in decreasing order of floor(log2(degree)) (a "partial
# sort" that avoids a full degree sort). Each unaggregated root absorbs its
# unaggregated neighbors; if that aggregate has ≤ `expand_max` (=6) vertices, the
# unaggregated neighbors-of-neighbors are appended.

"""
    dra_aggregate(A; active=nothing, expand_max=6) -> (agg, nc)

Compute the DRA partition of the (active subset of) vertices of graph Laplacian
`A`. Returns `agg::Vector{Int}` (aggregate id 1..`nc` per vertex; 0 for inactive
vertices) and the number of aggregates `nc`.

`active` (a `Bool` mask) restricts aggregation to a vertex subset — used by the
complexity-enhancement re-aggregation (§4.4); edges to inactive vertices are
ignored. Default: all vertices active.
"""
function dra_aggregate(A::SparseMatrixCSC; active::Union{Nothing,AbstractVector{Bool}}=nothing,
                       expand_max::Int=6)
    n = size(A, 1)
    act = active === nothing ? trues(n) : active
    rv = rowvals(A)

    # combinatorial degree among active neighbors, and the bucket key
    deg = zeros(Int, n)
    for j in 1:n
        act[j] || continue
        d = 0
        for idx in nzrange(A, j)
            k = rv[idx]
            (k != j && act[k]) && (d += 1)
        end
        deg[j] = d
    end
    key(j) = deg[j] >= 1 ? floor(Int, log2(deg[j])) : -1
    order = sort([j for j in 1:n if act[j]]; by = j -> -key(j))

    agg = zeros(Int, n)
    nc = 0
    for r in order
        (agg[r] != 0) && continue
        nc += 1
        agg[r] = nc
        base = Int[r]
        for idx in nzrange(A, r)             # step 5: root + unaggregated neighbors
            k = rv[idx]
            if k != r && act[k] && agg[k] == 0
                agg[k] = nc; push!(base, k)
            end
        end
        if length(base) <= expand_max        # step 6: + neighbors-of-neighbors
            for m in base, idx in nzrange(A, m)
                k = rv[idx]
                if k != m && act[k] && agg[k] == 0
                    agg[k] = nc
                end
            end
        end
    end
    return agg, nc
end
