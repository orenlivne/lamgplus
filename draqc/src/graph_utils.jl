# Graph-Laplacian utilities for the DRA-QC reimplementation
# (Napov & Notay, "An Efficient Multigrid Method for Graph Laplacian Systems II:
#  Robust Aggregation", SIAM J. Sci. Comput. 39(5), 2017).
#
# Conventions: A is a graph Laplacian — symmetric, nonpositive off-diagonal,
# zero row sum (A·1 = 0), positive diagonal a_jj = Σ_{k≠j}|a_jk| (total degree).

"""
    total_degrees(A) -> Vector

Total weighted degree of each vertex = diagonal of the Laplacian.
"""
total_degrees(A::SparseMatrixCSC) = diag(A)

"""
    subgraph_laplacian(A, G) -> (A_G, ext)

Induced-subgraph Laplacian on the ordered vertex set `G` (matrix `A_G` of
Napov–Notay Thm 3.2): off-diagonals are the original `a_{jk}` for `j,k ∈ G`, and
the diagonal is set so each row sums to zero (so `A_G·1 = 0`). Also returns
`ext[p] = Σ_{k∉G} |a_{G[p],k}|`, the external (cut) degree of vertex `G[p]`.

`A_G` is returned dense; aggregates are small by construction.
"""
function subgraph_laplacian(A::SparseMatrixCSC, G::AbstractVector{<:Integer})
    ng = length(G)
    pos = Dict{Int,Int}(Int(G[p]) => p for p in 1:ng)
    AG = zeros(ng, ng)
    ext = zeros(ng)
    rv = rowvals(A); nz = nonzeros(A)
    for p in 1:ng
        j = Int(G[p])
        intdeg = 0.0
        for idx in nzrange(A, j)
            k = rv[idx]
            k == j && continue
            v = nz[idx]                  # a_{kj} = a_{jk} ≤ 0
            q = get(pos, k, 0)
            if q != 0
                AG[p, q] = v             # original (negative) off-diagonal
                intdeg += -v             # |a_jk|, internal
            else
                ext[p] += -v             # |a_jk|, external
            end
        end
        AG[p, p] = intdeg                # internal degree ⇒ zero row sum
    end
    return AG, ext
end
