# DRA-QC-CE setup: the full Algorithm 3 partition (DRA + quality control +
# complexity enhancement), the Galerkin coarse Laplacian (eq. 2), and the
# multilevel hierarchy build (Napov–Notay 2017, §4.5 & §5.2).

"""
    form_tentative(A, root, aggregated; expand_max=6) -> Vector{Int}

Tentative aggregate around `root` (Algorithm 3 steps 5–6): root + unaggregated
neighbors, plus unaggregated neighbors-of-neighbors if the size is ≤ `expand_max`.
"""
function form_tentative(A::SparseMatrixCSC, root::Int, aggregated::AbstractVector{Bool};
                        expand_max::Int=6)
    rv = rowvals(A)
    G = Int[root]
    for idx in nzrange(A, root)
        k = rv[idx]
        (k != root && !aggregated[k]) && push!(G, k)
    end
    if length(G) <= expand_max
        base = copy(G)
        seen = Set(G)
        for m in base, idx in nzrange(A, m)
            k = rv[idx]
            if k != m && !aggregated[k] && !(k in seen)
                push!(G, k); push!(seen, k)
            end
        end
    end
    return G
end

"""
    draqc_partition(A; κbar=10.0) -> (agg, nc)

Algorithm 3 (DRA-QC-CE): greedy DRA with per-aggregate quality control, then the
complexity-enhancement reform of small aggregates. Returns `agg::Vector{Int}`
(aggregate id 1..nc for every vertex) and `nc`.
"""
function draqc_partition(A::SparseMatrixCSC; κbar::Real=10.0, maxdepth::Int=4)
    n = size(A, 1)
    δ = delta_vector(A)
    rv = rowvals(A)
    deg = [count(k -> rv[k] != j, nzrange(A, j)) for j in 1:n]
    key = [deg[j] >= 1 ? floor(Int, log2(deg[j])) : -1 for j in 1:n]
    order = sortperm(key; rev = true)

    aggregated = falses(n)
    agg = zeros(Int, n)
    inS = falses(n); scratch = Int[]          # reusable buffers for matrix-free QC
    nc = 0
    nleft = n
    while nleft > 0
        for r in order
            aggregated[r] && continue
            G = form_tentative(A, r, aggregated)
            G = refine_aggregate!(A, G, r, δ, inS, scratch; κbar = κbar, maxdepth = maxdepth)
            nc += 1
            for v in G
                aggregated[v] = true; agg[v] = nc
            end
            nleft -= length(G)
        end
    end

    # complexity enhancement (steps 21–24): if too little coarsening, dissolve the
    # smallest aggregates and re-aggregate the freed vertices without quality control.
    if nc > n / 4
        sizes = zeros(Int, nc)
        for v in 1:n; sizes[agg[v]] += 1; end
        keep = sizes .>= 4
        newid = zeros(Int, nc); nk = 0
        for a in 1:nc
            if keep[a]; nk += 1; newid[a] = nk; end
        end
        freed = falses(n)
        for v in 1:n
            if keep[agg[v]]; agg[v] = newid[agg[v]]; else; freed[v] = true; agg[v] = 0; end
        end
        if any(freed)
            agg2, nc2 = dra_aggregate(A; active = freed)   # Algorithm 2, no QC
            for v in 1:n
                freed[v] && (agg[v] = nk + agg2[v])
            end
            nc = nk + nc2
        else
            nc = nk
        end
    end
    return agg, nc
end

"""
    galerkin(A, agg, nc) -> (A_c, P)

Piecewise-constant prolongation `P` (0/1, eq. 3) and Galerkin coarse Laplacian
`A_c = Pᵀ A P` (eq. 2). `A_c` is itself a graph Laplacian (`A_c·1 = 0`).
"""
function galerkin(A::SparseMatrixCSC, agg::AbstractVector{<:Integer}, nc::Integer)
    n = size(A, 1)
    P = sparse(1:n, agg, ones(n), n, nc)
    Ac = P' * A * P
    Ac = (Ac + Ac') / 2          # force exact symmetry (P'AP differs by round-off)
    return Ac, P
end

"""
    DRAQCHierarchy

Multilevel hierarchy: `A[ℓ]` the level-ℓ Laplacian, `P[ℓ]` the prolongation from
level ℓ+1 to level ℓ. The coarsest level `A[end]` is solved directly.
"""
struct DRAQCHierarchy
    A::Vector{SparseMatrixCSC{Float64,Int}}
    P::Vector{SparseMatrixCSC{Float64,Int}}
end

num_levels(h::DRAQCHierarchy) = length(h.A)

"""
    draqc_setup(A; κbar=10.0, maxcoarse=100, maxlevels=40) -> DRAQCHierarchy

Build the DRA-QC multilevel hierarchy by repeated DRA-QC-CE coarsening until the
coarsest operator has ≤ `maxcoarse` rows (or coarsening stalls).
"""
function draqc_setup(A::SparseMatrixCSC; κbar::Real=10.0, maxcoarse::Int=100, maxlevels::Int=40)
    As = SparseMatrixCSC{Float64,Int}[A]
    Ps = SparseMatrixCSC{Float64,Int}[]
    while size(As[end], 1) > maxcoarse && length(As) < maxlevels
        Acur = As[end]
        agg, nc = draqc_partition(Acur; κbar = κbar)
        nc >= size(Acur, 1) && break          # no coarsening ⇒ stop
        Ac, P = galerkin(Acur, agg, nc)
        push!(Ps, P); push!(As, Ac)
    end
    return DRAQCHierarchy(As, Ps)
end

"""
    operator_complexity(h) -> Float64

Σ nnz(A[ℓ]) / nnz(A[1]) — the storage/work complexity of the hierarchy.
"""
operator_complexity(h::DRAQCHierarchy) = sum(nnz, h.A) / nnz(h.A[1])
