"""
    laplacian(W::SparseMatrixCSC) -> SparseMatrixCSC

Build the (weighted) graph Laplacian L = D - W from a symmetric adjacency matrix W.
W must be symmetric with zero diagonal; entries are edge weights (capacities).
Negative weights are allowed (LAMG handles SPSD graphs with negative edges, cf.
Fig.~1 of Livne-Brandt 2012).
"""
function laplacian(W::SparseMatrixCSC{T,Int}) where {T<:Real}
    @assert size(W, 1) == size(W, 2) "adjacency must be square"
    n = size(W, 1)
    d = vec(sum(W; dims = 2))
    return spdiagm(0 => d) - W
end

"""
    path_laplacian(n::Int) -> SparseMatrixCSC

1D path graph Laplacian (open boundaries). Useful for sanity tests where
the spectrum is known analytically: λ_k = 2(1 - cos(kπ/n)) for k=0..n-1.
"""
function path_laplacian(n::Int)
    @assert n >= 2
    rows = Int[]; cols = Int[]; vals = Float64[]
    for i in 1:(n-1)
        push!(rows, i);   push!(cols, i+1); push!(vals, 1.0)
        push!(rows, i+1); push!(cols, i);   push!(vals, 1.0)
    end
    W = sparse(rows, cols, vals, n, n)
    return laplacian(W)
end

"""
    grid2d_laplacian(nx::Int, ny::Int) -> SparseMatrixCSC

2D 5-point grid Laplacian on an nx-by-ny lattice with unit edge weights.
"""
function grid2d_laplacian(nx::Int, ny::Int)
    @assert nx >= 2 && ny >= 2
    n = nx * ny
    idx(i, j) = (j - 1) * nx + i
    rows = Int[]; cols = Int[]; vals = Float64[]
    for j in 1:ny, i in 1:nx
        a = idx(i, j)
        if i < nx
            b = idx(i + 1, j)
            push!(rows, a); push!(cols, b); push!(vals, 1.0)
            push!(rows, b); push!(cols, a); push!(vals, 1.0)
        end
        if j < ny
            b = idx(i, j + 1)
            push!(rows, a); push!(cols, b); push!(vals, 1.0)
            push!(rows, b); push!(cols, a); push!(vals, 1.0)
        end
    end
    W = sparse(rows, cols, vals, n, n)
    return laplacian(W)
end

"""
    grid3d_laplacian(nx::Int, ny::Int, nz::Int) -> SparseMatrixCSC

3D 7-point grid Laplacian on an nx × ny × nz lattice with unit edge weights.
"""
function grid3d_laplacian(nx::Int, ny::Int, nz::Int)
    @assert nx >= 2 && ny >= 2 && nz >= 2
    n = nx * ny * nz
    idx(i, j, k) = ((k - 1) * ny + (j - 1)) * nx + i
    rows = Int[]; cols = Int[]; vals = Float64[]
    for k in 1:nz, j in 1:ny, i in 1:nx
        a = idx(i, j, k)
        if i < nx
            b = idx(i + 1, j, k)
            push!(rows, a); push!(cols, b); push!(vals, 1.0)
            push!(rows, b); push!(cols, a); push!(vals, 1.0)
        end
        if j < ny
            b = idx(i, j + 1, k)
            push!(rows, a); push!(cols, b); push!(vals, 1.0)
            push!(rows, b); push!(cols, a); push!(vals, 1.0)
        end
        if k < nz
            b = idx(i, j, k + 1)
            push!(rows, a); push!(cols, b); push!(vals, 1.0)
            push!(rows, b); push!(cols, a); push!(vals, 1.0)
        end
    end
    W = sparse(rows, cols, vals, n, n)
    return laplacian(W)
end

"""
    random_graph_laplacian(n::Int, p::Real; rng=Random.default_rng()) -> SparseMatrixCSC

Erdős–Rényi G(n, p) graph with unit weights, returned as a connected Laplacian.
If the random draw produces a disconnected graph, retries up to 10 times.
"""
function random_graph_laplacian(n::Int, p::Real; rng = Random.default_rng())
    for _ in 1:10
        rows = Int[]; cols = Int[]; vals = Float64[]
        for i in 1:n, j in (i+1):n
            if rand(rng) < p
                push!(rows, i); push!(cols, j); push!(vals, 1.0)
                push!(rows, j); push!(cols, i); push!(vals, 1.0)
            end
        end
        W = sparse(rows, cols, vals, n, n)
        L = laplacian(W)
        if is_connected_laplacian(L)
            return L
        end
    end
    error("could not draw a connected G($n, $p) in 10 tries — increase p")
end

"""
    is_laplacian(L; tol=1e-10) -> Bool

Verify that L is symmetric and has zero row sums. Does NOT require non-positive
off-diagonals (those are tested separately by `is_graph_laplacian` since LAMG
admits negative edge weights — cf. Fig. 1 of Livne-Brandt 2012).
"""
function is_laplacian(L::AbstractMatrix; tol::Real = 1e-10)
    issymmetric(L) || return false
    maximum(abs, sum(L; dims = 2)) < tol || return false
    return true
end

"""
    is_graph_laplacian(L; tol=1e-10) -> Bool

Verify that L is a *graph* Laplacian: symmetric, zero row sums, AND non-positive
off-diagonals. This is the structural property we want preserved across coarse
levels in max-flow AMG, since `-L[i,j]` is a physical edge capacity (≥ 0).
"""
function is_graph_laplacian(L::AbstractMatrix; tol::Real = 1e-10)
    is_laplacian(L; tol = tol) || return false
    n = size(L, 1)
    for j in 1:n, i in 1:n
        i == j && continue
        L[i, j] <= tol || return false
    end
    return true
end

"""
    mean_bandwidth(A::SparseMatrixCSC) -> Float64

Mean |i−j| over off-diagonal nonzeros — a locality proxy. Lower = the sparsity is
more banded, so the SpMV/relaxation gather streams (cache-friendly).
"""
function mean_bandwidth(A::SparseMatrixCSC)
    rows = rowvals(A); s = 0.0; c = 0
    @inbounds for j in 1:size(A, 2), k in nzrange(A, j)
        i = rows[k]
        i != j && (s += abs(i - j); c += 1)
    end
    return c == 0 ? 0.0 : s / c
end

"""
    mean_bandwidth(A::SparseMatrixCSC, pinv::Vector{Int}) -> Float64

Mean bandwidth the matrix WOULD have under the symmetric permutation `p`, computed
matrix-free from the inverse permutation `pinv = invperm(p)` — the off-diagonal entry
at original `(i,j)` lands at `(pinv[i], pinv[j])` in `A[p,p]`, so its bandwidth is
`|pinv[i] - pinv[j]|`. Identical to `mean_bandwidth(A[p,p])` but WITHOUT materializing
the permuted matrix (used by `setup` to decide whether reordering helps).
"""
function mean_bandwidth(A::SparseMatrixCSC, pinv::Vector{Int})
    rows = rowvals(A); s = 0.0; c = 0
    @inbounds for j in 1:size(A, 2), k in nzrange(A, j)
        i = rows[k]
        i != j && (s += abs(pinv[i] - pinv[j]); c += 1)
    end
    return c == 0 ? 0.0 : s / c
end

"""
    rcm_order(A::SparseMatrixCSC) -> Vector{Int}

Reverse Cuthill–McKee ordering: BFS from a min-degree seed in each component,
visiting neighbours in ascending-degree order, then reversed. A symmetric
permutation `A[p,p]` is the SAME operator (same spectrum/convergence) with a more
local sparsity → cache-friendly gather AND (because greedy aggregation is order-
sensitive) often spatially-coherent aggregates. Used by `setup` to renumber
bandwidth-poor inputs (social/web); auto-skipped on already-local inputs (grids).
"""
function rcm_order(A::SparseMatrixCSC)
    n = size(A, 1); rows = rowvals(A)
    deg = [length(nzrange(A, j)) for j in 1:n]
    visited = falses(n); order = Vector{Int}(undef, 0); sizehint!(order, n)
    @inbounds while length(order) < n
        seed = 0; bestd = typemax(Int)        # unvisited min-degree seed (next component)
        for i in 1:n
            !visited[i] && deg[i] < bestd && (bestd = deg[i]; seed = i)
        end
        seed == 0 && break
        q = [seed]; visited[seed] = true; head = 1
        while head <= length(q)
            u = q[head]; head += 1; push!(order, u)
            nb = Int[]
            for k in nzrange(A, u)
                v = rows[k]
                (v != u && !visited[v]) && push!(nb, v)
            end
            sort!(nb; by = x -> deg[x])         # Cuthill–McKee: ascending degree
            for v in nb
                visited[v] = true; push!(q, v)
            end
        end
    end
    return reverse(order)
end

"""
    null_vector(L::AbstractMatrix) -> Vector

Return the null-space vector of a connected graph Laplacian: the all-ones
vector, normalized.
"""
function null_vector(L::AbstractMatrix)
    n = size(L, 1)
    return ones(n) ./ sqrt(n)
end

"""
    connected_components(L::SparseMatrixCSC) -> Vector{Int}

Return a node-to-component label vector. Labels are 1..M for an M-component
graph, in BFS-discovery order. Off-diagonal nonzeros of `L` define
adjacency. O(n + m).
"""
function connected_components(L::SparseMatrixCSC)
    n = size(L, 1)
    label = zeros(Int, n)
    rows = rowvals(L); vals = nonzeros(L)
    next_label = 0
    for start in 1:n
        label[start] != 0 && continue
        next_label += 1
        label[start] = next_label
        stack = [start]
        while !isempty(stack)
            u = pop!(stack)
            for k in nzrange(L, u)
                v = rows[k]
                v == u && continue
                vals[k] == 0 && continue
                if label[v] == 0
                    label[v] = next_label
                    push!(stack, v)
                end
            end
        end
    end
    return label
end

"""
    largest_component(L::SparseMatrixCSC) -> (L_largest, retained_nodes)

Extract the largest connected component as a new graph Laplacian.
Returns `(L_largest, retained)` where `retained` is the original-graph
indices kept. If `L` is already singly connected, returns `(L, 1:n)`.

The LAMG paper §3.7 mentions both options for handling multi-component
graphs: (a) native multi-component deflation at the coarsest level, or
(b) reduce to a singly-connected graph upfront. We do (b) — simpler and
sufficient for graphs whose interesting subgraph is the giant CC.
"""
function largest_component(L::SparseMatrixCSC)
    n = size(L, 1)
    label = connected_components(L)
    M = maximum(label)
    M == 1 && return L, collect(1:n)
    sizes = zeros(Int, M)
    for l in label
        sizes[l] += 1
    end
    biggest = argmax(sizes)
    retained = findall(==(biggest), label)
    return L[retained, retained], retained
end

function is_connected_laplacian(L::SparseMatrixCSC; rtol = 1e-8)
    # The graph defined by off-diagonals of L is connected iff the smallest
    # eigenvalue of L is unique-zero. Cheap check: BFS over the sparsity pattern.
    n = size(L, 1)
    visited = falses(n)
    stack = [1]; visited[1] = true
    while !isempty(stack)
        u = pop!(stack)
        for k in nzrange(L, u)
            v = L.rowval[k]
            if v != u && !visited[v]
                visited[v] = true
                push!(stack, v)
            end
        end
    end
    return all(visited)
end
