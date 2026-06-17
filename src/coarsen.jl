"""
Affinity-based aggregation, LAMG §3.4 + §4.2.

Two ingredients to mirror the paper as faithfully as our framework allows:

1. **Energy-ratio guard** `q_U(x) ≤ Q` (§3.4): only accept a pair (u, t) if
   the local energy at u after setting x_u ← x_t is at most Q × the
   locally-optimal energy E_u^*. Caps aggregate-induced "energy inflation"
   that hurts caliber-1 multilevel convergence. Default Q = 2.5.

2. **Two-stage δ-thresholding** (§9.2 line 842):
     stage 1 (δ = 0.9):  pair only very-high-affinity neighbors
     stage 2 (δ = 0.54): pair the rest with lower bar
   Both subject to the energy-ratio guard.

Both steps are purely local: each node only ever scans its own neighbors.
"""

"""
    Aggregation(aggregate::Vector{Int}, n_coarse::Int)
"""
struct Aggregation
    aggregate::Vector{Int}
    n_coarse::Int
end

"""
    affinity(test_vectors::AbstractMatrix, edges::Vector{Tuple{Int,Int}})
        -> Dict{Tuple{Int,Int}, Float64}

Public API: dict-keyed-by-edge. Internally we use a parallel Vector for speed.

The affinity is the squared cosine similarity of node profiles:
    c_{uv} = ⟨X[u,:], X[v,:]⟩² / (‖X[u,:]‖² ‖X[v,:]‖²)   ∈ [0, 1].
"""
function affinity(test_vectors::AbstractMatrix,
                  edges::Vector{Tuple{Int,Int}})
    aff_vec = _affinity_vector(test_vectors, edges)
    return Dict(edges[k] => aff_vec[k] for k in eachindex(edges))
end

function _affinity_vector(test_vectors::AbstractMatrix,
                          edges::Vector{Tuple{Int,Int}})
    K = size(test_vectors, 2)
    aff_values = Vector{Float64}(undef, length(edges))
    sq = vec(sum(abs2, test_vectors; dims = 2))
    @inbounds for k in eachindex(edges)
        a, b = edges[k]
        u, v = minmax(a, b)
        dot_uv = 0.0
        for j in 1:K
            dot_uv += test_vectors[u, j] * test_vectors[v, j]
        end
        denom = sq[u] * sq[v]
        c = denom > 0 ? (dot_uv * dot_uv) / denom : 0.0
        aff_values[k] = min(max(c, 0.0), 1.0)
    end
    return aff_values
end

"""
Per-node affinity list in flat CSR layout: for each node `j`, its retained open
neighbours are `nbr[ptr[j]:ptr[j+1]-1]` with the parallel affinities in `aff`.
Built once per `aggregate(L)` call in O(m). The flat form replaces the previous
array-of-`Vector{Tuple}` (n heap vectors + push! growth) with three preallocated
arrays — same per-node `nzrange` traversal order, same soc_τ veto, same affinity
values ⇒ bit-identical to the old layout (verified by the equivalence oracle).
"""
struct NodeAffinities
    ptr::Vector{Int}      # length n+1, CSR offsets
    nbr::Vector{Int}      # retained neighbour node indices
    aff::Vector{Float64}  # parallel affinities
end

# Strength-of-connection threshold for node j (soc_τ>0): drop neighbours whose
# matrix coupling is negligible vs j's strongest incident coupling. The affinity
# is a noisy K-sample statistic that can SPURIOUSLY rank a weak edge above a
# strong one (e.g. a weight-ε y-edge on an anisotropic grid scoring 0.97 > the
# weight-1 x-edge's 0.89); such a merge is never worth it and the matrix is not
# fooled. soc_τ=0 disables (default). Computed identically in the count and fill
# passes (deterministic ⇒ identical veto decisions ⇒ bit-identical).
@inline function _soc_threshold(L, rows, vals, j::Int, soc_τ::Float64)
    soc_τ <= 0 && return 0.0
    wmax = 0.0
    @inbounds for kp in nzrange(L, j)
        rows[kp] == j && continue
        w = abs(vals[kp]); w > wmax && (wmax = w)
    end
    return soc_τ * wmax
end

function _build_node_affinities(L::SparseMatrixCSC, X::AbstractMatrix; soc_τ::Float64 = 0.0)
    n = size(L, 1)
    K = size(X, 2)
    sq = vec(sum(abs2, X; dims = 2))
    rows = rowvals(L); vals = nonzeros(L)
    # Pass 1: count retained neighbours per node (identical predicate to the fill
    # pass, including the soc_τ veto) to size the flat buffers exactly.
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    @inbounds for j in 1:n
        thresh = _soc_threshold(L, rows, vals, j, soc_τ)
        cnt = 0
        for kp in nzrange(L, j)
            i = rows[kp]
            i == j && continue
            vals[kp] == 0 && continue
            soc_τ > 0 && abs(vals[kp]) < thresh && continue
            cnt += 1
        end
        ptr[j+1] = ptr[j] + cnt
    end
    total = ptr[n+1] - 1
    nbr = Vector{Int}(undef, total)
    aff = Vector{Float64}(undef, total)
    # Pass 2: fill, in the identical nzrange order with identical values.
    @inbounds for j in 1:n
        thresh = _soc_threshold(L, rows, vals, j, soc_τ)
        p = ptr[j]
        for kp in nzrange(L, j)
            i = rows[kp]
            i == j && continue
            vals[kp] == 0 && continue
            soc_τ > 0 && abs(vals[kp]) < thresh && continue
            dot_uv = 0.0
            for kk in 1:K
                dot_uv += X[i, kk] * X[j, kk]
            end
            denom = sq[i] * sq[j]
            c = denom > 0 ? (dot_uv * dot_uv) / denom : 0.0
            nbr[p] = i
            aff[p] = min(max(c, 0.0), 1.0)
            p += 1
        end
    end
    return NodeAffinities(ptr, nbr, aff)
end

"""
    energy_ratio(L, u, t, X) -> Float64

Local energy ratio q_ut(x) at node u, *averaged* over the K test-vector
columns of X. For each test vector x = X[:, k]:

    E_u(x_u; x_neigh) = ½ Σ_v c_uv (x_u − x_v)²
    E_u^*(x_neigh)    = min over x_u = ½ Σ_v c_uv (x_v − x̄_u)²
                        where x̄_u = (Σ c_uv x_v) / (Σ c_uv) is the Jacobi step
    q_ut(x) = E_u(x_t; x_neigh) / E_u^*(x_neigh)

Returns the **maximum** of q_ut(x_k) across the K test vectors — we want
the worst-case ratio to be bounded.

(The paper uses an average; the max is more conservative and yields the
same qualitative behavior on graphs we tested.)
"""
function energy_ratio(L::SparseMatrixCSC, u::Int, t::Int, X::AbstractMatrix)
    K = size(X, 2)
    # Precompute Σ c_uv for u's neighbors.
    rows = rowvals(L); vals = nonzeros(L)
    nbrs = Int[]; weights = Float64[]
    sum_c = 0.0
    @inbounds for kp in nzrange(L, u)
        v = rows[kp]
        v == u && continue
        # Edge weight = −L[u,v]. Skip non-positive (numerical zero / negative).
        c = -vals[kp]
        c <= 0 && continue
        push!(nbrs, v)
        push!(weights, c)
        sum_c += c
    end
    sum_c == 0 && return Inf
    q_max = 0.0
    @inbounds for k in 1:K
        sum_cx  = 0.0
        sum_cx2 = 0.0
        for (i, v) in enumerate(nbrs)
            xv = X[v, k]
            sum_cx  += weights[i] * xv
            sum_cx2 += weights[i] * xv * xv
        end
        x_star = sum_cx / sum_c
        # E_u^* = ½ (Σ c x_v² − (Σ c x_v)² / Σ c)
        E_star = 0.5 * (sum_cx2 - sum_cx * sum_cx / sum_c)
        # E_u at x_u = x_t: ½ Σ c (x_t − x_v)² = ½ (x_t² Σc − 2 x_t Σcx + Σ c x_v²)
        xt = X[t, k]
        E_post = 0.5 * (xt * xt * sum_c - 2.0 * xt * sum_cx + sum_cx2)
        if E_star > 1e-30
            q = E_post / E_star
            q > q_max && (q_max = q)
        end
        # If E_star is ~0, x is locally constant; aggregation here is free.
        # Leave q_max unchanged (we don't increase it).
    end
    return q_max
end

# Bulk-precompute the per-node energy sums so the guard Eq.(3.12) is O(K) per
# candidate instead of O(K·degree) (the latter is superlinear on high-degree
# graphs — the dominant setup cost). For node u and TV k, with c_uv=-L[u,v]>0:
#   sum_c[u] = Σ_v c_uv,  sum_cx[u,k] = Σ_v c_uv X[v,k],  sum_cx2[u,k] = Σ_v c_uv X[v,k]².
# (MATLAB updates the analogous local energies in bulk like TV residuals.)
function _precompute_energy_sums(L::SparseMatrixCSC, X::AbstractMatrix)
    n = size(L, 1); K = size(X, 2)
    rows = rowvals(L); vals = nonzeros(L)
    sum_c  = zeros(n)
    sum_cx = zeros(n, K)
    sum_cx2 = zeros(n, K)
    @inbounds for u in 1:n
        for kp in nzrange(L, u)
            v = rows[kp]; v == u && continue
            c = -vals[kp]; c <= 0 && continue
            sum_c[u] += c
            for k in 1:K
                xv = X[v, k]
                sum_cx[u, k]  += c * xv
                sum_cx2[u, k] += c * xv * xv
            end
        end
    end
    return sum_c, sum_cx, sum_cx2
end

# O(K) energy ratio q_{u←t} from the precomputed sums; identical math to
# energy_ratio(L,u,t,X).
@inline function _energy_ratio_fast(u::Int, t::Int, X::AbstractMatrix,
                                    sum_c, sum_cx, sum_cx2)
    sc = sum_c[u]
    sc == 0.0 && return Inf
    q_max = 0.0
    @inbounds for k in 1:size(X, 2)
        scx = sum_cx[u, k]; scx2 = sum_cx2[u, k]
        E_star = 0.5 * (scx2 - scx * scx / sc)
        xt = X[t, k]
        E_post = 0.5 * (xt * xt * sc - 2.0 * xt * scx + scx2)
        if E_star > 1e-30
            q = E_post / E_star
            q > q_max && (q_max = q)
        end
    end
    return q_max
end

# Aggregation-fidelity controls, matching the MATLAB lamg-2.2.1 reference
# (Livne & Brandt 2012 [LOP168], Eq. (3.12) §3.4.4 and §3.4.2). Both default
# to the paper-faithful behavior; the `false` settings reproduce the earlier
# (buggy) port behavior and are kept only for regression A/B testing.
#   _AGG_SKIP_ORPHAN=true: leftover undecided nodes stay singleton seeds
#     (paper-faithful), instead of being force-absorbed into a neighbor
#     WITHOUT the energy guard (which densified/over-aggregated coarse levels).
#   _AGG_ASYM_GUARD=true: test only q_{u←v} (u joins seed v), matching the
#     ASYMMETRIC energy-ratio guard Eq. (3.12), instead of the stricter
#     "both q_uv and q_vu ≤ Q" which rejected valid merges and forced the
#     orphan-absorption fallback (→ over-aggregation on structured graphs).
const _AGG_SKIP_ORPHAN = Ref(true)
const _AGG_ASYM_GUARD  = Ref(true)

"""
    _relax_test_vectors(L, K, ν, rng) -> X :: n × K

K random zero-mean unit vectors relaxed by ν Gauss–Seidel sweeps on `Lx = 0`
(LAMG §3.3). Shared by aggregation (affinity) and the caliber-2 interpolation
weight fit so both reuse the SAME test vectors — no extra setup cost.
"""
function _relax_test_vectors(L::SparseMatrixCSC, K::Int, ν::Int, rng)
    n = size(L, 1)
    rx = GaussSeidelRelaxer(L)
    # Fill X with the RNG draws in place (rand! draws the same values in the same
    # column-major order as rand(rng,n,K)), then apply x ↦ 2x−1 in place — removes
    # the transient rand matrix and the fused broadcast's separate output array.
    X = Matrix{Float64}(undef, n, K)
    rand!(rng, X)
    @inbounds @simd for i in eachindex(X)
        X[i] = 2.0 * X[i] - 1.0
    end
    b = zeros(n)
    for k in 1:K
        col = view(X, :, k)
        col .-= sum(col) / n
        nrm = norm(col)
        nrm > 0 && (col ./= nrm)
        for _ in 1:ν
            relax!(rx, col, b; sweeps = 1)
        end
    end
    return X
end

"""
    aggregate(L; ν=3, K=4, max_aggregate_size=8,
              δ_stages=(0.9, 0.7, 0.5, 0.4), Q=2.5, rng=...) -> Aggregation

LAMG aggregation — port of MATLAB `aggregationSweep_matlab.m` with the
§3.4 energy-ratio guard. Critical difference from a pair-only matching:
**aggregates grow incrementally**. Each pass over the node set lets an
undecided node `u` either pair with another undecided neighbor (creating
a new aggregate seeded at the neighbor) OR join an EXISTING aggregate by
attaching to its seed. This unbounded growth (capped only by
`max_aggregate_size`) is what makes the algorithm produce aggregates of
average size 3–4 on dense graphs, not stuck at 2.

Earlier pair-only port produced average aggregate size ≈ 2.4 on the FE
matrices (e.g., bone010), giving operator complexity that GREW with
problem size and a super-linear total cost. The fix below restores the
sequential-growth behaviour and brings operator complexity back to a
size-independent constant.

Algorithm (per δ stage, scanning bins of nodes by max affinity descending):
  for each undecided node u:
      Find candidate open neighbors (undecided OR seeds of existing aggregates)
      with affinity ≥ δ AND energy_ratio ≤ Q AND target_agg_size < max_size.
      Pick the highest-affinity candidate s.
      If s is undecided: s becomes seed of a new aggregate, u joins it.
      If s is a seed: u joins s's aggregate (size grows by 1).
"""
function aggregate(L::SparseMatrixCSC; ν::Int = 3, K::Int = 4,
                   max_aggregate_size::Int = 8,
                   δ_stages = (0.9, 0.7, 0.5, 0.4),
                   target_coarsening_ratio::Real = 0.5,
                   Q::Real = 2.5,
                   hub_threshold::Real = 8.0,
                   soc_τ::Float64 = 0.0,
                   jaccard_priority::Bool = false,
                   X_ext::Union{Nothing,AbstractMatrix} = nothing,
                   rng = Random.default_rng())
    n = size(L, 1)

    # 1. Test vectors. Default: K random vectors relaxed by ν GS sweeps on Lx=0
    # (LAMG §3.3). `X_ext` overrides this with externally-supplied test vectors
    # (e.g. bootstrapped / eigenvector TVs, LAMG §6.3) for the affinities.
    X = X_ext === nothing ? _relax_test_vectors(L, K, ν, rng) : Matrix{Float64}(X_ext)

    # 2. Per-node affinity lists (flat CSR; same nzrange order as before).
    naff = _build_node_affinities(L, X; soc_τ = soc_τ)

    # 2.1. Bulk per-node energy sums for the O(K)-per-candidate energy guard.
    e_sc, e_scx, e_scx2 = _precompute_energy_sums(L, X)

    # 2.2. Optional Jaccard-priority tie-break (default off → path unchanged).
    # Affinity SATURATES on dense low-clustering graphs (cosine≈1 for most edges),
    # so it cannot discriminate which admissible neighbor to merge. Among the
    # affinity-admissible, energy-feasible candidates we then rank by the Jaccard
    # overlap of their neighbourhoods — which is the exact Galerkin-fill predictor
    # (fill avoided by merging u,v = |N(u)∩N(v)|; LAMG §3.1.3). Adjacency sets are
    # precomputed once; Jaccard per candidate is O(min deg).
    adjset = Vector{Set{Int}}()
    if jaccard_priority
        adjset = [Set{Int}() for _ in 1:n]
        @inbounds for u in 1:n
            for k in L.colptr[u]:(L.colptr[u+1]-1)
                v = L.rowval[k]; v != u && push!(adjset[u], v)
            end
        end
    end
    @inline function jaccard(u::Int, v::Int)
        Su = adjset[u]; Sv = adjset[v]; inter = 0
        if length(Su) <= length(Sv)
            for w in Su; (w in Sv) && (inter += 1); end
        else
            for w in Sv; (w in Su) && (inter += 1); end
        end
        uni = length(Su) + length(Sv) - inter
        uni > 0 ? inter / uni : 0.0
    end

    # Status encoding:
    #   seed_of[u] == 0    : u is undecided
    #   seed_of[u] == u    : u is the seed of its own aggregate
    #   seed_of[u] == s    : u is an associate of seed s ≠ u
    seed_of = zeros(Int, n)
    agg_size = zeros(Int, n)  # only meaningful at seeds (= number of members)

    # 2.5. Hub isolation. Port of lamg-2.2.1 aggregationDegreeThreshold:
    #
    #     mark u as a forced singleton seed if
    #         degree(u) >= hub_threshold * median(degree(N(u)))
    #
    # The threshold is RELATIVE to the local median neighbor degree (not
    # the global median). This identifies hubs whose degree is anomalous
    # within their own neighborhood — exactly the densification-driving
    # nodes in scale-free graphs. Set hub_threshold=0 to disable.
    if hub_threshold > 0 && n > 1
        # Degree per node from the off-diagonal sparsity pattern of L
        # (excluding diagonal). For a graph Laplacian L = D - W, the off-
        # diagonal nonzeros of L correspond to edges; their count per row
        # is the degree.
        deg = zeros(Int, n)
        @inbounds for j in 1:n
            for k in L.colptr[j]:(L.colptr[j+1]-1)
                L.rowval[k] != j && (deg[j] += 1)
            end
        end
        # Median neighbor degree (per node).
        median_nbhr_deg = zeros(Float64, n)
        nbhr_buf = Int[]
        @inbounds for u in 1:n
            empty!(nbhr_buf)
            for k in L.colptr[u]:(L.colptr[u+1]-1)
                v = L.rowval[k]
                v != u && push!(nbhr_buf, deg[v])
            end
            median_nbhr_deg[u] = isempty(nbhr_buf) ? 1.0 : median(nbhr_buf)
        end
        # Forced singletons.
        n_hubs = 0
        @inbounds for u in 1:n
            if deg[u] >= hub_threshold * median_nbhr_deg[u] && deg[u] >= 2
                seed_of[u] = u
                agg_size[u] = 1
                n_hubs += 1
            end
        end
        # (n_hubs is computed but unused — could be logged with @debug.)
    end

    @inline function is_open(v)
        # Open = undecided OR seed (of an aggregate not yet full).
        s = seed_of[v]
        if s == 0
            return true
        elseif s == v
            return agg_size[s] < max_aggregate_size
        else
            return false   # associate of some other seed — can't join
        end
    end

    @inline function join_with!(u::Int, v::Int)
        # u is undecided; v is open (undecided or seed).
        sv = seed_of[v]
        if sv == 0
            # v becomes a new seed; u joins.
            seed_of[v] = v
            seed_of[u] = v
            agg_size[v] = 2
        else
            # v is already a seed (sv == v). u joins.
            seed_of[u] = sv
            agg_size[sv] += 1
        end
    end

    # 3. Stagewise growth — process nodes in descending δ-affinity order.
    #    Note: this is the heart of the fix. Where the old code did
    #    "pair undecided with undecided", we now allow "undecided joins
    #    open seed too".
    #
    #    Adaptive stopping: count effective aggregates after each stage
    #    (undecided node count as singletons + actual aggregates) and
    #    stop early if the target coarsening ratio is reached.
    target_n_aggs = ceil(Int, target_coarsening_ratio * n)
    for δ in δ_stages
        @inbounds for u in 1:n
            seed_of[u] != 0 && continue
            best_v = 0; best_c = -1.0
            for p in naff.ptr[u]:(naff.ptr[u+1]-1)
                v = naff.nbr[p]; c = naff.aff[p]
                c < δ && continue
                is_open(v) || continue
                # Energy-ratio guard. Compute symmetric q (worst direction).
                q_uv = _energy_ratio_fast(u, v, X, e_sc, e_scx, e_scx2)
                q_uv > Q && continue
                if !_AGG_ASYM_GUARD[]
                    q_vu = _energy_ratio_fast(v, u, X, e_sc, e_scx, e_scx2)
                    q_vu > Q && continue
                end
                # Rank by affinity (default) or, among these admissible candidates,
                # by neighbourhood-overlap Jaccard (tie-break for the saturated regime).
                score = jaccard_priority ? jaccard(u, v) : c
                if score > best_c
                    best_v = v; best_c = score
                end
            end
            if best_v != 0
                join_with!(u, best_v)
            end
        end
        # Stop early if target reached: count seeds + undecided.
        n_decided = 0; n_undecided = 0
        @inbounds for u in 1:n
            s = seed_of[u]
            if s == 0
                n_undecided += 1
            elseif s == u
                n_decided += 1     # seed
            end
        end
        # If undecided will singleton-out, total aggregates ≈ n_decided + n_undecided.
        n_decided + n_undecided <= target_n_aggs && break
    end

    # 4. Orphan absorption (no energy guard, just affinity-based; fills
    #    isolated undecided nodes into the nearest available aggregate
    #    that still has room). NOTE: not in the LOP168 design — the paper
    #    leaves still-undecided nodes as singleton seeds. Gated off by
    #    _AGG_SKIP_ORPHAN for fidelity to the reference.
    if !_AGG_SKIP_ORPHAN[]
    @inbounds for u in 1:n
        seed_of[u] != 0 && continue
        best = 0; best_c = -1.0
        for p in naff.ptr[u]:(naff.ptr[u+1]-1)
            v = naff.nbr[p]; c = naff.aff[p]
            sv = seed_of[v]
            sv == 0 && continue   # v also undecided
            (sv == v ? agg_size[sv] : agg_size[sv]) >= max_aggregate_size && continue
            if c > best_c
                best = v; best_c = c
            end
        end
        if best != 0
            s = seed_of[best]
            seed_of[u] = s
            agg_size[s] += 1
        end
    end
    end

    # 5. Singletons for whatever's left.
    @inbounds for u in 1:n
        if seed_of[u] == 0
            seed_of[u] = u
            agg_size[u] = 1
        end
    end

    # 6. Renumber: convert seed_of[u] (∈ original node indices) → aggregate
    #    indices 1..n_aggs.
    agg_id = zeros(Int, n)
    n_aggs = 0
    seed_to_idx = Dict{Int,Int}()
    for u in 1:n
        s = seed_of[u]
        idx = get(seed_to_idx, s, 0)
        if idx == 0
            n_aggs += 1
            idx = n_aggs
            seed_to_idx[s] = idx
        end
        agg_id[u] = idx
    end

    return Aggregation(agg_id, n_aggs)
end

"""
    collapse_aggregate(A, X; thr_frac=0.3) -> Aggregation

EXPERIMENTAL structure-adaptive coarsening for expander-like (high-dimensional)
regions, where affinity saturates and pairwise aggregation explodes the operator
complexity. Instead of matching, it CONTRACTS the strongly-converging part of the
graph: an edge `(u,v)` is "flat" if its algebraic distance across the relaxed test
vectors `X`, `√(mean_k (X[u,k]-X[v,k])²)`, is below `thr_frac` × median. The
connected components of the flat-edge subgraph each collapse to ONE coarse node
(the only place transitive aggregation is correct — a flat region is one mode);
non-flat nodes become singletons. The collapsed node's degree is just the region
BOUNDARY, so the internal expander edges vanish instead of densifying.
"""
function collapse_aggregate(A::SparseMatrixCSC, X::AbstractMatrix; thr_frac::Float64 = 0.3)
    n = size(A, 1); col = A.colptr; rv = A.rowval; K = size(X, 2)
    ad(u, v) = sqrt(sum((X[u, t] - X[v, t])^2 for t in 1:K) / K)
    eds = Float64[]
    @inbounds for u in 1:n, k in col[u]:(col[u+1]-1)
        v = rv[k]; v <= u && continue
        push!(eds, ad(u, v))
    end
    isempty(eds) && return Aggregation(collect(1:n), n)
    thr = thr_frac * median(eds)
    fI = Int[]; fJ = Int[]
    @inbounds for u in 1:n, k in col[u]:(col[u+1]-1)
        v = rv[k]; v <= u && continue
        ad(u, v) < thr && (push!(fI, u); push!(fJ, v))
    end
    isempty(fI) && return Aggregation(collect(1:n), n)
    Lf = laplacian(sparse([fI; fJ], [fJ; fI], ones(2length(fI)), n, n))
    lab = connected_components(Lf)
    cs = Dict{Int,Int}(); for l in lab; cs[l] = get(cs, l, 0) + 1; end
    assign = zeros(Int, n); seen = Dict{Int,Int}(); nc = 0
    @inbounds for u in 1:n
        l = lab[u]
        if cs[l] > 1
            haskey(seen, l) || (nc += 1; seen[l] = nc); assign[u] = seen[l]
        else
            nc += 1; assign[u] = nc
        end
    end
    Aggregation(assign, nc)
end
