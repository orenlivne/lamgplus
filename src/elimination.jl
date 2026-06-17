"""
Low-degree node elimination — LAMG §4.1.

The "elimination" coarsening removes a set of low-degree, mutually-independent
nodes (the F set) by exact Schur complement, leaving a smaller Laplacian on
the remaining C set. This removes the 1-D parts of the graph (paths, dangling
chains) — exactly where the energy ratio is highest and where aggregation
performs worst.

Multi-stage: repeat until no further nodes qualify, or the graph reaches a
minimum size. Each stage saves its (P, R, q, f, c) so we can interpolate
the eliminated values back during the cycle's post-processing.

Port of `+amg/+setup/CoarseningStrategyElimination.m`, `lowDegreeNodes_matlab.m`,
and `eliminationOperators` (the .m doc + cpp).
"""

"""
    EliminationStage

One stage of low-degree-node elimination. The operator at this stage is

    A_stage  ─ stage `q` ─→  A_stage[c, c] + A_stage[c, f] * P

where `P = A[f,f]^{-1} A[f,c]` viewed as caliber-(|f|/|c|) interpolation.

Fields:
- `f` :: indices (in the stage's local numbering) of the eliminated nodes.
- `c` :: indices of the surviving nodes.
- `n` :: total nodes in the stage's local numbering (= |f| + |c|).
- `P` :: |f| × |c| interpolation `x[f] = P * x[c] + q .* b[f]`.
- `R` :: |c| × |f| restriction (R = Pᵀ when A[f,f] is diagonal; in general
         R = (A[f,f]^{-1} A[f,c])ᵀ).
- `q` :: |f|-vector of inverse diagonal `1 ./ diag(A[f,f])` (used in the
         affine term during interpolation).
"""
struct EliminationStage
    f::Vector{Int}
    c::Vector{Int}
    n::Int
    P::SparseMatrixCSC{Float64,Int}
    R::SparseMatrixCSC{Float64,Int}
    q::Vector{Float64}
end

"""
    low_degree_nodes(A::SparseMatrixCSC; max_degree=4) -> (z, f, c)

Identify (z, f, c) for the elimination stage on graph Laplacian A:
- `z` :: 0-degree (isolated) nodes — eliminated trivially.
- `f` :: low-degree (1..max_degree) nodes that are MUTUALLY INDEPENDENT
        in A (no f-node is a neighbor of another f-node). These can be
        Schur-eliminated in parallel.
- `c` :: the remaining nodes.

Greedy selection ensures `A[f, f]` is diagonal (the elimination is exact and
the diagonal block is trivially invertible).

Port of `lowDegreeNodes_matlab.m`.
"""
function low_degree_nodes(A::SparseMatrixCSC{Float64,Int}; max_degree::Int = 4,
                          hub_min_degree::Int = typemax(Int),
                          fill_tol::Int = -1, fill_hard_cap::Int = max_degree,
                          fill_deg_budget::Int = 64,
                          fill_max_low::Int = typemax(Int))
    n = size(A, 1)
    # Degree = number of off-diagonal nonzeros per row.
    degree = zeros(Int, n)
    rows = rowvals(A); vals = nonzeros(A)
    for j in 1:n
        for k in nzrange(A, j)
            i = rows[k]
            if i != j && vals[k] != 0
                degree[i] += 1
            end
        end
    end
    # Visit codes:
    ZERO_DEGREE = 1; HIGH_DEGREE = 2; LOW_DEGREE = 3; NOT_ELIMINATED = 4
    visited = fill(HIGH_DEGREE, n)
    nbr = Vector{Int}(undef, max(fill_hard_cap, max_degree, 1))
    for i in 1:n
        if degree[i] == 0
            visited[i] = ZERO_DEGREE
        elseif degree[i] >= hub_min_degree && abs(A[i,i]) > 1e-300
            visited[i] = 0          # hub candidate (AC-sampled fill, if enabled).
        elseif degree[i] <= max_degree && abs(A[i,i]) > 1e-300
            # Low-degree exact-elimination candidate. Eliminating a degree-d node adds up
            # to d(d-1)/2 clique edges while removing d, so the NET change is −1 (d=1,2),
            # 0 (d=3) and +2 (d=4): only degree == max_degree (=4) can grow the operator,
            # and only when its neighbours are not already connected. `fill_max_low` gates
            # exactly that case — a degree-max_degree node is eliminated only if it adds
            # ≤ fill_max_low NEW coarse edges (lower-degree nodes are admitted freely).
            if degree[i] < max_degree || fill_max_low == typemax(Int) ||
               _elim_fill_count(A, i, rows, vals, degree, nbr, fill_deg_budget) <= fill_max_low
                visited[i] = 0
            end
        elseif fill_tol >= 0 && max_degree < degree[i] <= fill_hard_cap &&
               abs(A[i,i]) > 1e-300
            # Fill-gated elimination: admit a higher-degree node ONLY if removing it
            # adds ≤ fill_tol new coarse edges (its neighbours are already a near-
            # clique). With τ ≤ 2 the Schur complement stays strictly sparser, so the
            # next coarse graph is both SMALLER and SIMPLER — a coarsening/work win,
            # not just setup. Generalises LOP168 §3.2 (degree-4 elim already relies on
            # "neighbours are usually pre-connected"; here we VERIFY it at any degree).
            if _elim_fill_count(A, i, rows, vals, degree, nbr, fill_deg_budget) <= fill_tol
                visited[i] = 0
            end
        end
    end
    # Greedy pass in node order.
    for i in 1:n
        visited[i] != 0 && continue
        # Inspect neighbors of i (off-diagonal nonzeros of column i, since A is symmetric).
        ok = true
        for k in nzrange(A, i)
            j = rows[k]
            j == i && continue
            vals[k] == 0 && continue
            if visited[j] == LOW_DEGREE
                ok = false
                break
            end
        end
        if ok
            visited[i] = LOW_DEGREE
            for k in nzrange(A, i)
                j = rows[k]
                j == i && continue
                vals[k] == 0 && continue
                if visited[j] != ZERO_DEGREE
                    visited[j] = NOT_ELIMINATED
                end
            end
        else
            visited[i] = NOT_ELIMINATED
        end
    end
    z = findall(==(ZERO_DEGREE), visited)
    f = findall(==(LOW_DEGREE), visited)
    c = findall(v -> v == NOT_ELIMINATED || v == HIGH_DEGREE, visited)
    return z, f, c
end

# Fill created by Schur-eliminating node `u`: number of its neighbour pairs that
# are NOT already edges (= the clique among u's neighbours minus existing edges).
# Returns typemax(Int) the moment a neighbour is a hub (degree > budget) so the
# clique test is O(fill_hard_cap²·budget) per candidate and never runs on hubs
# (Brandt LOP168 §3.2: the neighbour-clique check must stay cheap).
@inline function _elim_fill_count(A::SparseMatrixCSC, u::Int, rows, vals,
                                  degree::Vector{Int}, nbr::Vector{Int},
                                  deg_budget::Int)
    d = 0
    @inbounds for k in nzrange(A, u)
        j = rows[k]; (j == u || vals[k] == 0) && continue
        degree[j] > deg_budget && return typemax(Int)   # hub-adjacent ⇒ don't check
        d += 1
        d > length(nbr) && return typemax(Int)
        nbr[d] = j
    end
    d <= 1 && return 0
    existing = 0
    @inbounds for a in 1:d
        j = nbr[a]
        for k in nzrange(A, j)
            i = rows[k]; (i == j || vals[k] == 0) && continue
            for b in (a + 1):d
                if nbr[b] == i; existing += 1; break; end
            end
        end
    end
    return d * (d - 1) ÷ 2 - existing
end

"""
    elimination_operators(A::SparseMatrixCSC, f::Vector{Int}, c::Vector{Int})
        -> (P::SparseMatrixCSC, R::SparseMatrixCSC, q::Vector{Float64})

Returns:
- `P :: |f| × |c|` interpolation `x[f] = P * x[c] + q .* b[f]`.
- `R :: |c| × |f|` such that the Schur coarse op is `A[c,c] + A[c,f] * P`,
                   equivalently `R = (A[f,f]^{-1} A[f,c])ᵀ = Pᵀ` when
                   `A[f,f]` is diagonal (which is the case by construction
                   of `f` via `low_degree_nodes`).
- `q :: |f|` = `1 ./ diag(A[f,f])`.

Port of `eliminationOperators` (uses the diagonal-`A[f,f]` simplification:
P = -A[f,c] ./ diag(A[f,f])).
"""
function elimination_operators(A::SparseMatrixCSC, f::Vector{Int},
                               c::Vector{Int})
    n = size(A, 1)
    n_f = length(f); n_c = length(c)
    # diag(A[f,f]) — A[f,f] is diagonal by construction.
    Aff_diag = Vector{Float64}(undef, n_f)
    for (k, i) in enumerate(f)
        Aff_diag[k] = A[i, i]
    end
    @assert all(Aff_diag .!= 0) "low_degree_nodes produced an f with a zero-diagonal node"
    q = 1.0 ./ Aff_diag
    # Build A[f, c] explicitly. Map c-index → c-position via a dense scratch
    # vector (0 = not in c) instead of a Dict — same O(1) lookups, far less garbage.
    c_pos = zeros(Int, n)
    for (k, i) in enumerate(c)
        c_pos[i] = k
    end
    rows = rowvals(A); vals = nonzeros(A)
    # Count the off-diagonal nonzeros first so the COO triplet buffers are
    # allocated once at the exact size instead of growing via repeated push!.
    # The triplets are emitted in the identical order with identical values, and
    # f-nodes are independent (no duplicate (row,col) pairs), so the assembled
    # `sparse` is bit-identical to the push!-built version.
    nP = 0
    @inbounds for i in f
        for kp in nzrange(A, i)
            j = rows[kp]
            (j == i || vals[kp] == 0) && continue
            nP += 1
        end
    end
    Prows = Vector{Int}(undef, nP)
    Pcols = Vector{Int}(undef, nP)
    Pvals = Vector{Float64}(undef, nP)
    t = 0
    @inbounds for (k_f, i) in enumerate(f)
        # Off-diagonal nonzeros in row i correspond to column j ∈ c (since
        # f-nodes are independent: no f neighbor).
        for kp in nzrange(A, i)
            j = rows[kp]
            j == i && continue
            v = vals[kp]
            v == 0 && continue
            pos = c_pos[j]
            @assert pos != 0 "row $i has a neighbor $j not in c — f set is not independent"
            t += 1
            Prows[t] = k_f
            Pcols[t] = pos
            Pvals[t] = -v / Aff_diag[k_f]
        end
    end
    P = sparse(Prows, Pcols, Pvals, n_f, n_c)
    R = sparse(P')
    return P, R, q
end

"""
    _schur_blocks(A, c, f) -> (Acc, Acf)

Build the Schur-complement input blocks `A[c, c]` and `A[c, f]` in a SINGLE pass over A's
c/f columns, producing results BIT-IDENTICAL to generic sparse `getindex` (same colptr,
rowval, nzval) — but without the per-entry binary searches that make `A[c,c]`/`A[c,f]`
dominate the elimination stage on large graphs (41+27 ms/stage on web-Google).

`A[c,c]` and `A[c,f]` share the row index set `c`, so one global→local row map `cpos`
(`cpos[c[k]]=k`) serves both; output column k is original column `c[k]` (resp. `f[k]`).
Two passes: count kept entries per column → exact colptrs; then fill in column order. Local
rows emit already-sorted (A's rows-within-column are ascending and `cpos` is monotone in the
original row index *when c is sorted*), so no re-sort is needed. Requires c sorted ascending —
guaranteed by `low_degree_nodes` (findall) and `_apply_fill_cap` (sort); otherwise we fall
back to generic getindex (still correct, just slower).
"""
function _schur_blocks(A::SparseMatrixCSC{Tv,Ti}, c::Vector{Int}, f::Vector{Int}) where {Tv,Ti}
    issorted(c) || return A[c, c], A[c, f]      # bit-identical fast path needs sorted c
    n = size(A, 2); n_c = length(c); n_f = length(f)
    rows = rowvals(A); vals = nonzeros(A)
    cpos = zeros(Ti, n)
    @inbounds for k in 1:n_c; cpos[c[k]] = k; end
    cc_colptr = zeros(Ti, n_c + 1); cf_colptr = zeros(Ti, n_f + 1)
    @inbounds for k in 1:n_c
        cnt = 0
        for p in nzrange(A, c[k]); cpos[rows[p]] != 0 && (cnt += 1); end
        cc_colptr[k + 1] = cnt
    end
    @inbounds for k in 1:n_f
        cnt = 0
        for p in nzrange(A, f[k]); cpos[rows[p]] != 0 && (cnt += 1); end
        cf_colptr[k + 1] = cnt
    end
    cc_colptr[1] = 1; @inbounds for k in 1:n_c; cc_colptr[k + 1] += cc_colptr[k]; end
    cf_colptr[1] = 1; @inbounds for k in 1:n_f; cf_colptr[k + 1] += cf_colptr[k]; end
    cc_rowval = Vector{Ti}(undef, cc_colptr[n_c + 1] - 1); cc_nzval = Vector{Tv}(undef, cc_colptr[n_c + 1] - 1)
    cf_rowval = Vector{Ti}(undef, cf_colptr[n_f + 1] - 1); cf_nzval = Vector{Tv}(undef, cf_colptr[n_f + 1] - 1)
    @inbounds for k in 1:n_c
        dst = cc_colptr[k]
        for p in nzrange(A, c[k])
            lr = cpos[rows[p]]
            lr != 0 && (cc_rowval[dst] = lr; cc_nzval[dst] = vals[p]; dst += 1)
        end
    end
    @inbounds for k in 1:n_f
        dst = cf_colptr[k]
        for p in nzrange(A, f[k])
            lr = cpos[rows[p]]
            lr != 0 && (cf_rowval[dst] = lr; cf_nzval[dst] = vals[p]; dst += 1)
        end
    end
    return SparseMatrixCSC{Tv,Ti}(n_c, n_c, cc_colptr, cc_rowval, cc_nzval),
           SparseMatrixCSC{Tv,Ti}(n_c, n_f, cf_colptr, cf_rowval, cf_nzval)
end

"""
    eliminate_once(A; max_degree=4) -> (stage::Union{EliminationStage,Nothing}, Anext, z)

Perform ONE stage of elimination on Laplacian A. Returns the stage struct,
the Schur-complement Laplacian on the c-set, and the zero-degree set z.
Returns `(nothing, A, Int[])` if no qualifying f-set is found (caller should
stop eliminating).
"""
function eliminate_once(A::SparseMatrixCSC{Float64,Int}; max_degree::Int = 4,
                        min_elim_fraction::Real = 0.01,
                        fill_cap::Real = 0.0,
                        fill_tol::Int = -1, fill_hard_cap::Int = max_degree,
                        fill_deg_budget::Int = 64, fill_max_low::Int = typemax(Int),
                        sample_rho::Real = 0.0, sample_hub_min_degree::Int = 16,
                        rng = nothing)
    n = size(A, 1)
    # HUB-ONLY split: when sampling, the F-set is {deg ≤ max_degree (exact)} ∪
    # {deg ≥ sample_hub_min_degree (AC-sampled hubs)}; the medium band is left for
    # AGGREGATION (low fill). This keeps OC down — only the few hubs get sampled fill.
    sampling = sample_rho > 0 && rng !== nothing
    z, f, c = sampling ?
        low_degree_nodes(A; max_degree = max_degree, hub_min_degree = sample_hub_min_degree,
                         fill_deg_budget = fill_deg_budget, fill_max_low = fill_max_low) :
        low_degree_nodes(A; max_degree = max_degree, fill_tol = fill_tol,
                         fill_hard_cap = fill_hard_cap, fill_deg_budget = fill_deg_budget,
                         fill_max_low = fill_max_low)
    if isempty(f) || length(f) <= min_elim_fraction * n
        return nothing, A, z
    end
    if fill_cap > 0.0 && rng !== nothing && max_degree >= 2 && !sampling
        f, c = _apply_fill_cap(A, f, c, max_degree, fill_cap, rng)
        isempty(f) && return nothing, A, z
    end
    # P, R, q (harmonic extension) are EXACT regardless of sampling — F-point
    # recovery in the cycle stays exact (A_FF is diagonal for an independent set).
    P, R, q = elimination_operators(A, f, c)
    stage = EliminationStage(f, c, n, P, R, q)
    if sampling
        # Only the COARSE OPERATOR's clique fill is approximated (AC sampling),
        # exact for degree ≤ max_degree, KS-sampled above.
        Anext = _eliminate_sampled_fill(A, f, c, max_degree, Float64(sample_rho), rng)
    else
        # FUSED exact Schur: build A[c,c]+A[c,f]*P directly into the coarse storage in a
        # single pass over A — NO A[c,f]*P matmul intermediate, NO separate Acc block, and NO
        # Acc+prod sum (those transients drove the setup GC). _schur_blocks is kept as the
        # bit-identical reference path; the fused kernel inlines the C–C extraction it does.
        # Convergence-equivalent (last-ULP operator diff vs Acc+Acf*P), not bit-identical.
        Anext = _eliminate_fused_fill(A, f, c)   # exact Schur, fused single-pass
    end
    return stage, Anext, z
end

# Append one undirected coarse clique edge (i,j,w>0) to the small fill COO: 2 off-diagonal −w
# entries + accumulate +w into the dense `cd` clique-degree vector.
@inline function _addclique!(I, J, V, cd, i::Int, j::Int, w::Float64)
    push!(I, i); push!(J, j); push!(V, -w)
    push!(I, j); push!(J, i); push!(V, -w)
    cd[i] += w; cd[j] += w
end

# Build the coarse Laplacian after eliminating independent set `f`, sampling (Kyng–Sachdeva,
# low variance) the clique fill of nodes with degree > `exact_cap`, exact for the rest.
# FAST path: take the C–C block natively via Acc = A[c,c] (off-diagonals already correct, diagonal
# = full degree), and ADD a small Δ = (clique off-diagonals) + diag(clique_deg − F_deg), where
# F_deg = rowsum(Acc) = degree lost to eliminated F-neighbours. Anext = Acc + Δ is then the exact
# coarse Laplacian (SPD, row-sums 0) — only the small hub-clique fill is built by hand; the bulk
# C–C block is never reconstructed edge-by-edge. Equals A[c,c]+A[c,f]*P when all-exact.
function _eliminate_sampled_fill(A::SparseMatrixCSC{Float64,Int}, f::Vector{Int},
                                 c::Vector{Int}, exact_cap::Int, rho::Float64, rng)
    n = size(A, 1); inF = falses(n); inF[f] .= true
    cmap = zeros(Int, n); for (k, i) in enumerate(c); cmap[i] = k; end
    nc = length(c); rows = rowvals(A); vals = nonzeros(A)
    Acc = A[c, c]                       # native C–C block (off-diag = −w_CC, diag = full degree)
    fdeg = vec(sum(Acc; dims = 2))      # row sums = F-degree each c-node loses
    cI = Int[]; cJ = Int[]; cV = Float64[]; cd = zeros(nc)   # SMALL clique fill
    nb = Int[]; w = Float64[]
    @inbounds for fi in f                            # clique fill per eliminated node
        empty!(nb); empty!(w)
        for kp in nzrange(A, fi)
            i = rows[kp]; (i == fi || inF[i]) && continue
            wi = -vals[kp]; wi > 0 && (push!(nb, cmap[i]); push!(w, wi))
        end
        d = length(nb); d < 2 && continue
        wf = sum(w)
        if d <= exact_cap
            for a in 1:d, b in (a+1):d
                _addclique!(cI, cJ, cV, cd, nb[a], nb[b], w[a]*w[b]/wf)
            end
        else
            reps = max(1, round(Int, rho))
            for _ in 1:reps
                ord = Random.randperm(rng, d); cums = zeros(d); s = 0.0
                for t in 1:d; s += w[ord[t]]; cums[t] = s; end
                csum = cums[d]; csum <= 0 && continue
                for idx in 1:(d-1)
                    a = ord[idx]; wa = w[a]; Wrem = csum - cums[idx]; Wrem <= 0 && continue
                    r = rand(rng)*Wrem + cums[idx]
                    ko = searchsortedfirst(cums, r); ko = min(max(ko, idx+1), d)
                    _addclique!(cI, cJ, cV, cd, nb[a], nb[ord[ko]], (wa*Wrem/csum)/reps)
                end
            end
        end
    end
    # Δ diagonal correction: clique_deg − F_deg (cancels Acc's surplus F-degree, adds clique degree).
    @inbounds for k in 1:nc; push!(cI, k); push!(cJ, k); push!(cV, cd[k] - fdeg[k]); end
    return Acc + sparse(cI, cJ, cV, nc, nc)   # exact coarse Laplacian (SPD, row-sums 0)
end

# FUSED EXACT Schur complement — the un-sampled analogue of `_eliminate_sampled_fill`.
# Computes Anext = A[c,c] + A[c,f]*P in a SINGLE Gustavson-style pass that writes the result
# DIRECTLY into one preallocated CSC. NOTHING is materialised in between: no A[c,c] block, no
# A[c,f] block, no A[c,f]*P matmul, and no Acc+prod sum — those four transients (the matmul
# intermediate chief among them) drove the setup GC. The C–C block is read straight out of A;
# every clique edge is added EXACTLY, none sampled:
#
#   (A[c,f]*P)[I,J] = Σ_k A[c_I,f_k]·(−A[f_k,c_J]/A[f_k,f_k]) = −Σ_k w(c_I,f_k)·w(c_J,f_k)/A[f_k,f_k]
#
# i.e. eliminating f-node k adds the clique edge weight w_I·w_J/A[f_k,f_k] over each of its
# ordered C-neighbour pairs; the I==J term supplies exactly the diagonal Schur correction, so the
# coarse operator stays a valid Laplacian (symmetric, row-sums 0) without a separate diagonal fix-up.
# Per coarse column J (original node c_J): the C-rows of A's column c_J ARE the A[c,c] column
# (off-diagonals + full diagonal), and the f-rows of that same column are the eliminated neighbours
# whose cliques feed the fill (A symmetric). The denominator is the LITERAL A[f_k,f_k] and ALL
# off-diagonal neighbours contribute (no weight-sign filter), so the result matches `Acc + Acf*P`
# to ~machine precision — only the summation ORDER differs ⇒ last-ULP, convergence-equivalent (NOT
# bit-identical). Requires c sorted ascending (true via low_degree_nodes / _apply_fill_cap).
function _eliminate_fused_fill(A::SparseMatrixCSC{Float64,Int}, f::Vector{Int},
                               c::Vector{Int})
    n = size(A, 1); nc = length(c)
    rows = rowvals(A); vals = nonzeros(A)
    cmap = zeros(Int, n); @inbounds for (k, i) in enumerate(c); cmap[i] = k; end
    inF = falses(n); @inbounds for fi in f; inF[fi] = true; end
    # Inverse f-diagonal, indexed by ORIGINAL node id (only f-rows touched), so the per-column
    # passes can look it up in O(1) without a binary search into A.
    qfull = zeros(n); @inbounds for fi in f; qfull[fi] = 1.0 / A[fi, fi]; end

    # ---- SYMBOLIC pass: exact nnz per output column = (C-rows of col c_J) ∪ (clique fill rows). ----
    marker = zeros(Int, nc)                       # marker[I]==J ⇒ row I already seen in column J
    colptr = Vector{Int}(undef, nc + 1); colptr[1] = 1
    @inbounds for J in 1:nc
        cnt = 0
        cj = c[J]
        for p in nzrange(A, cj)
            I0 = rows[p]; vals[p] == 0 && continue
            if inF[I0]                            # eliminated neighbour ⇒ expand its clique rows
                fk = I0
                for pp in nzrange(A, fk)
                    Ic = rows[pp]; (Ic == fk || inF[Ic] || vals[pp] == 0) && continue
                    I = cmap[Ic]
                    if marker[I] != J; marker[I] = J; cnt += 1; end
                end
            else                                  # C-row of A[c,c] column J (incl. diagonal Ic==cj)
                I = cmap[I0]
                if marker[I] != J; marker[I] = J; cnt += 1; end
            end
        end
        colptr[J+1] = colptr[J] + cnt
    end

    nnzC = colptr[nc+1] - 1
    rowval = Vector{Int}(undef, nnzC)
    nzval  = Vector{Float64}(undef, nnzC)

    # ---- NUMERIC pass: dense SPA accumulate (A[c,c] column + clique fill), emit each column sorted. ----
    spa = zeros(nc); touched = Vector{Int}(undef, nc); fill!(marker, 0)
    @inbounds for J in 1:nc
        nt = 0
        cj = c[J]
        for p in nzrange(A, cj)
            I0 = rows[p]; vals[p] == 0 && continue
            if inF[I0]
                # Clique fill from eliminated neighbour fk=I0: −w(c_I,fk)·w(c_J,fk)/A[fk,fk].
                fk = I0; coef = (-vals[p]) * qfull[fk]
                for pp in nzrange(A, fk)
                    Ic = rows[pp]; (Ic == fk || inF[Ic] || vals[pp] == 0) && continue
                    I = cmap[Ic]; v = vals[pp] * coef   # (−w_I)·(w_J/A_ff) = −w_I·w_J/A_ff
                    if marker[I] == J
                        spa[I] += v
                    else
                        marker[I] = J; spa[I] = v; nt += 1; touched[nt] = I
                    end
                end
            else
                # Native A[c,c] entry (off-diagonal −w_CC, or the full-degree diagonal at Ic==cj).
                I = cmap[I0]
                if marker[I] == J
                    spa[I] += vals[p]
                else
                    marker[I] = J; spa[I] = vals[p]; nt += 1; touched[nt] = I
                end
            end
        end
        # Emit column J in ascending row order (CSC canonical form).
        sort!(view(touched, 1:nt))
        dst = colptr[J]
        for t in 1:nt
            I = touched[t]; rowval[dst] = I; nzval[dst] = spa[I]; dst += 1
        end
    end
    return SparseMatrixCSC{Float64,Int}(nc, nc, colptr, rowval, nzval)
end

"""
    _apply_fill_cap(A, f, c, max_degree, fill_cap, rng) -> (f_kept, c_new)

Sub-sample the degree-`max_degree` slice of `f` so that the projected
post-elimination nnz/n on `c` does not exceed `fill_cap × nnz/n(A)`.

Lower-degree f-nodes (degree < max_degree) are always retained because
they fill less. We randomly drop a fraction of the degree-`max_degree`
candidates and move them to `c`.

Heuristic: each eliminated degree-d node contributes approximately
`d × (d−1) / 2` extra coarse edges (the upper bound is the clique among
its d neighbours). The actual extra fill is bounded by:
    Δnnz ≤ Σ_{f-node u with degree d_u} d_u(d_u−1)
and the resulting nnz/n on survivors:
    new_nnz_per_row ≈ (nnz_A − 2 |edges_to_f|) / |c|
                      + (Δnnz from cliques) / |c|.

We compute the projected new nnz/n with the FULL `f` and the
NO-elimination case; if the ratio exceeds `fill_cap`, we randomly drop
degree-`max_degree` candidates one by one (in random order) until under
the cap. Cheap O(|f|) heuristic.
"""
function _apply_fill_cap(A::SparseMatrixCSC, f::Vector{Int}, c::Vector{Int},
                         max_degree::Int, fill_cap::Real, rng)
    n = size(A, 1); nnz_A = nnz(A)
    nnz_per_row_orig = nnz_A / n
    # Recompute degrees of f.
    deg_f = zeros(Int, length(f))
    rows = rowvals(A); vals = nonzeros(A)
    @inbounds for (k, u) in enumerate(f)
        d = 0
        for kp in nzrange(A, u)
            i = rows[kp]
            i != u && vals[kp] != 0 && (d += 1)
        end
        deg_f[k] = d
    end
    # Estimate fill: every eliminated degree-d node contributes ≤ d(d−1) extra
    # off-diagonal entries (cliquing its d neighbours symmetrically).
    function projected_nnz_per_row(keep_mask::Vector{Bool})
        n_c = length(c) + count(.!keep_mask)
        # nnz on c = nnz_A minus rows/cols belonging to dropped f-nodes,
        # plus fill from kept f-nodes.
        edges_to_f_kept = sum(deg_f[i] for i in eachindex(keep_mask) if keep_mask[i]; init = 0)
        fill = sum(deg_f[i] * (deg_f[i] - 1) for i in eachindex(keep_mask) if keep_mask[i]; init = 0)
        # Conservative: subtract twice each f-incident edge (rows + cols).
        new_nnz = nnz_A - 2 * edges_to_f_kept + fill
        return new_nnz / max(n_c, 1)
    end
    target = fill_cap * nnz_per_row_orig
    keep = trues(length(f))
    pnr = projected_nnz_per_row(keep)
    if pnr <= target
        return f, c
    end
    # Drop degree-max_degree first; lower-degree stays.
    top_indices = findall(d -> d == max_degree, deg_f)
    Random.shuffle!(rng, top_indices)
    for idx in top_indices
        keep[idx] = false
        pnr = projected_nnz_per_row(keep)
        pnr <= target && break
    end
    f_kept = f[keep]
    f_drop = f[.!keep]
    c_new = sort(vcat(c, f_drop))
    return f_kept, c_new
end
