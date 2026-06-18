# Aggregate filtering: bad-vertex removal (§4.3.1) and subgroup extraction (§4.3.2),
# plus the per-aggregate quality-control orchestration (Napov–Notay 2017, §4).
#
# κ̄ (kappa-bar) = quality threshold (paper default 10); η = security factor (=2).

# --- Matrix-free quality kernels -------------------------------------------------
# These scan the sparse rows of A directly using a membership flag `inS` (a length-n
# Bool buffer, true exactly for the current aggregate), computing the per-vertex
# external/internal weight sums and root coupling without ever forming a dense
# subgraph matrix. This is the allocation-critical inner loop (hub aggregates can
# hold thousands of vertices; a dense |G|×|G| per sweep is fatal).

"""
    bad_vertex_removal_core!(A, S, root, δ, inS, scratch; κbar, η) -> S

In-place, matrix-free bad-vertex removal (§4.3.1). Assumes `inS[v]` is `true`
exactly for `v ∈ S` on entry; mutates `S` (shrinks it) and `inS` (clears removed
vertices) so both stay consistent. `scratch` is a reusable `Vector{Int}`.
"""
function bad_vertex_removal_core!(A::SparseMatrixCSC, S::Vector{Int}, root::Int,
                                  δ::AbstractVector, inS::AbstractVector{Bool},
                                  scratch::Vector{Int}; κbar::Real=10.0, η::Real=2.0)
    rv = rowvals(A); nz = nonzeros(A)
    c18 = κbar - 1; c19 = (κbar - 1) / η
    while true
        empty!(scratch)
        ng = length(S)
        smallenough = ng <= 1024
        for j in S
            j == root && continue
            ext = 0.0; intd = 0.0; ajr = 0.0
            @inbounds for idx in nzrange(A, j)
                k = rv[idx]; k == j && continue
                w = -nz[idx]
                inS[k] ? (intd += w) : (ext += w)
                k == root && (ajr = w)
            end
            num = 2 * ext + δ[j]
            keep = (ajr > 0.0 && num <= c18 * ajr) || (smallenough && intd > 0.0 && num <= c19 * intd)
            keep || push!(scratch, j)
        end
        isempty(scratch) && break
        @inbounds for j in scratch; inS[j] = false; end
        filter!(v -> inS[v], S)
        length(S) <= 1 && break
    end
    return S
end

"""
    criterion16_mf(A, S, root, δ, inS, κbar) -> Bool

Matrix-free cheap quality criterion (eq. 16): `(2 Σ_{k∉S}|a_jk| + δ_j) ≤ (κ̄−1)|a_jr|`
for every non-root `j`. Assumes `inS` marks `S`.
"""
function criterion16_mf(A::SparseMatrixCSC, S::Vector{Int}, root::Int,
                        δ::AbstractVector, inS::AbstractVector{Bool}, κbar::Real)
    rv = rowvals(A); nz = nonzeros(A); c = κbar - 1
    for j in S
        j == root && continue
        ext = 0.0; ajr = 0.0
        @inbounds for idx in nzrange(A, j)
            k = rv[idx]; k == j && continue
            w = -nz[idx]
            inS[k] || (ext += w)
            k == root && (ajr = w)
        end
        ajr <= 0.0 && return false
        (2 * ext + δ[j]) > c * ajr && return false
    end
    return true
end

"""
    bad_vertex_removal(A, members, root, δ; κbar=10.0, η=2.0) -> Vector{Int}

Allocating convenience wrapper around `bad_vertex_removal_core!` (for tests).
"""
function bad_vertex_removal(A::SparseMatrixCSC, members::AbstractVector{<:Integer},
                            root::Integer, δ::AbstractVector; κbar::Real=10.0, η::Real=2.0)
    inS = falses(size(A, 1)); for v in members; inS[Int(v)] = true; end
    S = collect(Int, members)
    bad_vertex_removal_core!(A, S, Int(root), δ, inS, Int[]; κbar=κbar, η=η)
    return S
end

"""
    subgroup_signsplit(A, members, root, δ) -> Gp

Preselect `Gp` ⊆ `members`: the vertices whose entry in the generalized Fiedler
vector (eigenvector of the 2nd-smallest generalized eigenvalue of `A_G z = λ X_G z`,
eq. 21) has the same sign as the root's (§4.3.2). We compute this eigenvector
exactly; the paper approximates it cheaply by a byproduct of the broken Cholesky
(`v = w + α1`), which it identifies as an approximation of this same vector.
"""
function subgroup_signsplit(A::SparseMatrixCSC, members::AbstractVector{<:Integer},
                            root::Integer, δ::AbstractVector)
    AG, γ, XG = aggregate_quality_matrices(A, members, δ)
    XGr = Symmetric(Matrix(XG) + 1e-12 * I)          # guard PD for the pencil
    F = eigen(Symmetric(AG), XGr)
    v = F.vectors[:, 2]                              # second-smallest gen-eigenvalue
    rloc = findfirst(==(Int(root)), members)
    posroot = v[rloc] >= 0
    return [Int(members[p]) for p in 1:length(members) if (v[p] >= 0) == posroot]
end

"""
    subgroup_extract(A, members, Gp, root, δ; κbar=10.0, η=2.0) -> Vector{Int}

New tentative aggregate `{root} ∪ {j ∈ members∖{root} : (22) w.r.t. Gp}` (§4.3.2,
eq. 22). Membership sums are taken relative to the preselected set `Gp`; vertices
not in `Gp` may still be kept if they satisfy (22).
"""
function subgroup_extract(A::SparseMatrixCSC, members::AbstractVector{<:Integer},
                          Gp::AbstractVector{<:Integer}, root::Integer, δ::AbstractVector;
                          κbar::Real=10.0, η::Real=2.0)
    inGp = Set(Int.(Gp))
    rv = rowvals(A); nz = nonzeros(A)
    new = Int[Int(root)]
    for j in members
        Int(j) == Int(root) && continue
        extj = 0.0; intj = 0.0; ajr = 0.0
        for idx in nzrange(A, Int(j))
            k = rv[idx]; k == Int(j) && continue
            w = -nz[idx]                             # |a_jk|
            if k in inGp; intj += w; else; extj += w; end
            k == Int(root) && (ajr = w)
        end
        num = 2 * extj + δ[Int(j)]
        keep = (ajr > 0 && num <= (κbar - 1) * ajr) || (intj > 0 && num <= ((κbar - 1) / η) * intj)
        keep && push!(new, Int(j))
    end
    return new
end

"""
    refine_aggregate!(A, members, root, δ, inS, scratch; κbar=10.0, η0=2.0, maxdepth=8) -> Vector{Int}

Full per-aggregate quality control (§4.5, Algorithm 3 inner loop), matrix-free and
buffer-reusing: bad-vertex removal → cheap criterion 16 (matrix-free) → for small
aggregates only, the eq.14 Cholesky test → on failure, subgroup extraction with
η escalating by 0.5, then repeat. `inS` (length-n Bool) and `scratch` (`Vector{Int}`)
are caller-owned reusable buffers; `inS` is left fully cleared on return. Returns the
accepted aggregate (always containing the root).
"""
function refine_aggregate!(A::SparseMatrixCSC, members::AbstractVector{<:Integer},
                           root::Integer, δ::AbstractVector, inS::AbstractVector{Bool},
                           scratch::Vector{Int}; κbar::Real=10.0, η0::Real=2.0, maxdepth::Int=8)
    rt = Int(root)
    S = collect(Int, members)
    for v in S; inS[v] = true; end
    try
        η = η0
        for _ in 0:maxdepth
            bad_vertex_removal_core!(A, S, rt, δ, inS, scratch; κbar=κbar, η=η)
            length(S) <= 1 && return copy(S)
            length(S) > 1024 && return copy(S)                       # crit 16 holds by construction
            criterion16_mf(A, S, rt, δ, inS, κbar) && return copy(S)
            AG, γ, XG = aggregate_quality_matrices(A, S, δ)          # dense path: small + rare
            quality_ok_factor(AG, XG, κbar) && return copy(S)
            Gp = subgroup_signsplit(A, S, rt, δ)
            new = subgroup_extract(A, S, Gp, rt, δ; κbar=κbar, η=η + 0.5)
            length(new) >= length(S) && return copy(S)               # no progress
            ns = Set(new)
            for v in S; (v in ns) || (inS[v] = false); end
            empty!(S); append!(S, new)
            η += 0.5
        end
        return copy(S)
    finally
        for v in members; inS[Int(v)] = false; end
        for v in S; inS[v] = false; end
    end
end

"""
    refine_aggregate(A, members, root, δ; κbar=10.0, η=2.0, maxdepth=4) -> Vector{Int}

Allocating convenience wrapper around `refine_aggregate!` (for tests).
"""
function refine_aggregate(A::SparseMatrixCSC, members::AbstractVector{<:Integer},
                          root::Integer, δ::AbstractVector;
                          κbar::Real=10.0, η::Real=2.0, maxdepth::Int=4)
    inS = falses(size(A, 1))
    return refine_aggregate!(A, members, root, δ, inS, Int[]; κbar=κbar, η0=η, maxdepth=maxdepth)
end
