# Aggregate filtering: bad-vertex removal (§4.3.1) and subgroup extraction (§4.3.2),
# plus the per-aggregate quality-control orchestration (Napov–Notay 2017, §4).
#
# κ̄ (kappa-bar) = quality threshold (paper default 10); η = security factor (=2).

"""
    bad_vertex_removal(A, members, root, δ; κbar=10.0, η=2.0) -> Vector{Int}

Sweep the non-root vertices of tentative aggregate `members` (global indices),
removing any vertex that satisfies neither keep-criterion (18) nor (19), and
repeat until a full sweep removes nothing (§4.3.1). Sums are recomputed against
the *current* membership each sweep. The root is always retained.
"""
function bad_vertex_removal(A::SparseMatrixCSC, members::AbstractVector{<:Integer},
                            root::Integer, δ::AbstractVector; κbar::Real=10.0, η::Real=2.0)
    S = collect(Int, members)
    while true
        AG, ext = subgraph_laplacian(A, S)
        rloc = findfirst(==(Int(root)), S)
        ng = length(S)
        remove = Int[]
        for p in 1:ng
            S[p] == root && continue
            num = 2 * ext[p] + δ[S[p]]
            ajr = -AG[p, rloc]                       # |a_{jr}| (0 if not adjacent)
            keep18 = ajr > 0 && num <= (κbar - 1) * ajr
            intp = AG[p, p]                          # internal degree Σ_{k∈S}|a_jk|
            keep19 = (ng <= 1024) && intp > 0 && num <= ((κbar - 1) / η) * intp
            (keep18 || keep19) || push!(remove, S[p])
        end
        isempty(remove) && break
        S = setdiff(S, remove)
        length(S) <= 1 && break
    end
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
    refine_aggregate(A, members, root, δ; κbar=10.0, η=2.0, maxdepth=4) -> Vector{Int}

Full per-aggregate quality control (§4.5, Algorithm 3 inner loop): bad-vertex
removal → quality test (cheap criterion 16, else the eq.14 Cholesky factorization
for |G|≤1024) → on failure, subgroup extraction, then recurse. Returns the
accepted aggregate (global indices), always containing the root.
"""
function refine_aggregate(A::SparseMatrixCSC, members::AbstractVector{<:Integer},
                          root::Integer, δ::AbstractVector;
                          κbar::Real=10.0, η::Real=2.0, maxdepth::Int=4)
    S = bad_vertex_removal(A, members, root, δ; κbar=κbar, η=η)
    length(S) <= 1 && return S
    if length(S) > 1024
        return S                                     # criterion 16 holds by construction
    end
    AG, γ, XG = aggregate_quality_matrices(A, S, δ)
    if criterion16(A, S, root, δ, κbar) || quality_ok_factor(AG, XG, κbar)
        return S
    end
    maxdepth <= 0 && return S
    Gp = subgroup_signsplit(A, S, root, δ)
    new = subgroup_extract(A, S, Gp, root, δ; κbar=κbar, η=η)
    length(new) >= length(S) && return S             # no progress ⇒ stop
    return refine_aggregate(A, new, root, δ; κbar=κbar, η=η, maxdepth=maxdepth - 1)
end
