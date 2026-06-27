# Generate the data for a LAMG+ hierarchical-coarsening figure on a SOCIAL graph
# (analogous to doc/figures/coarsening_airfoil.png, but for a community-structured network).
# Builds a pure-aggregation 4-level hierarchy and writes, for every fine node, which aggregate it
# belongs to at each level. Usage:
#   julia --project=<env-with-LAMG> scripts/fig_coarsening_social.jl <graph.mtx> <out_prefix>
using LAMG, SparseArrays, LinearAlgebra, Printf

# ---- minimal MatrixMarket coordinate reader -> symmetric adjacency (largest connected component) ----
function read_mtx_adj(path)
    started = false; sym = occursin("symmetric", lowercase(readline(path)))
    I = Int[]; J = Int[]; n = 0
    for ln in eachline(path)
        (isempty(ln) || ln[1] == '%') && continue
        t = split(ln)
        if !started
            n = parse(Int, t[1]); started = true; continue
        end
        length(t) < 2 && continue
        i = parse(Int, t[1]); j = parse(Int, t[2]); i == j && continue
        push!(I, i); push!(J, j); push!(I, j); push!(J, i)   # symmetrize
    end
    A = sparse(I, J, 1.0, n, n); fill!(A.nzval, 1.0); A = max.(A, A')
    # largest connected component
    comp = fill(0, n); nc = 0
    rows = rowvals(A)
    for s in 1:n
        comp[s] == 0 || continue; nc += 1; comp[s] = nc; q = [s]
        while !isempty(q)
            u = popfirst!(q)
            for k in nzrange(A, u); v = rows[k]; comp[v] == 0 && (comp[v] = nc; push!(q, v)); end
        end
    end
    best = argmax([count(==(c), comp) for c in 1:nc]); keep = findall(==(best), comp)
    A[keep, keep], keep
end

path = ARGS[1]; outpref = ARGS[2]
A, keep = read_mtx_adj(path)
n = size(A, 1)
@printf("graph: %s  n(LCC)=%d  edges=%d\n", basename(path), n, nnz(A) ÷ 2)

# ---- 4-level pure-aggregation hierarchy; compose fine->level-k aggregate labels ----
L = laplacian(A)
cur = collect(1:n)                 # fine node -> current-level node (starts identity)
labels = zeros(Int, n, 4); ncoarse = Int[]
Lk = L
for k in 1:4
    agg = aggregate(Lk)            # current-level node -> next-coarser aggregate id
    m = length(agg.aggregate)
    global cur = agg.aggregate[cur]
    labels[:, k] = cur
    push!(ncoarse, agg.n_coarse)
    P = sparse(1:m, agg.aggregate, 1.0, m, agg.n_coarse)   # 0/1 aggregation
    global Lk = P' * Lk * P                                 # Galerkin coarse Laplacian
    @printf("  level %d: %d aggregates (%.1fx coarser)\n", k, agg.n_coarse, n/agg.n_coarse)
end

# ---- write edges (fine, 0-based) + per-level labels + meta for the python plotter ----
open(outpref * "_edges.csv", "w") do io
    rows = rowvals(A)
    for j in 1:n, k in nzrange(A, j); i = rows[k]; i < j && @printf(io, "%d,%d\n", i-1, j-1); end
end
open(outpref * "_labels.csv", "w") do io
    println(io, "L1,L2,L3,L4")
    for i in 1:n; @printf(io, "%d,%d,%d,%d\n", labels[i,1], labels[i,2], labels[i,3], labels[i,4]); end
end
open(outpref * "_meta.csv", "w") do io
    println(io, "n,$(n)"); println(io, "edges,$(nnz(A)÷2)")
    for k in 1:4; println(io, "ncoarse$k,$(ncoarse[k])"); end
end
println("wrote ", outpref, "_{edges,labels,meta}.csv")
