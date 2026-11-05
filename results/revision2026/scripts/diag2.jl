using LAMG, SparseArrays, LinearAlgebra, Random, Statistics, Printf
include("/Users/oren/code/aier/lamgplus/scripts/mm_loader.jl"); include(joinpath(@__DIR__, "acfhelpers.jl"))
const DATA = "/Users/oren/code/aier/data/suitesparse"
function aggstats(h)
    for (i,l) in enumerate(h.levels)
        l.level_type == :agg || continue
        sz = diff(sparse(l.p').colptr) # rows per coarse col? no: p is n×nc; count members per aggregate via R
        cnt = zeros(Int, size(l.p,2)); for (r,c,v) in zip(findnz(l.p)...); v == 1.0 && (cnt[c]+=1); end
        return @sprintf("L%d agg sizes: mean %.2f max %d  |1|=%d%% |≥6|=%d%%", i, mean(cnt), maximum(cnt), round(Int,100*count(==(1),cnt)/length(cnt)), round(Int,100*count(>=(6),cnt)/length(cnt)))
    end
    ""
end
names = split(get(ENV,"GRAPHS","SNAP__web-Stanford.mtx DIMACS10__delaunay_n18.mtx DIMACS10__delaunay_n17.mtx LAW__cnr-2000.mtx"))
variants = [("default", Dict()), ("scan-aff", Dict(:agg_scan_by_affinity=>true)), ("K=8", Dict(:agg_K=>8)),
            ("scan-aff+K8", Dict(:agg_scan_by_affinity=>true, :agg_K=>8)), ("no-reorder", Dict(:reorder=>false)),
            ("scan-aff+no-reorder", Dict(:agg_scan_by_affinity=>true, :reorder=>false))]
for f in names
    W,L = reduce_to_lcc(read_mm_adj(joinpath(DATA,f))...); n=size(L,1)
    bR = randn(MersenneTwister(12), n); bR .-= mean(bR)
    println("\n=== $f n=$n")
    for (lab, kw) in variants
        o = LAMGOptions(; tol=1e-8, max_cycles=200, kw...)
        setup(L; options=o); ts = @elapsed h = setup(L; options=o)
        x, info = solve(h, bR; options=o)
        a = acf_homog(h, L, o)
        # two-level: truncate at first agg level
        k = findfirst(l -> l.level_type == :agg, h.levels)
        a2 = k === nothing ? (acf=NaN,) : acf_homog(LAMG.Multilevel(h.levels[1:k], h.perm), L, o; ncyc=25)
        @printf("  %-20s setup %.2fs lv=%2d cx=%.2f | RAND cyc=%3d t=%.2fs esc=%d | ACF homog %.3f (2-level %.3f) | %s\n",
                lab, ts, length(h), sum(nnz(l.a) for l in h.levels)/nnz(L), info.cycles, info.solve_time, info.gamma_escalated, a.acf, a2.acf, aggstats(h))
    end
end
