using LAMG, SparseArrays, LinearAlgebra, Random, Statistics, Printf
include("/Users/oren/code/aier/lamgplus/scripts/mm_loader.jl"); include(joinpath(@__DIR__, "acfhelpers.jl"))
const DATA = "/Users/oren/code/aier/data/suitesparse"
names = split(get(ENV,"GRAPHS","SNAP__web-BerkStan.mtx Kamvar__Stanford_Berkeley.mtx SNAP__web-Stanford.mtx LAW__cnr-2000.mtx DIMACS10__delaunay_n18.mtx Rajat__rajat29.mtx"))
fmt(v) = join([@sprintf("%.2f", x) for x in v], " ")
function run1(lab, L, bR, o)
    h = setup(L; options=o); ts = @elapsed h = setup(L; options=o)
    x, i = solve(h, bR; options=o); hh = i.rebuilt_hierarchy === nothing ? h : i.rebuilt_hierarchy
    a = acf_homog(hh, L, o)
    @printf("  %-28s setup %.2fs cyc=%3d solve=%.2fs esc=%d rev=%d | homog ACF %.3f (%d valid) | relres %.1e | f: %s\n",
            lab, ts, i.cycles, i.solve_time, i.tv_escalated, i.tv_reverted, a.acf, a.valid, norm(L*x-bR)/norm(bR), fmt(i.conv_factors))
end
for f in names
    W,L = reduce_to_lcc(read_mm_adj(joinpath(DATA,f))...); n=size(L,1)
    bR = randn(MersenneTwister(12), n); bR .-= mean(bR)
    println("\n=== $f n=$n")
    run1("new default (thr .3, guard)", L, bR, LAMGOptions(tol=1e-8, max_cycles=200))
    run1("old (escalate_tv=false)", L, bR, LAMGOptions(tol=1e-8, max_cycles=200, escalate_tv=false))
    # falsifier: K=4 with new seeds — sampling variance or K?
    for sd in (1,2,3)
        run1("forced K=4 seed+$sd", L, bR, LAMGOptions(tol=1e-8, max_cycles=200, escalate_tv_threshold=0.0, escalate_tv_K=4, seed=UInt(0xfa11+sd), escalate_tv_guard=false))
    end
end
