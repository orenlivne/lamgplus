using LAMG, SparseArrays, LinearAlgebra, Random, Statistics, Printf
include("/Users/oren/code/aier/lamgplus/scripts/mm_loader.jl"); include(joinpath(@__DIR__, "acfhelpers.jl"))
const DATA = "/Users/oren/code/aier/data/suitesparse"
names = split(get(ENV,"GRAPHS","SNAP__web-Stanford.mtx DIMACS10__delaunay_n18.mtx LAW__cnr-2000.mtx SNAP__web-BerkStan.mtx Kamvar__Stanford.mtx DIMACS10__delaunay_n17.mtx"))
variants = [(@sprintf("rebuild K=%d cyc=%d ν=%d", K, c, ν), LAMGOptions(tol=1e-8, max_cycles=200, bootstrap_K=K, bootstrap_cycles=c, bootstrap_ν=ν))
            for K in (8,16), c in (0,1), ν in (3,8)]
for f in names
    W,L = reduce_to_lcc(read_mm_adj(joinpath(DATA,f))...); n=size(L,1)
    bR = randn(MersenneTwister(12), n); bR .-= mean(bR)
    println("\n=== $f n=$n")
    for (lab, o) in variants
        setup(L; options=o); ts = @elapsed h = setup(L; options=o)
        x, info = solve(h, bR; options=o)
        hh = info.rebuilt_hierarchy === nothing ? h : info.rebuilt_hierarchy
        a = acf_homog(hh, L, o)
        @printf("  %-26s setup %.2fs | cyc=%3d solve=%.2fs total=%.2fs boot=%d | ACF homog %.3f | relres %.1e\n",
                lab, ts, info.cycles, info.solve_time, ts+info.solve_time, info.bootstrapped, a.acf, norm(L*x-bR)/norm(bR))
    end
end
