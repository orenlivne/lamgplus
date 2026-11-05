using LAMG, SparseArrays, LinearAlgebra, Random, Statistics, Printf
include("/Users/oren/code/aier/lamgplus/scripts/mm_loader.jl"); include(joinpath(@__DIR__, "acfhelpers.jl"))
const DATA = "/Users/oren/code/aier/data/suitesparse"
names = split(get(ENV,"GRAPHS","SNAP__web-Stanford.mtx DIMACS10__delaunay_n18.mtx DIMACS10__delaunay_n17.mtx LAW__cnr-2000.mtx SNAP__web-BerkStan.mtx Kamvar__Stanford.mtx"))
for f in names
    W,L = reduce_to_lcc(read_mm_adj(joinpath(DATA,f))...); n=size(L,1)
    bR = randn(MersenneTwister(12), n); bR .-= mean(bR)
    xt = randn(MersenneTwister(11), n); xt .-= mean(xt); bLX = L*xt
    println("\n=== $f n=$n")
    for (lab, o) in (("bootstrap off (legacy restart)", LAMGOptions(tol=1e-8, max_cycles=200, bootstrap=false)),
                     ("bootstrap on (default)", LAMGOptions(tol=1e-8, max_cycles=200)),
                     ("bootstrap on, 3 cycles", LAMGOptions(tol=1e-8, max_cycles=200, bootstrap_cycles=3)),
                     ("bootstrap on, thr 0.4", LAMGOptions(tol=1e-8, max_cycles=200, bootstrap_threshold=0.4)))
        setup(L; options=o); ts = @elapsed h = setup(L; options=o)
        for (rl, b) in (("RAND", bR), ("LX", bLX))
            x, info = solve(h, b; options=o)
            hh = info.rebuilt_hierarchy === nothing ? h : info.rebuilt_hierarchy
            a = acf_homog(hh, L, o)
            @printf("  %-32s %-4s setup %.2fs | cyc=%3d t=%.2fs boot=%d | ACF homog (final hierarchy) %.3f | relres %.1e\n",
                    lab, rl, ts, info.cycles, info.solve_time, info.bootstrapped, a.acf, norm(L*x-b)/norm(b))
        end
    end
end
