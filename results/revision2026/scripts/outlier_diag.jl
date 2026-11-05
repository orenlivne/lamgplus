using LAMG, SparseArrays, LinearAlgebra, Random, Statistics, Printf
include("/Users/oren/code/aier/lamgplus/scripts/mm_loader.jl"); include(joinpath(@__DIR__, "acfhelpers.jl"))
const DATA = "/Users/oren/code/aier/data/suitesparse"
names = split(get(ENV,"GRAPHS","Rajat__rajat29.mtx SNAP__web-BerkStan.mtx Kamvar__Stanford_Berkeley.mtx"))
fmt(v) = join([@sprintf("%.2f", x) for x in v], " ")
for f in names
    W,L = reduce_to_lcc(read_mm_adj(joinpath(DATA,f))...); n=size(L,1)
    bR = randn(MersenneTwister(12), n); bR .-= mean(bR)
    println("\n=== $f n=$n")
    o = LAMGOptions(tol=1e-8, max_cycles=200)
    h = setup(L; options=o); x, info = solve(h, bR; options=o)
    println("  default: cyc=$(info.cycles) esc=$(info.tv_escalated) residual factors: ", fmt(info.conv_factors))
    a = acf_homog(h, L, o); println("  homog factors (default hierarchy, $(a.valid) valid of $(a.cycles) cyc, acf $(round(a.acf,digits=3))): ", fmt(a.factors))
    for (K, mx) in ((8,1), (16,1), (8,2))
        o2 = LAMGOptions(tol=1e-8, max_cycles=200, escalate_tv_threshold=0.0, escalate_tv_K=K, escalate_tv_max=mx)
        h2 = setup(L; options=o2); ts = @elapsed h2 = setup(L; options=o2)
        x2, i2 = solve(h2, bR; options=o2); hh = i2.rebuilt_hierarchy === nothing ? h2 : i2.rebuilt_hierarchy
        a2 = acf_homog(hh, L, o2)
        @printf("  forced K=%2d max=%d: cyc=%3d solve=%.2fs esc=%d | homog ACF %.3f (%d valid of %d cyc) | relres %.1e\n", K, mx, i2.cycles, i2.solve_time, i2.tv_escalated, a2.acf, a2.valid, a2.cycles, norm(L*x2-bR)/norm(bR))
    end
end
