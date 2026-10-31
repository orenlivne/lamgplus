using LAMG, SparseArrays, LinearAlgebra, Random, Statistics, Printf
include("/Users/oren/code/aier/lamgplus/scripts/mm_loader.jl")
const DATA = "/Users/oren/code/aier/data/suitesparse"
for k in (17,18,19), ml in (20, 40)
    W,L = reduce_to_lcc(read_mm_adj(joinpath(DATA,"DIMACS10__delaunay_n$k.mtx"))...); n=size(L,1)
    o = LAMGOptions(tol=1e-8, max_cycles=100, max_levels=ml); h = setup(L; options=o)
    sizes = [size(l.a,1) for l in h.levels]; types = [string(l.level_type)[1] for l in h.levels]
    b = L*(randn(MersenneTwister(11),n) |> v->v .- mean(v)); x,info = solve(h,b;options=o)
    cf = info.conv_factors
    @printf("delaunay_n%d max_levels=%d: levels=%d coarsest n=%d  cycles=%d acf(last5)=%.3f | sizes: %s\n", k, ml, length(h.levels), sizes[end], info.cycles, exp(mean(log.(cf[max(1,end-4):end]))), join(["$(t)$(s)" for (t,s) in zip(types,sizes)], " "))
end
