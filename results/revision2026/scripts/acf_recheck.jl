# Re-measure homogeneous ACF (30 cycles, tail 10, increase-detected floor) of the default-K hierarchy
# and of the escalated hierarchy (forced rebuild, K=8) for the slow/triggered graphs of robust_verify2.
using LAMG, SparseArrays, LinearAlgebra, Random, Statistics, Printf
include("/Users/oren/code/aier/lamgplus/scripts/mm_loader.jl"); include(joinpath(@__DIR__, "acfhelpers.jl"))
const DATA = "/Users/oren/code/aier/data/suitesparse"
io = open(ENV["OUT"], "w"); println(io, "instance,n,acf_default,valid_default,acf_rebuilt,valid_rebuilt,rand_cycles_default,rand_cycles_new,esc,rev")
o = LAMGOptions(tol=1e-8, max_cycles=300); oo = LAMGOptions(tol=1e-8, max_cycles=300, escalate_tv=false)
of = LAMGOptions(tol=1e-8, max_cycles=300, escalate_tv_threshold=0.0, escalate_tv_guard=false)
for f in readlines(ENV["LIST"])
    W,L = reduce_to_lcc(read_mm_adj(joinpath(DATA,f))...); n=size(L,1)
    bR = randn(MersenneTwister(12), n); bR .-= mean(bR)
    h = setup(L; options=o)
    a0 = acf_homog(h, L, o)
    _, i_old = solve(h, bR; options=oo); _, i_new = solve(h, bR; options=o)
    _, i_f = solve(h, bR; options=of); hb = i_f.rebuilt_hierarchy
    a1 = hb === nothing ? a0 : acf_homog(hb, L, o)
    @printf(io, "%s,%d,%.4f,%d,%.4f,%d,%d,%d,%d,%d\n", f, n, a0.acf, a0.valid, a1.acf, a1.valid, i_old.cycles, i_new.cycles, i_new.tv_escalated, i_new.tv_reverted); flush(io)
    @printf("%-34s default %.3f (%d valid) | rebuilt %.3f (%d valid) | RAND cyc %d -> %d esc=%d rev=%d\n", f, a0.acf, a0.valid, a1.acf, a1.valid, i_old.cycles, i_new.cycles, i_new.tv_escalated, i_new.tv_reverted)
    h=nothing; hb=nothing; GC.gc()
end
