# robust_verify.jl — before/after the test-vector escalation on the 202 large graphs, both RHS.
# "old" = escalate_tv=false (legacy restart escalation), "new" = default. ACF = homogeneous-problem
# energy-norm factor of the FINAL hierarchy (20 cycles, tail 8). Resumable CSV.
using LAMG, SparseArrays, LinearAlgebra, Random, Statistics, Printf
include("/Users/oren/code/aier/lamgplus/scripts/mm_loader.jl"); include(joinpath(@__DIR__, "acfhelpers.jl"))
const DATA = "/Users/oren/code/aier/data/suitesparse"
out = ENV["OUT"]; list = readlines(ENV["LIST"]); maxnnz = parse(Int, get(ENV,"MAXNNZ","60000000"))
done = Set{String}(); isfile(out) && for l in readlines(out)[2:end]; push!(done, split(l,",")[1]); end
io = open(out, "a")
isempty(done) && println(io, "instance,n,m,rhs,old_cycles,old_solve_s,old_esc,old_acf,new_setup_s,new_cycles,new_solve_s,new_esc,new_acf")
o_new = LAMGOptions(tol=1e-8, max_cycles=300); o_old = LAMGOptions(tol=1e-8, max_cycles=300, escalate_tv=false)
# JIT warm-up
let Lw = grid2d_laplacian(60,60); bw = Lw*(randn(MersenneTwister(1),3600) |> v->v.-mean(v))
    for o in (o_new,o_old); h=setup(Lw;options=o); solve(h,bw;options=o); acf_homog(h,Lw,o;ncyc=3,tail=2); end
    solve(Lw,bw;options=LAMGOptions(tol=1e-8,escalate_tv_threshold=0.0))
end
for f in list
    f in done && continue
    W,L = try read_mm_adj(joinpath(DATA,f)) catch e; println("skip $f: $e"); continue; end
    W,L = reduce_to_lcc(W,L); n=size(L,1); m=div(nnz(L)-n,2)
    nnz(L) > maxnnz && (println("skip $f nnz=$(nnz(L))"); continue)
    xt = randn(MersenneTwister(11), n); xt .-= mean(xt); bLX = L*xt
    bR = randn(MersenneTwister(12), n); bR .-= mean(bR)
    ts = @elapsed h = setup(L; options=o_new)
    a_old = acf_homog(h, L, o_old; ncyc=20, tail=8).acf
    for (kind, b) in (("LX", bLX), ("RAND", bR))
        xo, io_ = solve(h, b; options=o_old)
        xn, in_ = solve(h, b; options=o_new)
        a_new = in_.rebuilt_hierarchy === nothing ? a_old : acf_homog(in_.rebuilt_hierarchy, L, o_new; ncyc=20, tail=8).acf
        @printf(io, "%s,%d,%d,%s,%d,%.4f,%d,%.4f,%.4f,%d,%.4f,%d,%.4f\n", f, n, m, kind, io_.cycles, io_.solve_time, io_.gamma_escalated ? 1 : 0, a_old,
                ts, in_.cycles, in_.solve_time, in_.tv_escalated ? 1 : 0, a_new); flush(io)
        @printf("%-36s %-4s old cyc=%3d %.2fs acf=%.3f | new cyc=%3d %.2fs esc=%d acf=%.3f\n", f, kind, io_.cycles, io_.solve_time, a_old, in_.cycles, in_.solve_time, in_.tv_escalated, a_new)
    end
    h=nothing; W=nothing; L=nothing; GC.gc()
end
close(io)
