# true_acf.jl — is the paper's "ACF" (geomean of last 5 factors of a 5-cycle solve) actually asymptotic?
# Run 40 cycles to tol 1e-15 under LX and RAND RHS and report per-cycle factors + escalation flag.
using LAMG, Laplacians, SparseArrays, LinearAlgebra, Random, Statistics, Printf
include("/Users/oren/code/aier/lamgplus/scripts/mm_loader.jl")
const DATA = "/Users/oren/code/aier/data/suitesparse"
for f in ("SNAP__web-Stanford.mtx", "DIMACS10__delaunay_n18.mtx", "SNAP__com-DBLP.mtx", "GHS_psdef__bmwcra_1.mtx")
    W,L = reduce_to_lcc(read_mm_adj(joinpath(DATA,f))...); n=size(L,1)
    xt = randn(MersenneTwister(11), n); xt .-= mean(xt); bLX = L*xt
    bR = randn(MersenneTwister(12), n); bR .-= mean(bR)
    o = LAMGOptions(tol=1e-15, max_cycles=40)
    h = setup(L; options=o)
    for (kind,b) in (("LX",bLX),("RAND",bR))
        x, info = solve(h, b; options=o)
        cf = info.conv_factors
        @printf("%-28s %-4s cycles=%d escalated=%s  factors: %s\n", f, kind, info.cycles, info.gamma_escalated, join([@sprintf("%.2f",c) for c in cf], " "))
    end
end
