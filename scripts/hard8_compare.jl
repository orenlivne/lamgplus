# Re-run the 8 graphs LAMG+ failed on in the saved scaling corpus (anisotropy/thermal/
# Darcy), with the CURRENT caliber-2 LAMG+, plus DRA-QC and approxChol, same (L,b),
# solve to 1e-8. Does caliber-2 now converge them? How do DRA-QC and AC fare?
#   julia --project=examples/repro_env scripts/hard8_compare.jl
using LAMG, LinearAlgebra, SparseArrays, Random, Printf
import Laplacians
include(joinpath(@__DIR__, "work_per_edge.jl"))   # lamg_work, draqc_work, read_mm_adj, lcc (driver guarded)
const D = DRAQCHybrid.DRAQC

corpus = "/Users/oren/code/mg/maxflow/LAMG.jl/data"
adj(L) = (Lo = L - spdiagm(0=>diag(L)); W=-Lo; for r in 1:nnz(W); W.nzval[r]<0 && (W.nzval[r]=0.0); end; dropzeros!(W))

function ac_run(L, b, tol)
    f = Laplacians.approxchol_lap(adj(L); tol=tol, verbose=false)
    its = Int[0]; x = f(b; tol=tol, pcgIts=its, verbose=false)
    rel = norm(L*x - b)/norm(b)
    (it=its[1], rel=rel, ok = rel <= tol*10)
end

graphs = ["GHS_indef__dtoc","Schmid__thermal1","Botonakis__thermomech_TK","Botonakis__thermomech_TC",
          "Botonakis__thermomech_dM","GHS_indef__darcy003","GHS_indef__mario002","CEMW__tmt_sym"]
tol = 1e-8
@printf("%-26s %9s | %-14s | %-16s | %-14s\n","graph(failed in saved run)","n",
        "LAMG+ cal2","DRA-QC","approxChol")
println("-"^86)
for g in graphs
    L = lcc(read_mm_adj(joinpath(corpus, g*".mtx"))); n=size(L,1)
    Random.seed!(1); xt=randn(n); xt.-=sum(xt)/n; b=L*xt
    lw = lamg_work(L, b, tol)
    dw = draqc_work(L, b, tol; maxiter=400)
    ac = ac_run(L, b, tol)
    @printf("%-26s %9d | cyc=%-3d %-4s | it=%-4d %-4s | it=%-4d %-4s\n",
        g, n, lw.cyc, lw.ok ? "✓" : "✗(✗)", dw.it, dw.ok ? "✓" : "✗",
        ac.it, ac.ok ? "✓" : "✗")
    flush(stdout)
end
println("\n✓ = converged to 1e-8. LAMG+ here uses the CURRENT caliber-2 code (these 8 hit the 100-cycle cap in scaling_new.csv).")
