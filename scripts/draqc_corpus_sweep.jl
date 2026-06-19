# Run our DRA-QC reimplementation over the ENTIRE SuiteSparse corpus used for the
# LAMG+ scaling study, to test whether DRA-QC is empirically robust and order-m
# across the whole corpus (like LAMG+) or has outliers / non-convergent graphs.
#
# For each graph: load adjacency → graph Laplacian → largest connected component,
# solve L x = b (b = L x_true, zero-mean) with DRA-QC to 1e-8 (budget maxiter),
# and record convergence, iterations, and work-per-edge. Results stream to a CSV.
#
# Run (background):
#   julia --project=. scripts/draqc_corpus_sweep.jl [corpusdir] [out.csv] [maxnnz] [maxiter]
using LAMG, LinearAlgebra, SparseArrays, Random, Printf, Statistics
include(joinpath(@__DIR__, "work_per_edge.jl"))   # read_mm_adj, lcc, draqc_work (driver guarded)

corpus  = length(ARGS) >= 1 ? ARGS[1] : "/Users/oren/code/mg/maxflow/LAMG.jl/data"
outcsv  = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "results", "draqc_corpus.csv")
maxnnz  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10_000_000
maxiter = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 300
minm    = 1000
tol     = 1e-8
mkpath(dirname(outcsv))

files = String[]
for (root, _, fs) in walkdir(corpus), f in fs
    endswith(f, ".mtx") || continue
    p = joinpath(root, f)
    filesize(p) > 300_000_000 && continue          # skip giants (memory)
    push!(files, p)
end
sort!(files)
@printf("corpus: %d candidate .mtx under %s\n", length(files), corpus)

function run_sweep(files, outcsv, maxnnz, maxiter, minm, tol)
io = open(outcsv, "w")
println(io, "name,n,m,converged,iters,relres,wpe,setup_s,solve_s,status"); flush(io)

done = 0; conv = 0; skipped = 0; failed = 0
t0 = time()
for (idx, p) in enumerate(files)
    name = basename(p)
    local L
    try
        L = lcc(read_mm_adj(p))
    catch e
        skipped += 1; println(io, "$name,0,0,0,0,0,0,0,0,loaderr"); flush(io); continue
    end
    n = size(L, 1); m = (nnz(L) - n) ÷ 2
    if n < 50 || m < minm || (nnz(L) > maxnnz)
        skipped += 1; println(io, "$name,$n,$m,0,0,0,0,0,0,skip(size)"); flush(io); continue
    end
    Random.seed!(1); xt = randn(n); xt .-= sum(xt)/n; b = L * xt
    try
        DQ = DRAQCHybrid.DRAQC
        ts = @elapsed (h = DQ.draqc_setup(L); s = DQ.DRAQCSolver(h))
        tso = @elapsed ((x, info) = DQ.draqc_solve(s, b; tol = tol, maxiter = maxiter))
        rel = norm(L*x - b)/norm(b); ok = rel <= tol*10
        nl = DQ.num_levels(h)
        nnz_a = [nnz(h.A[ℓ]) for ℓ in 1:nl]
        transfer = [ ℓ <= length(h.P) ? 2*nnz(h.P[ℓ]) : 0 for ℓ in 1:nl ]
        v = kcycle_visits([size(h.A[ℓ],1) for ℓ in 1:nl])
        wpe_val = wpe(nnz_a, transfer, v, info.iters, m)
        done += 1; ok && (conv += 1)
        println(io, "$name,$n,$m,$(Int(ok)),$(info.iters),$rel,$wpe_val,$ts,$tso,ok"); flush(io)
    catch e
        failed += 1; println(io, "$name,$n,$m,0,0,0,0,0,0,solveerr"); flush(io)
    end
    if idx % 50 == 0
        @printf("[%d/%d] %.0fs  solved=%d converged=%d skipped=%d failed=%d\n",
                idx, length(files), time()-t0, done, conv, skipped, failed)
    end
end
close(io)
@printf("\nDONE: %d solved, %d converged (%.1f%%), %d skipped, %d errored. CSV: %s\n",
        done, conv, 100*conv/max(done,1), skipped, failed, outcsv)
end  # run_sweep

run_sweep(files, outcsv, maxnnz, maxiter, minm, tol)
