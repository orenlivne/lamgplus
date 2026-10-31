# compete_rhs.jl — re-run the paper's main head-to-head (Table 4.2 bottom block: 201 graphs with m > 1e6)
# under BOTH right-hand sides: LX (b = L x_true, paper + GKS protocol) and RAND (b ~ N(0,1) zero-mean, referee's).
# Solvers: LAMG+, approxChol (approxchol_lap), AC (approxchol_lap2). Serial, warm, tol 1e-8. Incremental CSV.
using LAMG, Laplacians, SparseArrays, LinearAlgebra, Random, Statistics, Printf
include("/Users/oren/code/aier/lamgplus/scripts/mm_loader.jl")
const TOL = 1e-8; const MAXIT = 1000
const DATA = "/Users/oren/code/aier/data/suitesparse"
out = get(ENV, "OUT", "compete_rhs.csv"); list = readlines(get(ENV, "LIST", "large201.txt"))
maxnnz = parse(Int, get(ENV, "MAXNNZ", "60000000"))
acf(cf) = isempty(cf) ? NaN : exp(mean(log.(cf[max(1,end-4):end])))
relres(L,x,b) = norm(L*x .- b)/norm(b)
done = Set{String}()
if isfile(out); for l in readlines(out)[2:end]; push!(done, split(l, ",")[1]); end; end
io = open(out, "a")
isempty(done) && println(io, "instance,n,m,nnz,rhs,lp_setup_s,lp_solve_s,lp_cycles,lp_acf,lp_escal,lp_relres,apx_setup_s,apx_solve_s,apx_its,apx_relres,ac_setup_s,ac_solve_s,ac_its,ac_relres")
opts = LAMGOptions(tol=TOL, max_cycles=300)
for f in list
    f in done && continue
    W,L = try read_mm_adj(joinpath(DATA,f)) catch e; println("skip $f: $e"); continue; end
    W,L = reduce_to_lcc(W,L); n=size(L,1); m=div(nnz(L)-n,2)
    nnz(L) > maxnnz && (println("skip $f nnz=$(nnz(L))"); continue)
    rng = MersenneTwister(11); xt = randn(rng, n); xt .-= mean(xt); bLX = L*xt
    bR = randn(MersenneTwister(12), n); bR .-= mean(bR)
    # warm-up (JIT) on this instance, untimed
    h0 = setup(L; options=opts); solve(h0, bLX; options=opts); h0 = nothing
    f0 = approxchol_lap(W; tol=TOL, verbose=false); f0(bLX); f0 = nothing
    g0 = approxchol_lap2(W; tol=TOL, verbose=false); g0(bLX); g0 = nothing
    tls = @elapsed h = setup(L; options=opts)
    tas = @elapsed fa = approxchol_lap(W; tol=TOL, verbose=false)
    tcs = @elapsed fc = approxchol_lap2(W; tol=TOL, verbose=false)
    for (kind, b) in (("LX", bLX), ("RAND", bR))
        x, info = solve(h, b; options=opts)
        ia=[0]; ta = @elapsed xa = fa(b; tol=TOL, maxits=MAXIT, verbose=false, pcgIts=ia)
        ic=[0]; tc = @elapsed xc = fc(b; tol=TOL, maxits=MAXIT, verbose=false, pcgIts=ic)
        @printf(io, "%s,%d,%d,%d,%s,%.4f,%.4f,%d,%.4f,%d,%.2e,%.4f,%.4f,%d,%.2e,%.4f,%.4f,%d,%.2e\n", f, n, m, nnz(L), kind,
                tls, info.solve_time, info.cycles, acf(info.conv_factors), info.gamma_escalated ? 1 : 0, relres(L,x,b),
                tas, ta, ia[1], relres(L,xa,b), tcs, tc, ic[1], relres(L,xc,b)); flush(io)
        @printf("%-36s %-4s LAMG+ %.2f+%.2fs cyc=%3d acf=%.3f | apx %.2f+%.2fs its=%3d | AC %.2f+%.2fs its=%3d\n", f, kind,
                tls, info.solve_time, info.cycles, acf(info.conv_factors), tas, ta, ia[1], tcs, tc, ic[1])
    end
    h=nothing; fa=nothing; fc=nothing; W=nothing; L=nothing; GC.gc()
end
close(io)
