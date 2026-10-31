# large_compete.jl — Referee-1 point 5: is LAMG+ ever the fastest? Table 4.3's FE/structural instances were
# small (n ≤ 8e4). Re-run the multi-solver comparison on the LARGE FE/structural + mesh instances of Table 4.2,
# with BoomerAMG and pyAMG included. Serial, warm timed run, time per nonzero (µs/nnz) to 1e-8.
using LAMG, Laplacians, SparseArrays, LinearAlgebra, Random, Statistics, Printf
using HYPRE; HYPRE.Init()
include("/Users/oren/code/aier/lamgplus/scripts/mm_loader.jl")
const TOL = 1e-8; const MAXIT = 300
const DATA = "/Users/oren/code/aier/data/suitesparse"
const PYBENCH = "/Users/oren/code/aier/lamgplus/scripts/pyamg_bench.py"
out = get(ENV, "OUT", "large_compete.csv"); budget = parse(Float64, get(ENV, "BUDGET", "400"))
relres(L,x,b) = norm(L*x .- b)/max(norm(b),1e-30)
function dirichlet(L,b); n=size(L,1); k=2:n; (L[k,k], b[k]); end
lift(y,n) = (x=vcat(0.0,y); x .-= mean(x); x)
function s_lamg(W,L,b)
    o = LAMGOptions(tol=TOL, max_cycles=MAXIT)
    h = setup(L; options=o); solve(h,b;options=o)
    t = @elapsed (h2 = setup(L;options=o)); x,info = solve(h2,b;options=o)
    (t+info.solve_time, relres(L,x,b) ≤ TOL*10, info.cycles)
end
function s_apx(W,L,b)
    f0 = approxchol_lap(W; tol=TOL, verbose=false); f0(b)
    t = @elapsed f = approxchol_lap(W; tol=TOL, verbose=false)
    its=[0]; ts = @elapsed x = f(b; tol=TOL, maxits=MAXIT, verbose=false, pcgIts=its)
    (t+ts, relres(L,x,b) ≤ TOL*100, its[1])
end
function s_ac2(W,L,b)
    f0 = approxchol_lap2(W; tol=TOL, verbose=false); f0(b)
    t = @elapsed f = approxchol_lap2(W; tol=TOL, verbose=false)
    its=[0]; ts = @elapsed x = f(b; tol=TOL, maxits=MAXIT, verbose=false, pcgIts=its)
    (t+ts, relres(L,x,b) ≤ TOL*100, its[1])
end
function s_hypre(W,L,b)
    A,bc = dirichlet(L,b)
    amg=HYPRE.BoomerAMG(;Tol=TOL,MaxIter=MAXIT); HYPRE.solve!(amg,HYPRE.HYPREVector(zeros(length(bc))),HYPRE.HYPREMatrix(A),HYPRE.HYPREVector(bc))
    t=@elapsed begin AH=HYPRE.HYPREMatrix(A); bH=HYPRE.HYPREVector(bc); xH=HYPRE.HYPREVector(zeros(length(bc))); a2=HYPRE.BoomerAMG(;Tol=TOL,MaxIter=MAXIT) end
    ts=@elapsed HYPRE.solve!(a2,xH,AH,bH)
    y=copy(HYPRE.copy!(zeros(length(bc)),xH)); (t+ts, relres(L,lift(y,size(L,1)),b) ≤ TOL*100, -1)
end
function write_mtx_sym(path, W)
    n=size(W,1); rows=rowvals(W); vals=nonzeros(W)
    cnt=0; for j in 1:n, k in nzrange(W,j); rows[k]>j && (cnt+=1); end
    open(path,"w") do io
        println(io,"%%MatrixMarket matrix coordinate real symmetric"); println(io,"$n $n $cnt")
        for j in 1:n, k in nzrange(W,j); i=rows[k]; i>j && println(io,"$i $j $(vals[k])"); end
    end
end
function s_pyamg(W,L,b)
    tmp=tempname()*".mtx"; outf=tmp*".out"; write_mtx_sym(tmp,W)
    proc=run(pipeline(`python3 $PYBENCH $tmp`; stdout=outf); wait=false)
    t0=time(); while process_running(proc) && time()-t0 < budget; sleep(0.5); end
    process_running(proc) && kill(proc)
    o = isfile(outf) ? read(outf,String) : ""; rm(tmp;force=true); rm(outf;force=true)
    p=split(strip(o)); length(p)==3 || return (NaN,false,-1)
    su=tryparse(Float64,p[1]); sv=tryparse(Float64,p[2])
    (su===nothing||sv===nothing||isnan(su)) ? (NaN,false,-1) : (su+sv, p[3]=="1", -1)
end
SOLVERS = [("LAMG+",s_lamg),("approxChol",s_apx),("AC",s_ac2),("BoomerAMG",s_hypre),("pyAMG",s_pyamg)]
insts = [("FE/structural","GHS_psdef__bmwcra_1.mtx"),("FE/structural","GHS_psdef__crankseg_1.mtx"),("FE/structural","DNVS__troll.mtx"),
         ("FE/structural","Boeing__pwtk.mtx"),("FE/structural","Schenk_AFE__af_shell8.mtx"),("FE/structural","GHS_psdef__inline_1.mtx"),
         ("FE/structural","Janna__Fault_639.mtx"),("FE/structural","Oberwolfach__bone010.mtx"),
         ("mesh/grid","GHS_psdef__apache2.mtx"),("mesh/grid","AMD__G3_circuit.mtx"),("mesh/grid","Schmid__thermal2.mtx"),
         ("mesh/grid","Wissgott__parabolic_fem.mtx"),("mesh/grid","McRae__ecology1.mtx"),("mesh/grid","DIMACS10__delaunay_n20.mtx"),
         ("road","SNAP__roadNet-CA.mtx"),("social","SNAP__com-DBLP.mtx"),("web","SNAP__web-Google.mtx")]
io = open(out,"w"); println(io,"instance,class,n,m,nnz,solver,total_s,per_nnz_us,ok,its")
for (cls,f) in insts
    W,L = try read_mm_adj(joinpath(DATA,f)) catch e; println("skip $f: $e"); continue; end
    W,L = reduce_to_lcc(W,L); n=size(L,1); m=div(nnz(L)-n,2)
    rng=MersenneTwister(1); x=randn(rng,n); x.-=mean(x); b=L*x
    for (nm,s) in SOLVERS
        t,ok,its = try s(W,L,b) catch e; println("  $nm errored: $(sprint(showerror,e)[1:min(end,120)])"); (NaN,false,-1) end
        @printf(io,"%s,%s,%d,%d,%d,%s,%.4f,%.4f,%d,%d\n", f, cls, n, m, nnz(L), nm, t, 1e6*t/nnz(L), ok ? 1 : 0, its); flush(io)
        @printf("%-32s %-14s %-10s %8.3fs %6.3f us/nnz %s its=%d\n", f, cls, nm, t, 1e6*t/nnz(L), ok ? "ok" : "DNF", its)
    end
    W=nothing; L=nothing; GC.gc()
end
close(io)
