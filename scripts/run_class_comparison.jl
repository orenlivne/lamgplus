# GKS-STYLE multi-solver comparison BY GRAPH CLASS (Gao-Kyng-Spielman 2023 layout).
# Runs every practical Laplacian solver on a class-stratified instance set and writes a
# long-format CSV (one row per instance x solver); class_comparison_table.py aggregates to
# per-class mean / worst-case time-per-nonzero + DNF counts -> one big table.
#
# Solvers: LAMG+ ; approxChol (approxchol_lap) ; AC (approxchol_lap2, robust) ;
#          BoomerAMG (HYPRE) ; pyAMG (smoothed aggregation) ; CMG (CombinatorialMultigrid).
# Classes: real SuiteSparse/SNAP (FE/structural, mesh/grid, social, citation, web, road)
#          + GKS synthetic families (chimera, weighted chimera, anisotropic grid, stars).
# Safeguards (so the autonomous chain never hangs): hard nnz cap, MaxIter caps on every
# solver (non-convergence => DNF, not a hang), per-solver try/catch.
#   julia -t1 --project=scripts/competitor_env scripts/run_class_comparison.jl [--cap=5000000] [--per-class=6]
import Pkg
Pkg.activate(joinpath(@__DIR__, "competitor_env"))
using LAMG, LinearAlgebra, SparseArrays, Random, Statistics, Printf
import Laplacians
const Lap = Laplacians
include(joinpath(@__DIR__, "mm_loader.jl"))     # read_mm_adj, reduce_to_lcc
const HAS_PYAMG = try; @eval using PyCall; @eval const PYAMG = pyimport("pyamg"); @eval const SPSP = pyimport("scipy.sparse"); true; catch; false; end
const HAS_HYPRE = try; @eval using HYPRE; HYPRE.Init(); true; catch; false; end
const HAS_CMG   = try; @eval import CombinatorialMultigrid; true; catch; false; end

const DATA = joinpath(@__DIR__, "..", "data")
const TOL = 1e-8
const MAXIT = 300
cap = 5_000_000; per_class = 6
for a in ARGS
    startswith(a,"--cap=") && (global cap = parse(Int, split(a,"=")[2]))
    startswith(a,"--per-class=") && (global per_class = parse(Int, split(a,"=")[2]))
end

adjW(L) = (Joff=L-spdiagm(0=>diag(L)); W=-Joff; for r in 1:nnz(W); W.nzval[r]<0&&(W.nzval[r]=0.0);end; dropzeros!(W))
relres(L,x,b) = norm(L*x .- b)/max(norm(b),1e-30)
function dirichlet(L,b); n=size(L,1); k=2:n; (L[k,k], b[k]); end
lift(y,n) = (x=vcat(0.0,y); x .-= mean(x); x)

# Each wrapper: (W,L,b) -> (total_seconds, ok). Built once, timed once (warm). DNF => ok=false.
function s_lamg(W,L,b)
    o = LAMGOptions(tol=TOL, max_cycles=MAXIT)
    h = setup(L; options=o); solve(h,b;options=o)               # warm
    t = @elapsed (h2 = setup(L;options=o)); x,_ = solve(h2,b;options=o)
    ts = @elapsed solve(h2,b;options=o)
    (t+ts, relres(L,x,b) ≤ TOL*10)
end
function s_apx(W,L,b)
    f0 = Lap.approxchol_lap(W; tol=TOL, verbose=false); f0(b)    # warm
    t = @elapsed f = Lap.approxchol_lap(W; tol=TOL, verbose=false)
    its=[0]; ts = @elapsed x = f(b; tol=TOL, maxits=MAXIT, verbose=false, pcgIts=its)
    (t+ts, relres(L,x,b) ≤ TOL*100)
end
function s_ac2(W,L,b)
    f0 = Lap.approxchol_lap2(W; tol=TOL, verbose=false); f0(b)   # warm
    t = @elapsed f = Lap.approxchol_lap2(W; tol=TOL, verbose=false)
    its=[0]; ts = @elapsed x = f(b; tol=TOL, maxits=MAXIT, verbose=false, pcgIts=its)
    (t+ts, relres(L,x,b) ≤ TOL*100)
end
function s_hypre(W,L,b)
    HAS_HYPRE || return (NaN,false)
    A,bc = dirichlet(L,b)
    amg=HYPRE.BoomerAMG(;Tol=TOL,MaxIter=MAXIT); HYPRE.solve!(amg,HYPRE.HYPREVector(zeros(length(bc))),HYPRE.HYPREMatrix(A),HYPRE.HYPREVector(bc)) # warm
    t=@elapsed begin AH=HYPRE.HYPREMatrix(A); bH=HYPRE.HYPREVector(bc); xH=HYPRE.HYPREVector(zeros(length(bc))); a2=HYPRE.BoomerAMG(;Tol=TOL,MaxIter=MAXIT) end
    ts=@elapsed HYPRE.solve!(a2,xH,AH,bH)
    y=copy(HYPRE.copy!(zeros(length(bc)),xH)); (t+ts, relres(L,lift(y,size(L,1)),b) ≤ TOL*100)
end
function s_pyamg(W,L,b)
    HAS_PYAMG || return (NaN,false)
    A,bc = dirichlet(L,b); Acsr = SPSP.csr_matrix(PyCall.PyObject(A)); bp=PyCall.PyObject(bc)
    ml0=PYAMG.smoothed_aggregation_solver(Acsr); ml0.solve(bp;tol=TOL)               # warm
    t=@elapsed ml=PYAMG.smoothed_aggregation_solver(Acsr)
    res=PyCall.PyObject([]); ts=@elapsed x=ml.solve(bp;tol=TOL,maxiter=MAXIT,residuals=res)
    y=convert(Vector{Float64},x); (t+ts, relres(L,lift(y,size(L,1)),b) ≤ TOL*100)
end
function s_cmg(W,L,b)
    HAS_CMG || return (NaN,false)
    # CombinatorialMultigrid.jl: cmg_preconditioner_lap(L) -> (pfunc, h); solve via PCG.
    pfunc,_ = CombinatorialMultigrid.cmg_preconditioner_lap(L)
    x = Lap.pcg(L, b, pfunc; tol=TOL, maxits=MAXIT)                                   # warm+solve
    t = @elapsed (pf2,_) = CombinatorialMultigrid.cmg_preconditioner_lap(L)
    ts = @elapsed x2 = Lap.pcg(L, b, pf2; tol=TOL, maxits=MAXIT)
    (t+ts, relres(L,x2,b) ≤ TOL*100)
end
# Four robust NATIVE-Julia solvers. pyAMG (via PyCall) hard-crashes the process on some graphs
# and CMG (CombinatorialMultigrid) has no maintained Julia package; we cite GKS-2023 for both
# instead of re-running them. (CMG/pyAMG auto-included only if a working install is detected.)
const SOLVERS = vcat(Tuple{String,Function}[("LAMG+",s_lamg),("approxChol",s_apx),("AC",s_ac2),("BoomerAMG",s_hypre)],
                     HAS_CMG ? Tuple{String,Function}[("CMG",s_cmg)] : Tuple{String,Function}[])

# ---- class-stratified instance set ----
function app_category(name)
    n=lowercase(name)
    occursin(r"bmwcra|crankseg|troll|pwtk|bone|ldoor|hood|fault|af_shell|nasa|engine|gearbox|shipsec|thread|x104|s3dk|ct20|pkustk|bcsstk|nd6k|nd12k|nd24k|inline|fcondp|halfb|af_0|af_1|af_2|af_3|af_4|af_5",n) && return "FE/structural"
    occursin(r"web-|cnr|eu-2005|in-2004|uk-|webbase",n) && return "web"
    occursin(r"soc|epinion|slashdot|email|amazon|com-|cit-|coauth|flickr|hollywood|dblp|citeseer|p2p|gowalla|brightkite|wiki",n) && return "social/citation"
    occursin(r"road|osm",n) && return "road"
    occursin(r"delaunay|grid|apache|thermal|ecology|parabolic|g3_circuit|tmt|atmos|rgg|333sp|adaptive|channel|hugetric|hugebubble|venturi|wave|debr|net|cage|mesh|AS365|NACA|M6|fe_|denormal",n) && return "mesh/grid"
    return "other"
end
function pick_real()
    cands = Dict{String,Vector{Tuple{String,Int}}}()
    for f in readdir(DATA)
        endswith(f,".mtx") || continue
        cat = app_category(f); cat=="other" && continue
        z = try; open(joinpath(DATA,f)) do io; readline(io); l=readline(io); while startswith(strip(l),"%"); l=readline(io); end; parse(Int,split(l)[3]); end; catch; 0; end
        (z>50_000 && z<cap) || continue
        push!(get!(cands,cat,[]), (f,z))
    end
    sel = Tuple{String,String}[]   # (file, class)
    for (cat,v) in cands
        sort!(v, by=x->x[2]); k = min(per_class, length(v))
        idx = k == 1 ? [cld(length(v),2)] : round.(Int, range(1, length(v); length=k))
        for i in unique(idx); push!(sel, (v[i][1], cat)); end
    end
    sel
end
# GKS synthetic families (Laplacians.jl generators); adjacency W, class tag.
function gks_families()
    out = Tuple{SparseMatrixCSC{Float64,Int},String,String}[]
    for i in 1:per_class
        push!(out, (Lap.chimera(20000, i),               "chimera",        "chimera-$i"))
        push!(out, (Lap.wtedChimera(20000, i),           "wtd-chimera",    "wtdchimera-$i"))
    end
    for (k,(a,b,c)) in enumerate([(60,60,60),(80,80,40),(100,100,20)])  # anisotropic 3D grids
        g = Lap.grid3(a,b,c); out=out; push!(out, (g, "aniso-grid", "grid3-$a-$b-$c"))
    end
    out
end

out = joinpath(@__DIR__,"..","results","class_comparison.csv")
mkpath(dirname(out)); io = open(out,"w"); println(io,"instance,class,n,m,solver,total_s,per_nnz_us,ok"); flush(io)
emit(io,name,cls,n,m,sv,t,ok) = (nz=n+2m; @printf(io,"%s,%s,%d,%d,%s,%.5f,%.4f,%d\n",name,cls,n,m,sv,t,isnan(t) ? NaN : t*1e6/nz,ok ? 1 : 0); flush(io))

println("solvers available: ", join([s for (s,_) in SOLVERS if true], ", "),
        "  | pyAMG=$HAS_PYAMG HYPRE=$HAS_HYPRE CMG=$HAS_CMG")

function run_one(name,cls,W,L)
    n=size(L,1); nz=nnz(L); m=div(nz-n,2)
    rng=MersenneTwister(0xff+n); xt=randn(rng,n); xt.-=sum(xt)/n; b=L*xt
    for (sv,fn) in SOLVERS
        t,ok = try; fn(W,L,b); catch e; (@warn "$sv $name" e; (NaN,false)); end
        emit(io,name,cls,n,m,sv,t,ok)
        @printf("  %-22s %-12s %8s %s\n", name, sv, isnan(t) ? "DNF" : @sprintf("%.3fs",t), ok ? "" : "(no conv)"); flush(stdout)
    end
end

println("\n== real graphs ==");
for (f,cls) in pick_real()
    W,L = try; reduce_to_lcc(read_mm_adj(joinpath(DATA,f))...); catch e; (@warn "load $f" e; continue); end
    nnz(L) ≤ cap || continue
    run_one(f,cls,W,L)
end
println("\n== GKS synthetic families ==")
for (W,cls,name) in gks_families()
    L = LAMG.laplacian(W); W2,L2 = reduce_to_lcc(W,L)
    run_one(name,cls,W2,L2)
end
close(io); println("\nDONE -> $out")
