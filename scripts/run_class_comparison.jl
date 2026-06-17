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
const HAS_CMG   = true   # CMG runs via a cmg_env subprocess (cmg_bench.jl); avoids the MOI dep conflict
const HAS_PETSC = true   # PETSc GAMG runs via a competitor_env subprocess (petsc_bench.jl); GAMG setup
                         # can hard-crash on low-diameter/high-contrast graphs, so we isolate it like pyAMG/CMG.

const DATA = joinpath(@__DIR__, "..", "data")
const TOL = 1e-8
const MAXIT = 300
# Per-graph time budget: a solver gets BUDGET_MULT x (the fastest solver's time on that graph),
# floored/capped to [MIN_BUDGET, MAX_BUDGET] s. Exceeding it (or not converging) is flagged as
# "not converged within budget" (ok=0) and, for subprocess solvers, the process is killed at the cap.
const BUDGET_MULT = 50.0
const MIN_BUDGET  = 10.0
const MAX_BUDGET  = 90.0
cap = 5_000_000; per_class = 6; synthetic_only = false
for a in ARGS
    startswith(a,"--cap=") && (global cap = parse(Int, split(a,"=")[2]))
    startswith(a,"--per-class=") && (global per_class = parse(Int, split(a,"=")[2]))
    a=="--synthetic-only" && (global synthetic_only = true)   # skip real graphs; write *_synth.csv
end

adjW(L) = (Joff=L-spdiagm(0=>diag(L)); W=-Joff; for r in 1:nnz(W); W.nzval[r]<0&&(W.nzval[r]=0.0);end; dropzeros!(W))
relres(L,x,b) = norm(L*x .- b)/max(norm(b),1e-30)
function dirichlet(L,b); n=size(L,1); k=2:n; (L[k,k], b[k]); end
lift(y,n) = (x=vcat(0.0,y); x .-= mean(x); x)

# Each wrapper: (W,L,b) -> (total_seconds, ok). Built once, timed once (warm). DNF => ok=false.
function s_lamg(W,L,b,budget=Inf)
    o = LAMGOptions(tol=TOL, max_cycles=MAXIT)
    h = setup(L; options=o); solve(h,b;options=o)               # warm
    t = @elapsed (h2 = setup(L;options=o)); x,_ = solve(h2,b;options=o)
    ts = @elapsed solve(h2,b;options=o)
    (t+ts, relres(L,x,b) ≤ TOL*10)
end
function s_apx(W,L,b,budget=Inf)
    f0 = Lap.approxchol_lap(W; tol=TOL, verbose=false); f0(b)    # warm
    t = @elapsed f = Lap.approxchol_lap(W; tol=TOL, verbose=false)
    its=[0]; ts = @elapsed x = f(b; tol=TOL, maxits=MAXIT, verbose=false, pcgIts=its)
    (t+ts, relres(L,x,b) ≤ TOL*100)
end
function s_ac2(W,L,b,budget=Inf)
    f0 = Lap.approxchol_lap2(W; tol=TOL, verbose=false); f0(b)   # warm
    t = @elapsed f = Lap.approxchol_lap2(W; tol=TOL, verbose=false)
    its=[0]; ts = @elapsed x = f(b; tol=TOL, maxits=MAXIT, verbose=false, pcgIts=its)
    (t+ts, relres(L,x,b) ≤ TOL*100)
end
function s_hypre(W,L,b,budget=Inf)
    HAS_HYPRE || return (NaN,false)
    A,bc = dirichlet(L,b)
    amg=HYPRE.BoomerAMG(;Tol=TOL,MaxIter=MAXIT); HYPRE.solve!(amg,HYPRE.HYPREVector(zeros(length(bc))),HYPRE.HYPREMatrix(A),HYPRE.HYPREVector(bc)) # warm
    t=@elapsed begin AH=HYPRE.HYPREMatrix(A); bH=HYPRE.HYPREVector(bc); xH=HYPRE.HYPREVector(zeros(length(bc))); a2=HYPRE.BoomerAMG(;Tol=TOL,MaxIter=MAXIT) end
    ts=@elapsed HYPRE.solve!(a2,xH,AH,bH)
    y=copy(HYPRE.copy!(zeros(length(bc)),xH)); (t+ts, relres(L,lift(y,size(L,1)),b) ≤ TOL*100)
end
# Write a symmetric non-negative adjacency W to Matrix Market (lower triangle).
function write_mtx_sym(path, W)
    n=size(W,1); rows=rowvals(W); vals=nonzeros(W)
    cnt=0; for j in 1:n, k in nzrange(W,j); rows[k]>j && (cnt+=1); end
    open(path,"w") do io
        println(io,"%%MatrixMarket matrix coordinate real symmetric"); println(io,"$n $n $cnt")
        for j in 1:n, k in nzrange(W,j); i=rows[k]; i>j && println(io,"$i $j $(vals[k])"); end
    end
end
# pyAMG (smoothed aggregation) via a standalone Python subprocess: PyCall-free, so a pyAMG
# failure cannot crash this harness. Times are measured inside Python (exclude startup).
function s_pyamg(W,L,b,budget=Inf)
    tmp=tempname()*".mtx"; outf=tmp*".out"; write_mtx_sym(tmp,W)
    proc=run(pipeline(`python3 $(joinpath(@__DIR__,"pyamg_bench.py")) $tmp`; stdout=outf); wait=false)
    t0=time(); while process_running(proc) && time()-t0 < budget; sleep(0.5); end   # 150s hard cap -> DNF
    process_running(proc) && kill(proc)
    out = isfile(outf) ? read(outf,String) : ""
    rm(tmp;force=true); rm(outf;force=true)
    p=split(strip(out)); length(p)==3 || return (NaN,false)
    su=tryparse(Float64,p[1]); sv=tryparse(Float64,p[2])
    (su===nothing||sv===nothing||isnan(su)) ? (NaN,false) : (su+sv, p[3]=="1")
end
# CMG (Koutis Combinatorial Multigrid) via a cmg_env subprocess (CombinatorialMultigrid + Laplacians),
# isolated from competitor_env's JuMP/HiGHS (MathOptInterface) conflict. Times measured internally.
function s_cmg(W,L,b,budget=Inf)
    tmp=tempname()*".mtx"; outf=tmp*".out"; write_mtx_sym(tmp,W)
    proc=run(pipeline(`julia $(joinpath(@__DIR__,"cmg_bench.jl")) $tmp`; stdout=outf); wait=false)
    t0=time(); while process_running(proc) && time()-t0 < budget; sleep(0.5); end   # 200s cap -> DNF
    process_running(proc) && kill(proc)
    out=isfile(outf) ? read(outf,String) : ""; rm(tmp;force=true); rm(outf;force=true)
    p=split(strip(out)); length(p)==3 || return (NaN,false)
    su=tryparse(Float64,p[1]); sv=tryparse(Float64,p[2])
    (su===nothing||sv===nothing||isnan(su)) ? (NaN,false) : (su+sv, p[3]=="1")
end
# PETSc GAMG via a competitor_env subprocess (petsc_bench.jl): isolated because GAMG setup can
# hard-crash (C-level abort/OOM) on low-diameter or extreme-contrast graphs, which would take down
# the whole in-process harness. Times measured internally (exclude PETSc startup).
function s_petsc(W,L,b,budget=Inf)
    HAS_PETSC || return (NaN,false)
    tmp=tempname()*".mtx"; outf=tmp*".out"; write_mtx_sym(tmp,W)
    proc=run(pipeline(`julia $(joinpath(@__DIR__,"petsc_bench.jl")) $tmp`; stdout=outf); wait=false)
    t0=time(); while process_running(proc) && time()-t0 < budget; sleep(0.5); end
    process_running(proc) && kill(proc)
    out=isfile(outf) ? read(outf,String) : ""; rm(tmp;force=true); rm(outf;force=true)
    p=split(strip(out)); length(p)==3 || return (NaN,false)
    su=tryparse(Float64,p[1]); sv=tryparse(Float64,p[2])
    (su===nothing||sv===nothing||isnan(su)) ? (NaN,false) : (su+sv, p[3]=="1")
end
# The full GKS-style solver set, all RUN first-hand: LAMG+, approxChol (fast + robust AC),
# BoomerAMG (hypre), pyAMG (subprocess), CMG (cmg_env subprocess), PETSc GAMG (PETSc_jll).
const SOLVERS = vcat(Tuple{String,Function}[("LAMG+",s_lamg),("approxChol",s_apx),("AC",s_ac2),
                     ("BoomerAMG",s_hypre),("pyAMG",s_pyamg),("CMG",s_cmg)],
                     HAS_PETSC ? Tuple{String,Function}[("PETSc-GAMG",s_petsc)] : Tuple{String,Function}[])

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
# GKS synthetic families (Spielman's Laplacians.jl generators + standard reductions); each entry is
# (adjacency W, class tag, instance name). Covers all of the AC benchmark's family categories:
# random chimeras, weighted chimeras, SDDM chimeras, uniform & anisotropic Poisson grids, and
# adversarial weighted (Sachdeva) stars. The high-contrast grid category is the SPE10 family below.
function gks_families()
    out = Tuple{SparseMatrixCSC{Float64,Int},String,String}[]
    rng = MersenneTwister(20250617)
    for i in 1:per_class
        push!(out, (Lap.chimera(20000, i),     "chimera",      "chimera-$i"))
        push!(out, (Lap.wtedChimera(20000, i), "wtd-chimera",  "wtdchimera-$i"))
    end
    for i in 1:3                                       # SDDM chimera = chimera + grounded boundary node
        W = Lap.chimera(20000, i); n = size(W,1); Iw,Jw,Vw = findnz(W)
        nb = max(1, round(Int, sqrt(n))); idx = randperm(rng, n)[1:nb]; gw = 0.5 .+ rand(rng, nb)
        Wg = sparse(vcat(Iw, idx, fill(n+1,nb)), vcat(Jw, fill(n+1,nb), idx), vcat(Vw, gw, gw), n+1, n+1)
        push!(out, (Wg, "sddm-chimera", "sddmchimera-$i"))
    end
    for (a,b,c) in [(60,60,60),(80,80,40),(100,100,20)]            # uniform (isotropic) 3-D Poisson grids
        push!(out, (Lap.grid3(a,b,c), "grid-3D", "grid3-$a-$b-$c"))
    end
    for (N,ε) in [(450,1e-2),(450,1e-4),(640,1e-3)]               # grid-aligned ANISOTROPIC 2-D grids
        push!(out, (Lap.grid2(N,N; isotropy=ε), "aniso-grid", "aniso-$(N)x$(N)-eps$(ε)"))
    end
    for (k,nl) in enumerate([50_000,100_000,150_000])            # adversarial high-degree (Sachdeva) stars
        w = 10.0 .^ (2 .* rand(rng, nl) .- 1)                     # log-uniform weights, contrast 1e2
        S = sparse(vcat(fill(1,nl), 2:nl+1), vcat(2:nl+1, fill(1,nl)), vcat(w,w), nl+1, nl+1)
        push!(out, (S, "star", "star-$nl"))                      # the hub degree is the adversarial feature
    end
    out
end

out = joinpath(@__DIR__,"..","results",
               synthetic_only ? "class_comparison_synth.csv" : "class_comparison.csv")
io = open(out,"w"); println(io,"instance,class,n,m,solver,total_s,per_nnz_us,ok"); flush(io)
emit(io,name,cls,n,m,sv,t,ok) = (nz=n+2m; @printf(io,"%s,%s,%d,%d,%s,%.5f,%.4f,%d\n",name,cls,n,m,sv,t,isnan(t) ? NaN : t*1e6/nz,ok ? 1 : 0); flush(io))

println("solvers available: ", join([s for (s,_) in SOLVERS if true], ", "),
        "  | pyAMG=$HAS_PYAMG HYPRE=$HAS_HYPRE CMG=$HAS_CMG")

function run_one(name,cls,W,L)
    n=size(L,1); nz=nnz(L); m=div(nz-n,2)
    rng=MersenneTwister(0xff+n); xt=randn(rng,n); xt.-=sum(xt)/n; b=L*xt
    res = Dict{String,Tuple{Float64,Bool}}()
    # Phase 1: the two fastest solvers (LAMG+, approxChol) set the per-graph time budget.
    for (sv,fn) in SOLVERS[1:2]
        res[sv] = try; fn(W,L,b,Inf); catch e; (@warn "$sv $name" e; (NaN,false)); end
    end
    tfast = minimum(Float64[t for (t,ok) in values(res) if ok && isfinite(t)]; init=Inf)
    budget = isfinite(tfast) ? clamp(BUDGET_MULT*tfast, MIN_BUDGET, MAX_BUDGET) : MAX_BUDGET
    # Phase 2: every other solver runs under that budget; over-budget (or non-convergent) => ok=false.
    for (sv,fn) in SOLVERS[3:end]
        t,ok = try; fn(W,L,b,budget); catch e; (@warn "$sv $name" e; (NaN,false)); end
        (isfinite(t) && t > budget) && (ok = false)
        res[sv] = (t,ok)
    end
    for (sv,_) in SOLVERS
        t,ok = res[sv]
        emit(io,name,cls,n,m,sv,t,ok)
        why = ok ? "" : (isnan(t) ? "(DNF)" : (t > budget ? "(>budget)" : "(no conv)"))
        @printf("  %-22s %-12s %9s %s\n", name, sv, isnan(t) ? "DNF" : @sprintf("%.3fs",t), why); flush(stdout)
    end
    @printf("    [budget %.1fs = %.0fx fastest %.3fs]\n", budget, BUDGET_MULT, tfast); flush(stdout)
end

if !synthetic_only
    println("\n== real graphs ==");
    for (f,cls) in pick_real()
        W,L = try; reduce_to_lcc(read_mm_adj(joinpath(DATA,f))...); catch e; (@warn "load $f" e; continue); end
        nnz(L) ≤ cap || continue
        run_one(f,cls,W,L)
    end
end
println("\n== GKS synthetic families ==")
for (W,cls,name) in gks_families()
    L = LAMG.laplacian(W); W2,L2 = reduce_to_lcc(W,L)
    run_one(name,cls,W2,L2)
end
# SPE10 (Tenth SPE Comparative Solution Project, Christie-Blunt 2001) Model 2: the canonical "SPE"
# family of GKS 2023. TPFA harmonic-mean-transmissibility graph Laplacians (high-contrast, anisotropic)
# built by scripts/build_spe10.jl from the OPM perm field; nz=20/43/85 is a 0.26M->1.12M size ladder.
println("\n== SPE10 reservoir (Christie-Blunt; TPFA high-contrast) ==")
for nz in (20, 43, 85)
    f = joinpath(DATA, "SPE__spe10_2_nz$(nz).mtx")
    isfile(f) || (@warn "missing $f (run scripts/build_spe10.jl)"; continue)
    W,L = try; reduce_to_lcc(read_mm_adj(f)...); catch e; (@warn "load $f" e; continue); end
    run_one("spe10_2_nz$(nz)", "SPE/reservoir", W, L)
end
close(io); println("\nDONE -> $out")
