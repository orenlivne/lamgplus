# Current caliber-2 LAMG+ over the ENTIRE corpus (refreshes the stale scaling_new.csv;
# the 8 anisotropy/thermal/Darcy fails there predate caliber-2). Same conditions as the
# AC sweep: build hierarchy once, solve L x = b to 1e-8.
#   julia --project=examples/repro_env scripts/lamg_corpus_sweep.jl [corpusdir] [out.csv] [maxnnz]
using LAMG, LinearAlgebra, SparseArrays, Random, Printf, Statistics
include(joinpath(@__DIR__, "work_per_edge.jl"))   # lamg_work, read_mm_adj, lcc (driver guarded)
corpus = length(ARGS)>=1 ? ARGS[1] : "/Users/oren/code/mg/maxflow/LAMG.jl/data"
outcsv = length(ARGS)>=2 ? ARGS[2] : joinpath(@__DIR__,"..","results","lamg_corpus.csv")
maxnnz = length(ARGS)>=3 ? parse(Int,ARGS[3]) : 10_000_000
minm=1000; tol=1e-8; mkpath(dirname(outcsv))
function run_sweep(corpus,outcsv,maxnnz,minm,tol)
    files=String[]; for (root,_,fs) in walkdir(corpus), f in fs
        endswith(f,".mtx") || continue; p=joinpath(root,f); filesize(p)>300_000_000 && continue; push!(files,p); end
    sort!(files); @printf("corpus: %d candidate .mtx\n", length(files))
    io=open(outcsv,"w"); println(io,"name,n,m,converged,cycles,wpe,setup_s,solve_s,status"); flush(io)
    done=0; conv=0; skipped=0; failed=0; t0=time()
    for (idx,p) in enumerate(files)
        name=basename(p); local L
        try; L=lcc(read_mm_adj(p)); catch; skipped+=1; println(io,"$name,0,0,0,0,0,0,0,loaderr"); flush(io); continue; end
        n=size(L,1); m=(nnz(L)-n)÷2
        if n<50 || m<minm || nnz(L)>maxnnz; skipped+=1; println(io,"$name,$n,$m,0,0,0,0,0,skip(size)"); flush(io); continue; end
        Random.seed!(1); xt=randn(n); xt.-=sum(xt)/n; b=L*xt
        try
            ts=@elapsed h=setup(L); tso=@elapsed ((x,info)=solve(h,b))
            rel=norm(L*x-b)/norm(b); ok=rel<=tol*10
            lw=lamg_work(L,b,tol)
            done+=1; ok && (conv+=1)
            println(io,"$name,$n,$m,$(Int(ok)),$(info.cycles),$(lw.wpe),$ts,$tso,ok"); flush(io)
        catch e; failed+=1; println(io,"$name,$n,$m,0,0,0,0,0,solveerr"); flush(io); end
        idx%50==0 && @printf("[%d/%d] %.0fs solved=%d conv=%d skip=%d fail=%d\n",idx,length(files),time()-t0,done,conv,skipped,failed)
    end
    close(io); @printf("\nDONE: %d solved, %d converged (%.2f%%), %d skipped, %d errored.\n",done,conv,100*conv/max(done,1),skipped,failed)
end
run_sweep(corpus,outcsv,maxnnz,minm,tol)
