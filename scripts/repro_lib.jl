# repro_lib.jl — minimal, dependency-faithful reproduction helpers for the
# LAMG+ vs approxChol timing/accuracy comparison (Table "Multi-solver comparison").
# Extracted verbatim (loader + solver timing) from scripts/competitor_benchmark.jl
# so the reproduced numbers use the SAME code path as the paper.
#
# Usage (from a Julia session with this repo's competitor_env active):
#   include("scripts/repro_lib.jl")
#   r = repro_graph("data/SNAP__web-Stanford.mtx"); show_repro(r, reported=(0.26,0.19))
using LAMG, Laplacians, LinearAlgebra, SparseArrays, Random, Statistics, Printf

function read_mm_adj(path::AbstractString)
    open(path, "r") do io
        header = readline(io); tokens = split(lowercase(header))
        @assert tokens[1]=="%%matrixmarket" && tokens[2]=="matrix"
        field=tokens[4]; symmetry=tokens[5]
        line = readline(io); while startswith(strip(line),"%"); line=readline(io); end
        nrows,ncols,nentries = parse.(Int, split(line))
        rows=Vector{Int}(undef,2nentries); cols=similar(rows); vals=Vector{Float64}(undef,2nentries); k=0
        for _ in 1:nentries
            parts=split(readline(io)); i=parse(Int,parts[1]); j=parse(Int,parts[2])
            v = field=="pattern" ? 1.0 : (field=="complex" ? sqrt(parse(Float64,parts[3])^2+parse(Float64,parts[4])^2) : parse(Float64,parts[3]))
            k+=1; rows[k]=i; cols[k]=j; vals[k]=v
            if symmetry in ("symmetric","hermitian","skew-symmetric") && i!=j
                k+=1; rows[k]=j; cols[k]=i; vals[k]=v
            end
        end
        W_raw=sparse(rows[1:k],cols[1:k],vals[1:k],nrows,ncols)
        upper=triu(W_raw,1); W=upper+sparse(transpose(upper))
        had_diag=any(!iszero,diag(W_raw))
        if had_diag
            ii,jj,_=findnz(W); mn,nn=size(W); W=sparse(ii,jj,ones(length(ii)),mn,nn)
        end
        if !had_diag && nnz(W)>0 && minimum(nonzeros(W))<0
            W=abs.(W)
        end
        for j in 1:size(W,2), r in nzrange(W,j); W.rowval[r]==j && (W.nzval[r]=0.0); end
        dropzeros!(W)
        return W, laplacian(W)
    end
end
function reduce_to_lcc(W,L)
    label=LAMG.connected_components(L); M=maximum(label); M==1 && return W,L
    sizes=zeros(Int,M); for l in label; sizes[l]+=1; end
    keep=findall(==(argmax(sizes)),label); return W[keep,keep], L[keep,keep]
end

# returns (setup_us, solve_us, ok, rel, iters_or_cycles)
function bench_lamg(L,b,tol,repeats)
    opts=LAMGOptions(tol=tol,max_cycles=100)
    h0=setup(L;options=opts); solve(h0,b;options=opts)         # warm-up
    su=Float64[];so=Float64[];cyc=0;x=nothing
    for _ in 1:repeats
        t=@elapsed h=setup(L;options=opts); push!(su,t)
        t2=@elapsed ((x,info)=solve(h,b;options=opts)); push!(so,t2); cyc=info.cycles
    end
    rel=norm(L*x-b)/max(norm(b),1e-30)
    (median(su)*1e6, median(so)*1e6, rel<=tol*10, rel, cyc)
end
function bench_approxchol(W,b,tol,repeats)
    f0=Laplacians.approxchol_lap(W;tol=tol,verbose=false); f0(b)   # warm-up
    su=Float64[];so=Float64[];it=0;x=nothing
    for _ in 1:repeats
        t=@elapsed f=Laplacians.approxchol_lap(W;tol=tol,verbose=false); push!(su,t)
        pcg=[0]; t2=@elapsed x=f(b;tol=tol,verbose=false,pcgIts=pcg); push!(so,t2); it=pcg[1]
    end
    L=LAMG.laplacian(W); rel=norm(L*x-b)/max(norm(b),1e-30)
    (median(su)*1e6, median(so)*1e6, rel<=tol*100, rel, it)
end

function repro_graph(path; tol=1e-8, repeats=2)
    W,L=read_mm_adj(path); W,L=reduce_to_lcc(W,L)
    n=size(L,1); nnzL=nnz(L); m=div(nnzL-n,2)
    rng=MersenneTwister(0xff+n); xt=randn(rng,n); xt.-=sum(xt)/n; b=L*xt
    ls,lso,lok,lrel,lcyc=bench_lamg(L,b,tol,repeats)
    as,aso,aok,arel,ait=bench_approxchol(W,b,tol,repeats)
    nnz_norm=n+2m
    (name=basename(path), n=n, m=m, nnz=nnz_norm,
     lamg_us_per_nnz=(ls+lso)/nnz_norm, lamg_cycles=lcyc, lamg_rel=lrel, lamg_ok=lok,
     apx_us_per_nnz=(as+aso)/nnz_norm, apx_iters=ait, apx_rel=arel, apx_ok=aok,
     winner=((ls+lso)<(as+aso) ? "LAMG+" : "approxChol"))
end
