# Wall-clock benchmark of the newest hybrid (DRA-QC quality control + LAMG SoC veto
# + degree-≤4 elimination + caliber-2) vs LAMG+ and plain DRA-QC, across ALL test
# instances: the real SuiteSparse/SNAP/SPE graphs in data/ plus the synthetic
# adversarial + control families. Total end-to-end time (setup+solve), warm-timed.
#
# NOTE: wall-clock from these Julia reimplementations of DRA-QC / the hybrid is not
# authoritative vs an optimized Fortran DRA-QC; iteration counts are the fair metric.
using LAMG, LinearAlgebra, SparseArrays, Random, Printf
include(joinpath(@__DIR__, "..", "draqc_hybrid", "src", "DRAQCHybrid.jl"))
using .DRAQCHybrid
const H = DRAQCHybrid; const D = DRAQCHybrid.DRAQC

# ---- loaders ----
function read_mm_adj(path); open(path,"r") do io
    tok=split(lowercase(readline(io))); field=tok[4]; sym=tok[5]
    line=readline(io); while startswith(strip(line),"%"); line=readline(io); end
    nr,nc,ne=parse.(Int,split(line)); rows=Vector{Int}(undef,2ne); cols=similar(rows); vals=Vector{Float64}(undef,2ne); k=0
    for _ in 1:ne; p=split(readline(io)); i=parse(Int,p[1]); j=parse(Int,p[2]); v=field=="pattern" ? 1.0 : parse(Float64,p[3])
      k+=1; rows[k]=i; cols[k]=j; vals[k]=v; if sym in ("symmetric","hermitian","skew-symmetric")&&i!=j; k+=1; rows[k]=j; cols[k]=i; vals[k]=v; end; end
    Wr=sparse(rows[1:k],cols[1:k],vals[1:k],nr,nc); up=triu(Wr,1); W=up+sparse(transpose(up))
    if any(!iszero,diag(Wr)); ii,jj,_=findnz(W); W=sparse(ii,jj,ones(length(ii)),size(W)...); end
    nnz(W)>0&&minimum(nonzeros(W))<0&&(W=abs.(W))
    for j in 1:size(W,2),r in nzrange(W,j); W.rowval[r]==j&&(W.nzval[r]=0.0); end
    dropzeros!(W); laplacian(W); end; end
lcc(L)=(lab=LAMG.connected_components(L); M=maximum(lab); M==1 ? L : (s=zeros(Int,M); for l in lab; s[l]+=1; end; k=findall(==(argmax(s)),lab); L[k,k]))
function aniso2d(nx,ny,ε); idx(i,j)=(j-1)*nx+i; I,J,V=Int[],Int[],Float64[]
  for j in 1:ny,i in 1:nx
    i<nx&&(push!(I,idx(i,j));push!(J,idx(i+1,j));push!(V,-1.0)); i>1&&(push!(I,idx(i,j));push!(J,idx(i-1,j));push!(V,-1.0))
    j<ny&&(push!(I,idx(i,j));push!(J,idx(i,j+1));push!(V,-ε)); j>1&&(push!(I,idx(i,j));push!(J,idx(i,j-1));push!(V,-ε)); end
  W=sparse(I,J,V,nx*ny,nx*ny); sparse(Diagonal(-vec(sum(W;dims=2))))+W; end
function hicontrast2d(nx,ny;seed=1); rng=MersenneTwister(seed); idx(i,j)=(j-1)*nx+i; I,J,V=Int[],Int[],Float64[]
  ae(a,b)=(w=10.0^(7*rand(rng)-3.5); push!(I,a);push!(J,b);push!(V,-w);push!(I,b);push!(J,a);push!(V,-w))
  for j in 1:ny,i in 1:nx; i<nx&&ae(idx(i,j),idx(i+1,j)); j<ny&&ae(idx(i,j),idx(i,j+1)); end
  W=sparse(I,J,V,nx*ny,nx*ny); sparse(Diagonal(-vec(sum(W;dims=2))))+W; end
function grid3d(m); idx(i,j,k)=((k-1)*m+(j-1))*m+i; N=m^3; I,J,V=Int[],Int[],Float64[]
  for k in 1:m,j in 1:m,i in 1:m,(di,dj,dk) in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
    ii,jj,kk=i+di,j+dj,k+dk; (1<=ii<=m&&1<=jj<=m&&1<=kk<=m)&&(push!(I,idx(i,j,k));push!(J,idx(ii,jj,kk));push!(V,-1.0)); end
  W=sparse(I,J,V,N,N); sparse(Diagonal(-vec(sum(W;dims=2))))+W; end
star(k)=(W=spzeros(k+1,k+1); for j in 2:k+1; W[1,j]=1.0; W[j,1]=1.0; end; sparse(Diagonal(vec(sum(W;dims=2))))-W)

datadir = joinpath(@__DIR__, "..", "data")
real_graphs = ["GHS_psdef__bmwcra_1","Boeing__pwtk","SPE__spe10_2_nz20","SNAP__web-Stanford","SNAP__web-Google"]
cases = Tuple{String,SparseMatrixCSC}[]
for g in real_graphs
    p = joinpath(datadir, g*".mtx"); isfile(p) && push!(cases, (g, lcc(read_mm_adj(p))))
end
append!(cases, [
    ("aniso 128² ε=1e-4", aniso2d(128,128,1e-4)),
    ("aniso 256² ε=1e-4", aniso2d(256,256,1e-4)),
    ("hi-contrast 128²",  hicontrast2d(128,128)),
    ("grid3d 40³",        grid3d(40)),
    ("isotropic 256²",    aniso2d(256,256,1.0)),
    ("star 50k",          star(50_000)),
])

tol = 1e-8; maxit = 2000
@printf("%-22s %9s %10s | %-18s | %-22s\n", "instance", "n", "m", "LAMG+ (cyc, s)", "Hybrid* (it, s)")
println("-"^88)
for (name, L) in cases
    n=size(L,1); m=(nnz(L)-n)÷2; Random.seed!(1); xt=randn(n); xt.-=sum(xt)/n; b=L*xt
    o=LAMGOptions(tol=tol,max_cycles=200)
    setup(L;options=o); solve(setup(L;options=o),b;options=o)                     # warm
    tl=@elapsed (hl=setup(L;options=o); (xl,il)=solve(hl,b;options=o))
    lok = norm(L*xl-b)/norm(b)<=tol*10
    H.hybrid_elim(L,b;dmax=4,caliber2=true,tol=tol,maxiter=maxit)                 # warm
    th=@elapsed ((xh,ih,_)=H.hybrid_elim(L,b;dmax=4,caliber2=true,tol=tol,maxiter=maxit))
    hok = norm(L*xh-b)/norm(b)<=tol*10
    @printf("%-22s %9d %10d | cyc=%-3d %6.2fs %s | it=%-4d %6.2fs %s\n",
        name, n, m, il.cycles, tl, lok ? "✓" : "✗", ih.iters, th, hok ? "✓" : "✗")
end
println("\nHybrid* = DRA-QC quality control + SoC veto + degree-≤4 elimination + caliber-2")
