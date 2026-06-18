# Fair work-per-edge comparison: LAMG+ vs DRA-QC. Iteration counts alone are unfair
# (a DRA-QC K-cycle iteration and a LAMG+ γ-cycle differ in cost), so we count the
# total operator work across the whole solve, normalized per edge — folding in
# iteration count, operator complexity, and the cycle's per-level visit pattern.
#
# Work model (identical accounting for both, units = nnz multiply-adds):
#   WPE = n_iter · Σ_ℓ visits_ℓ · ( PASSES·nnz(A_ℓ) + transfer_ℓ ) / m
# where PASSES = smoothing sweeps + residual matvecs per visit (5 for both: LAMG+
# does 2+2 GS + 1 residual; DRA-QC does 1+1 GS + ~3 matvec incl. the FCG inner
# matvec), transfer_ℓ = nnz of the restriction+prolongation at level ℓ, and
# visits_ℓ is the method's own cycle-visit count (LAMG+ γ=1.5 truncated; DRA-QC
# K-cycle doubling with the n^{1/3} work cutoff). Both solve to 1e-8 (≈8 digits),
# so WPE is directly comparable; WPE/8 is work-per-digit.
using LAMG, LinearAlgebra, SparseArrays, Random, Printf
include(joinpath(@__DIR__, "..", "draqc_hybrid", "src", "DRAQCHybrid.jl"))
const D = DRAQCHybrid.DRAQC

const PASSES = 5

# γ=1.5 truncated cycle visit counts per relative level (ports run_cycle! accounting)
function gamma_visits(γ::Float64, L::Int)
    L == 1 && return [1]
    v = zeros(Int, L); num = zeros(Int, L - 1)   # num[i] = descents from level i
    l = 1
    while true
        i = l
        if l == L
            k = l - 1
        else
            maxv = (i == 1) ? 1 : γ * num[i - 1]
            k = (num[i] < maxv) ? (l + 1) : (l - 1)
        end
        l == L && (v[L] += 1)
        if k < 1; break
        elseif k > l; num[i] += 1; v[l] += 1
        end
        l = k
    end
    return v
end

# DRA-QC K-cycle: level ℓ+1 visited 2× per visit of ℓ when n_{ℓ+1} > n_ℓ^{1/3}, else 1×.
function kcycle_visits(sizes::Vector{Int})
    L = length(sizes); v = ones(Int, L)
    for ℓ in 1:L-1
        v[ℓ+1] = v[ℓ] * (sizes[ℓ+1] > sizes[ℓ]^(1/3) ? 2 : 1)
    end
    return v
end

# work-per-edge from per-level (nnz_a, transfer_nnz), visits, iterations
function wpe(nnz_a, transfer, visits, niter, m)
    percyc = sum(visits[ℓ] * (PASSES * nnz_a[ℓ] + transfer[ℓ]) for ℓ in 1:length(nnz_a))
    return niter * percyc / m
end

function lamg_work(L, b, tol)
    h = setup(L); (x, info) = solve(h, b)
    lev = h.levels; nl = length(lev)
    nnz_a = [nnz(lev[ℓ].a) for ℓ in 1:nl]
    transfer = [ (lev[ℓ].p === nothing ? 0 : nnz(lev[ℓ].p)) + (lev[ℓ].r === nothing ? 0 : nnz(lev[ℓ].r)) for ℓ in 1:nl ]
    v = gamma_visits(1.5, nl)
    m = (nnz(L) - size(L,1)) ÷ 2
    oc = sum(nnz_a) / nnz_a[1]
    return (cyc=info.cycles, oc=oc, wpe=wpe(nnz_a, transfer, v, info.cycles, m),
            ok = norm(L*x-b)/norm(b) <= tol*10)
end

function draqc_work(L, b, tol; maxiter=2000)
    h = D.draqc_setup(L); s = D.DRAQCSolver(h); (x, info) = D.draqc_solve(s, b; tol=tol, maxiter=maxiter)
    nl = D.num_levels(h)
    nnz_a = [nnz(h.A[ℓ]) for ℓ in 1:nl]
    transfer = [ ℓ <= length(h.P) ? 2*nnz(h.P[ℓ]) : 0 for ℓ in 1:nl ]
    sizes = [size(h.A[ℓ],1) for ℓ in 1:nl]
    v = kcycle_visits(sizes)
    m = (nnz(L) - size(L,1)) ÷ 2
    oc = sum(nnz_a) / nnz_a[1]
    return (it=info.iters, oc=oc, wpe=wpe(nnz_a, transfer, v, info.iters, m),
            ok = norm(L*x-b)/norm(b) <= tol*10)
end

# ---------- loaders + curated subset ----------
function read_mm_adj(path); open(path,"r") do io
  tok=split(lowercase(readline(io))); field=tok[4]; sym=tok[5]
  line=readline(io); while startswith(strip(line),"%"); line=readline(io); end
  nr,nc,ne=parse.(Int,split(line)); rows=Vector{Int}(undef,2ne); cols=similar(rows); vals=Vector{Float64}(undef,2ne); k=0
  for _ in 1:ne; p=split(readline(io)); i=parse(Int,p[1]); j=parse(Int,p[2]); v=field=="pattern" ? 1.0 : parse(Float64,p[3])
    k+=1; rows[k]=i; cols[k]=j; vals[k]=v; if sym in ("symmetric","hermitian","skew-symmetric")&&i!=j; k+=1; rows[k]=j; cols[k]=i; vals[k]=v; end; end
  Wr=sparse(rows[1:k],cols[1:k],vals[1:k],nr,nc); up=triu(Wr,1); W=up+sparse(transpose(up))
  if any(!iszero,diag(Wr)); ii,jj,_=findnz(W); W=sparse(ii,jj,ones(length(ii)),size(W)...); end
  nnz(W)>0&&minimum(nonzeros(W))<0&&(W=abs.(W)); for j in 1:size(W,2),r in nzrange(W,j); W.rowval[r]==j&&(W.nzval[r]=0.0); end
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

dd = joinpath(@__DIR__,"..","data")
cases = Tuple{String,SparseMatrixCSC}[]
for g in ("Boeing__pwtk","GHS_psdef__bmwcra_1","SPE__spe10_2_nz20","SNAP__web-Stanford")
  p=joinpath(dd,g*".mtx"); isfile(p) && push!(cases,(g, lcc(read_mm_adj(p))))
end
append!(cases, [("aniso 128² ε=1e-2",aniso2d(128,128,1e-2)),
                ("aniso 128² ε=1e-4",aniso2d(128,128,1e-4)),
                ("hi-contrast 128²",hicontrast2d(128,128)),
                ("isotropic 256²",aniso2d(256,256,1.0))])

tol=1e-8
@printf("%-22s %9s | %-26s | %-26s | %s\n","instance","n",
        "LAMG+  cyc OC  WPE","DRA-QC it  OC  WPE","WPE ratio (DRAQC/LAMG+)")
println("-"^104)
for (name,L) in cases
  n=size(L,1); Random.seed!(1); xt=randn(n); xt.-=sum(xt)/n; b=L*xt
  lw=lamg_work(L,b,tol); dw=draqc_work(L,b,tol)
  r = dw.wpe/lw.wpe
  @printf("%-22s %9d | %3d  %4.2f  %7.1f %s | %4d  %4.2f  %7.1f %s | %5.1f×  %s\n",
    name,n, lw.cyc,lw.oc,lw.wpe, lw.ok ? "✓" : "✗",
    dw.it,dw.oc,dw.wpe, dw.ok ? "✓" : "✗", r, r>1 ? "(LAMG+ leaner)" : "(DRA-QC leaner)")
end
println("\nWPE = work per edge (nnz multiply-adds / edge) to 1e-8; PASSES=$(PASSES)/visit; both ≈8 digits.")
