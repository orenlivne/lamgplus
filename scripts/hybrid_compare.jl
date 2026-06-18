# Hybrid (DRA-QC + LAMG SoC veto) vs DRA-QC vs LAMG+ on the families where DRA-QC
# is slow (grid-aligned anisotropy, high-contrast) plus controls. Iteration counts
# (DRA-QC and hybrid share the same K-cycle/FCG solve, so their iteration counts are
# directly comparable); LAMG+ shown as cycles.
using LAMG, LinearAlgebra, SparseArrays, Random, Printf
include(joinpath(@__DIR__, "..", "draqc_hybrid", "src", "DRAQCHybrid.jl"))
using .DRAQCHybrid
const H = DRAQCHybrid; const D = DRAQCHybrid.DRAQC

function aniso2d(nx, ny, ε)
    idx(i,j)=(j-1)*nx+i; I,J,V=Int[],Int[],Float64[]
    for j in 1:ny, i in 1:nx
        i<nx&&(push!(I,idx(i,j));push!(J,idx(i+1,j));push!(V,-1.0)); i>1&&(push!(I,idx(i,j));push!(J,idx(i-1,j));push!(V,-1.0))
        j<ny&&(push!(I,idx(i,j));push!(J,idx(i,j+1));push!(V,-ε)); j>1&&(push!(I,idx(i,j));push!(J,idx(i,j-1));push!(V,-ε))
    end
    W=sparse(I,J,V,nx*ny,nx*ny); sparse(Diagonal(-vec(sum(W;dims=2))))+W
end
function hicontrast2d(nx,ny;seed=1)
    rng=MersenneTwister(seed); idx(i,j)=(j-1)*nx+i; I,J,V=Int[],Int[],Float64[]
    ae(a,b)=(w=10.0^(7*rand(rng)-3.5); push!(I,a);push!(J,b);push!(V,-w);push!(I,b);push!(J,a);push!(V,-w))
    for j in 1:ny,i in 1:nx; i<nx&&ae(idx(i,j),idx(i+1,j)); j<ny&&ae(idx(i,j),idx(i,j+1)); end
    W=sparse(I,J,V,nx*ny,nx*ny); sparse(Diagonal(-vec(sum(W;dims=2))))+W
end
grid2d(nx,ny)=aniso2d(nx,ny,1.0)

cases = [
    ("aniso 128² ε=1e-1", aniso2d(128,128,1e-1)),
    ("aniso 128² ε=1e-2", aniso2d(128,128,1e-2)),
    ("aniso 128² ε=1e-4", aniso2d(128,128,1e-4)),
    ("aniso 256² ε=1e-4", aniso2d(256,256,1e-4)),
    ("hi-contrast 128²",  hicontrast2d(128,128)),
    ("isotropic 128²",    grid2d(128,128)),
]
tol = 1e-8; maxit = 1500
fmt(L, x, info, b) = norm(L*x - b)/norm(b) <= tol*10 ? string(info.iters) : ">$(maxit)"
@printf("%-20s %8s | %-9s | %-8s | %-8s | %-11s | %-12s\n",
        "family", "n", "LAMG+ cyc", "DRA-QC", "+SoC", "+SoC+el4", "+SoC+el4+c2")
println("-"^88)
for (name, L) in cases
    n = size(L,1); Random.seed!(1); xt=randn(n); xt.-=sum(xt)/n; b=L*xt
    o = LAMGOptions(tol=tol, max_cycles=200); hl = setup(L; options=o); (xl, il) = solve(hl, b; options=o)
    lc = norm(L*xl-b)/norm(b) <= tol*10 ? il.cycles : -1
    sd = D.DRAQCSolver(D.draqc_setup(L)); (xq, iq) = D.draqc_solve(sd, b; tol=tol, maxiter=maxit)
    sh = D.DRAQCSolver(H.hybrid_setup(L)); (xh, ih) = D.draqc_solve(sh, b; tol=tol, maxiter=maxit)
    (xe, ie, _) = H.hybrid_elim(L, b; dmax=4, tol=tol, maxiter=maxit)
    (xc, ic, _) = H.hybrid_elim(L, b; dmax=4, caliber2=true, tol=tol, maxiter=maxit)
    @printf("%-20s %8d | cyc=%-5d | %-8s | %-8s | %-11s | %-12s\n",
        name, n, lc, fmt(L,xq,iq,b), fmt(L,xh,ih,b), fmt(L,xe,ie,b), fmt(L,xc,ic,b))
end
