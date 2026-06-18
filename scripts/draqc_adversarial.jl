# DRA-QC vs LAMG+ on adversarial families that Napov–Notay never tested
# (grid-aligned anisotropy — LAMG's documented weakness that LAMG+'s caliber-2
# fixes — plus high-contrast grids, 3-D grids, and stars). Algorithm-level metrics
# (iterations / cycles, convergence); wall-clock from this unoptimized DRA-QC
# reimplementation is NOT authoritative.
using LAMG, LinearAlgebra, SparseArrays, Random, Printf
include(joinpath(@__DIR__, "..", "draqc", "src", "DRAQC.jl"))
using .DRAQC

function aniso2d(nx, ny, ε)
    idx(i, j) = (j - 1) * nx + i; I, J, V = Int[], Int[], Float64[]
    for j in 1:ny, i in 1:nx
        i < nx && (push!(I, idx(i,j)); push!(J, idx(i+1,j)); push!(V, -1.0))
        i > 1  && (push!(I, idx(i,j)); push!(J, idx(i-1,j)); push!(V, -1.0))
        j < ny && (push!(I, idx(i,j)); push!(J, idx(i,j+1)); push!(V, -ε))
        j > 1  && (push!(I, idx(i,j)); push!(J, idx(i,j-1)); push!(V, -ε))
    end
    W = sparse(I, J, V, nx*ny, nx*ny); sparse(Diagonal(-vec(sum(W; dims=2)))) + W
end

# high-contrast: random edge weights spanning ~7 decades (à la SPE).
function hicontrast2d(nx, ny; seed=1)
    rng = MersenneTwister(seed); idx(i, j) = (j - 1) * nx + i
    I, J, V = Int[], Int[], Float64[]
    addedge(a, b) = (w = 10.0^(7*rand(rng) - 3.5); push!(I,a); push!(J,b); push!(V,-w); push!(I,b); push!(J,a); push!(V,-w))
    for j in 1:ny, i in 1:nx
        i < nx && addedge(idx(i,j), idx(i+1,j))
        j < ny && addedge(idx(i,j), idx(i,j+1))
    end
    W = sparse(I, J, V, nx*ny, nx*ny); sparse(Diagonal(-vec(sum(W; dims=2)))) + W
end

function grid3d(n)
    idx(i,j,k) = ((k-1)*n + (j-1))*n + i; N = n^3; I,J,V = Int[],Int[],Float64[]
    for k in 1:n, j in 1:n, i in 1:n, (di,dj,dk) in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
        ii,jj,kk = i+di, j+dj, k+dk
        (1<=ii<=n && 1<=jj<=n && 1<=kk<=n) && (push!(I,idx(i,j,k)); push!(J,idx(ii,jj,kk)); push!(V,-1.0))
    end
    W = sparse(I,J,V,N,N); sparse(Diagonal(-vec(sum(W;dims=2)))) + W
end

star(k) = (W = spzeros(k+1,k+1); for j in 2:k+1; W[1,j]=1.0; W[j,1]=1.0; end; sparse(Diagonal(vec(sum(W;dims=2))))-W)

cases = [
    ("aniso 128² ε=1e-1", aniso2d(128,128,1e-1)),
    ("aniso 128² ε=1e-2", aniso2d(128,128,1e-2)),
    ("aniso 128² ε=1e-4", aniso2d(128,128,1e-4)),
    ("aniso 256² ε=1e-4", aniso2d(256,256,1e-4)),
    ("hi-contrast 128²",  hicontrast2d(128,128)),
    ("grid3d 40³",        grid3d(40)),
    ("star 50k",          star(50_000)),
]

tol = 1e-8; maxit = 1500
@printf("%-22s %9s | %-12s | %-26s\n", "family", "n", "LAMG+ cyc", "DRA-QC iters / converged")
println("-"^78)
for (name, L) in cases
    n = size(L, 1)
    Random.seed!(1); xt = randn(n); xt .-= sum(xt)/n; b = L * xt
    opts = LAMGOptions(tol=tol, max_cycles=200)
    local lcyc, lok
    try
        h = setup(L; options=opts); (x,info) = solve(h, b; options=opts)
        lcyc = info.cycles; lok = norm(L*x-b)/norm(b) <= tol*10
    catch e; lcyc = -1; lok = false; end
    hd = DRAQC.draqc_setup(L); s = DRAQC.DRAQCSolver(hd)
    (xd, infod) = DRAQC.draqc_solve(s, b; tol=tol, maxiter=maxit)
    drel = norm(L*xd - b)/norm(b); dok = drel <= tol*10
    status = dok ? (infod.iters <= 40 ? "✓" : "✓ but SLOW") : "✗ >$(maxit)it"
    @printf("%-22s %9d | cyc=%-3d %s | it=%-4d rel=%.1e %s  (OC=%.2f)\n",
        name, n, lcyc, lok ? "✓" : "✗", infod.iters, drel, status, DRAQC.operator_complexity(hd))
end
