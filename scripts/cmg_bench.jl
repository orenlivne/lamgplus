# Standalone CMG (Koutis Combinatorial Multigrid) benchmark for ONE graph-Laplacian adjacency .mtx.
# Run via the cmg_env (CombinatorialMultigrid + Laplacians) as a subprocess from the main harness,
# because CMG's dependency tree conflicts (MathOptInterface) with the competitor_env's JuMP/HiGHS.
# Reads a symmetric non-negative adjacency W (lower-triangle Matrix Market), builds L = lap(W),
# and times CMG-preconditioned PCG to relative residual 1e-8. Times are measured internally
# (exclude Julia startup). Output: "<setup_s> <solve_s> <ok:0|1>"  (or "nan nan 0" on failure).
import Pkg
Pkg.activate(joinpath(@__DIR__, "cmg_env"))
using CombinatorialMultigrid, Laplacians, SparseArrays, LinearAlgebra
const TOL = 1e-8; const MAXIT = 300

function readadj(path)
    open(path) do io
        readline(io); l = readline(io); while startswith(strip(l), "%"); l = readline(io); end
        n, _, ne = parse.(Int, split(l)); I = Int[]; J = Int[]; V = Float64[]
        for _ in 1:ne
            p = split(readline(io)); i = parse(Int, p[1]); j = parse(Int, p[2])
            v = length(p) >= 3 ? parse(Float64, p[3]) : 1.0
            push!(I, i); push!(J, j); push!(V, v)
            i != j && (push!(I, j); push!(J, i); push!(V, v))
        end
        W = sparse(I, J, V, n, n)
        for j in 1:n, k in nzrange(W, j); rowvals(W)[k] == j && (nonzeros(W)[k] = 0.0); end
        dropzeros!(W); W
    end
end

try
    W = readadj(ARGS[1]); n = size(W, 1)
    L = Laplacians.lap(W)
    b = randn(n); b .-= sum(b) / n
    pf0, _ = cmg_preconditioner_lap(L); Laplacians.pcg(L, b, pf0; tol=TOL, maxits=MAXIT)   # warm
    t0 = time(); pf, _ = cmg_preconditioner_lap(L); t1 = time()
    x = Laplacians.pcg(L, b, pf; tol=TOL, maxits=MAXIT); t2 = time()
    rel = norm(L * x - b) / max(norm(b), 1e-30)
    println("$(t1-t0) $(t2-t1) $(rel <= TOL*100 ? 1 : 0)")
catch e
    println("nan nan 0")
end
