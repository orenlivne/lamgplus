# Standalone PETSc GAMG benchmark for ONE graph-Laplacian adjacency .mtx, run as a subprocess from
# run_class_comparison.jl. Isolated because PETSc's GAMG setup can hard-crash (C-level abort/OOM) on
# low-diameter or extreme-contrast graphs, which would take down the whole in-process harness; a
# subprocess crash is caught as a DNF instead. Builds L=lap(W), Dirichlet-pins one node to make an
# SPD A, and times CG + GAMG (PETSc_jll) to relative residual 1e-8. Times measured internally
# (exclude Julia/PETSc startup). Output: "<setup_s> <solve_s> <ok:0|1>"  (or "nan nan 0" on failure).
import Pkg; Pkg.activate(joinpath(@__DIR__, "competitor_env"))
using PETSc, SparseArrays, LinearAlgebra, Statistics, Random
const TOL = 1e-8; const MAXIT = 300
const PETSCLIB = PETSc.petsclibs[1]; PETSc.initialize(PETSCLIB)
const PCOMM = PETSc.LibPETSc.PETSC_COMM_SELF

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
lap(W) = spdiagm(0 => vec(sum(W, dims=2))) - W

try
    W = readadj(ARGS[1]); n = size(W, 1); L = lap(W)
    rng = MersenneTwister(1); xt = randn(rng, n); xt .-= sum(xt)/n; b = L*xt
    A = L[2:n, 2:n]; bc = b[2:n]                                   # Dirichlet-pin node 1 -> SPD
    k0 = PETSc.KSP(PETSCLIB, PCOMM, A; ksp_type="cg", pc_type="gamg", ksp_rtol=TOL, ksp_max_it=MAXIT); k0\bc  # warm
    t0 = time(); ksp = PETSc.KSP(PETSCLIB, PCOMM, A; ksp_type="cg", pc_type="gamg", ksp_rtol=TOL, ksp_max_it=MAXIT); t1 = time()
    y = ksp\bc; t2 = time()
    x = vcat(0.0, y); x .-= mean(x)
    rel = norm(L*x - b) / max(norm(b), 1e-30)
    println("$(t1-t0) $(t2-t1) $(rel <= TOL*100 ? 1 : 0)")
catch e
    println("nan nan 0")
end
