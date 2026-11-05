using SparseArrays, LinearAlgebra, Random, Statistics, Printf
include("/Users/oren/code/aier/lamgplus/scripts/mm_loader.jl")
const DATA = "/Users/oren/code/aier/data/suitesparse"
for (nm,f) in (("webStanford","SNAP__web-Stanford.mtx"), ("delaunay18","DIMACS10__delaunay_n18.mtx"), ("delaunay17","DIMACS10__delaunay_n17.mtx"))
    W,L = reduce_to_lcc(read_mm_adj(joinpath(DATA,f))...); n=size(L,1)
    I,J,V = findnz(L)
    open(joinpath(ENV["OUT"], "$(nm)_A.txt"), "w") do io
        for k in eachindex(I); @printf(io, "%d %d %.17g\n", I[k], J[k], V[k]); end
    end
    bR = randn(MersenneTwister(12), n); bR .-= mean(bR)
    open(joinpath(ENV["OUT"], "$(nm)_b.txt"), "w") do io; for v in bR; @printf(io, "%.17g\n", v); end; end
    println("$nm n=$n nnz=$(nnz(L))")
end
