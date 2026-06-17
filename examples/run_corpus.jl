#!/usr/bin/env julia
# examples/run_corpus.jl — sweep many graphs, time LAMG+ on each, and fit the O(m) scaling.
#
#   Usage:  julia --project=. examples/run_corpus.jl <list.txt> [out.csv] [tol]
#   E.g.:   julia --project=. examples/run_corpus.jl data/lamg_full_list.txt corpus.csv 1e-8
#
# <list.txt> has one graph per line, either "Group/Name" (mapped to
# data/Group__Name.mtx) or a path/filename under data/. Writes a CSV with, per
# graph, n, m, levels, setup_s, solve_s, cycles, the achieved residual, and
# per-edge time; then reports the empirical scaling exponent beta in
# total_time ~ m^beta (beta ~ 1 is the O(m) statement) and the convergence rate.

using LAMG, LinearAlgebra, SparseArrays, Random, Statistics, Printf
import LAMG: solve, setup, LAMGOptions
include(joinpath(@__DIR__, "mm_reader.jl"))

length(ARGS) >= 1 || error("usage: julia --project=. examples/run_corpus.jl <list.txt> [out.csv] [tol]")
list = ARGS[1]
out  = length(ARGS) >= 2 ? ARGS[2] : "corpus_results.csv"
tol  = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1e-8

to_path(s) = (s = strip(s); (isempty(s) || startswith(s, "#")) ? nothing :
              occursin("/", s) ? "data/$(replace(s, "/" => "__")).mtx" :
              (isfile(s) ? s : "data/$s"))

function sweep(paths)
    io = open(out, "w")
    println(io, "name,n,m,levels,setup_s,solve_s,cycles,residual,converged,per_edge_us"); flush(io)
    ms = Float64[]; ts = Float64[]; nconv = 0; nrun = 0
    for p in paths
        (p === nothing || !isfile(p)) && continue
        local W, L
        try; W, L = reduce_to_lcc(read_mm_adj(p)...); catch; continue; end
        n = size(L, 1); m = (nnz(L) - n) ÷ 2; m < 50 && continue
        rng = MersenneTwister(0xff + n); xt = randn(rng, n); xt .-= sum(xt) / n
        b = L * xt; bn = norm(b); o = LAMGOptions(tol = tol, max_cycles = 200)
        local h, info, x
        try
            tb = @elapsed (h = setup(L; options = o))
            tv = @elapsed ((x, info) = solve(h, b; options = o))
        catch; continue; end
        res = norm(b - L * x) / bn; cv = res <= tol; nrun += 1; nconv += cv
        push!(ms, m); push!(ts, tb + tv)
        @printf(io, "%s,%d,%d,%d,%.5f,%.5f,%d,%.3e,%d,%.4f\n",
                basename(p), n, m, length(h.levels), tb, tv, info.cycles, res, Int(cv), 1e6 * (tb + tv) / m)
        flush(io); nrun % 100 == 0 && (@printf("  [%d] converged %d\n", nrun, nconv); flush(stdout))
    end
    close(io); (ms, ts, nrun, nconv)
end

ms, tsum, nrun, nconv = sweep(map(to_path, readlines(list)))
lx = log.(ms); ly = log.(tsum); mx = mean(lx); my = mean(ly)
beta = sum((lx .- mx) .* (ly .- my)) / sum((lx .- mx) .^ 2)
@printf("\n%d graphs solved | %d converged (%.1f%%) to %.0e\n", nrun, nconv, 100 * nconv / max(nrun, 1), tol)
@printf("empirical scaling: total_time ~ m^%.3f   (1.0 = O(m));  CSV -> %s\n", beta, out)
