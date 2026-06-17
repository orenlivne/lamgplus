#!/usr/bin/env julia
# Full competition sweep over all local graphs with m > 1e6 edges (nnz_L <= cap),
# at BOTH tol=1e-8 and tol=1e-4, for LAMG+, approxChol (approxchol_lap) and the
# robust AC (approxchol_lap2). Each solver is BUILT ONCE per graph (setup is
# tolerance-independent) and its solve timed at both tolerances -> a matched pair.
# Outputs one CSV row per graph; aggregate win-rates computed separately.
#
# Usage: julia --project=scripts/competitor_env scripts/bench_env/run_full_tol_sweep.jl [--cap=20000000] [--out=PATH]
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "competitor_env"))
using LAMG, SparseArrays, LinearAlgebra, Random, Statistics, Printf
using Laplacians

# ---- robust Matrix Market loader (mirrors competitor_benchmark.jl: FE abs, LCC) ----
function read_mm_adj(path::AbstractString)
    open(path, "r") do io
        tokens = split(lowercase(readline(io)))
        field = tokens[4]; symmetry = tokens[5]
        line = readline(io); while startswith(strip(line), "%"); line = readline(io); end
        nrows, ncols, nentries = parse.(Int, split(line))
        rows = Vector{Int}(undef, 2nentries); cols = Vector{Int}(undef, 2nentries); vals = Vector{Float64}(undef, 2nentries)
        k = 0
        for _ in 1:nentries
            parts = split(readline(io)); i = parse(Int, parts[1]); j = parse(Int, parts[2])
            v = field == "pattern" ? 1.0 :
                field == "complex" ? sqrt(parse(Float64, parts[3])^2 + parse(Float64, parts[4])^2) :
                parse(Float64, parts[3])
            k += 1; rows[k] = i; cols[k] = j; vals[k] = v
            if symmetry in ("symmetric", "hermitian", "skew-symmetric") && i != j
                k += 1; rows[k] = j; cols[k] = i; vals[k] = v
            end
        end
        W_raw = sparse(rows[1:k], cols[1:k], vals[1:k], nrows, ncols)
        upper = triu(W_raw, 1); W = upper + sparse(transpose(upper))
        had_diag = any(!iszero, diag(W_raw))
        if had_diag
            ii, jj, _ = findnz(W); W = sparse(ii, jj, ones(length(ii)), size(W)...)
        end
        if !had_diag && nnz(W) > 0 && minimum(nonzeros(W)) < 0
            rW = rowvals(W); vW = nonzeros(W); nW = size(W, 1)
            rneg = zeros(nW); rmax = zeros(nW)
            for j in 1:nW, r in nzrange(W, j)
                i = rW[r]; v = vW[r]; v < 0 && (rneg[i] += v); rmax[i] = max(rmax[i], abs(v))
            end
            any(i -> rneg[i] < -1e-5 * rmax[i], 1:nW) && (W = abs.(W))
        end
        for j in 1:size(W, 2), r in nzrange(W, j); W.rowval[r] == j && (W.nzval[r] = 0.0); end
        if nnz(W) > 0
            thr = sqrt(eps()) * maximum(abs, nonzeros(W))
            for r in 1:nnz(W); abs(W.nzval[r]) < thr && (W.nzval[r] = 0.0); end
        end
        dropzeros!(W); return W, laplacian(W)
    end
end
function reduce_to_lcc(W, L)
    label = LAMG.connected_components(L); M = maximum(label); M == 1 && return W, L
    sizes = zeros(Int, M); for l in label; sizes[l] += 1; end
    keep = findall(==(argmax(sizes)), label); return W[keep, keep], L[keep, keep]
end

best(f, k) = (minimum((@elapsed f()) for _ in 1:k))
relres(L, x, b) = norm(L * x - b) / max(norm(b), 1e-30)

cap = 20_000_000
lim = typemax(Int)
reps = 1                 # timed runs per measurement (1 = single warm run; after an untimed warmup)
nshards = 1; shard = 0   # shard==-1 means "giants only" (header nnz > giant_nnz), run serially
giant_nnz = 7_000_000
out = joinpath(@__DIR__, "..", "..", "full_tol_sweep.csv")
for a in ARGS
    startswith(a, "--cap=") && (global cap = parse(Int, split(a, "=")[2]))
    startswith(a, "--limit=") && (global lim = parse(Int, split(a, "=")[2]))
    startswith(a, "--reps=") && (global reps = parse(Int, split(a, "=")[2]))
    startswith(a, "--nshards=") && (global nshards = parse(Int, split(a, "=")[2]))
    startswith(a, "--shard=") && (global shard = parse(Int, split(a, "=")[2]))
    startswith(a, "--giant-nnz=") && (global giant_nnz = parse(Int, split(a, "=")[2]))
    startswith(a, "--out=") && (global out = split(a, "=", limit=2)[2])
end

# instance list: all local .mtx with header nnz > 1.0e6 (real m>1e6 filtered after LCC),
# tagged with header nnz so we can isolate the memory-heavy giants into one serial shard.
datadir = joinpath(@__DIR__, "..", "..", "data")
all_cands = Tuple{String,Int}[]
for f in readdir(datadir)
    endswith(f, ".mtx") || continue
    p = joinpath(datadir, f)
    try
        open(p) do io
            readline(io); line = readline(io)
            while startswith(strip(line), "%"); line = readline(io); end
            nnzh = parse(Int, split(line)[3])
            nnzh > 1_000_000 && nnzh < cap && push!(all_cands, (f, nnzh))
        end
    catch; end
end
sort!(all_cands, by = x -> x[1])
# Shard selection. Giants (header nnz > giant_nnz) run only in the dedicated giant shard
# (shard == -1), serially, so two memory-heavy graphs never co-run; everything else is
# round-robin across nshards small workers so memory bandwidth is not saturated.
if shard == -1
    cands = [f for (f, z) in all_cands if z > giant_nnz]
else
    smalls = [f for (f, z) in all_cands if z <= giant_nnz]
    cands = [f for (i, f) in enumerate(smalls) if (i - 1) % nshards == shard]
end
println("# shard=$shard nshards=$nshards reps=$reps -> $(length(cands)) graphs (of $(length(all_cands)) candidates)")

# global JIT warm-up
let (W, L) = read_mm_adj(joinpath(datadir, "SNAP__ca-HepTh.mtx"))
    W, L = reduce_to_lcc(W, L); n = size(L, 1); b = L * randn(n); b .-= sum(b) / n
    h = setup(L); solve(h, b)
    approxchol_lap(W)(b); approxchol_lap2(W)(b)
end

io = open(out, "w")
println(io, "instance,n,m,lp_setup,lp_s8,cyc8,lp_s4,cyc4,lp_ok,ac_setup,ac_s8,ac_s4,ac_ok,ac2_setup,ac2_s8,ac2_s4,ac2_ok")
flush(io)
o8 = LAMGOptions(tol = 1e-8, max_cycles = 200); o4 = LAMGOptions(tol = 1e-4, max_cycles = 200)
done = 0
for (idx, f) in enumerate(cands)
    done >= lim && break
    p = joinpath(datadir, f)
    local W, L
    try; (W, L) = read_mm_adj(p); (W, L) = reduce_to_lcc(W, L)
    catch e; @warn "load $f" e; continue; end
    n = size(L, 1); nnzL = nnz(L); m = div(nnzL - n, 2)
    (m <= 1_000_000 || nnzL > cap) && continue
    rng = MersenneTwister(0xff + n); xt = randn(rng, n); xt .-= sum(xt) / n; b = L * xt

    lp_setup = lp_s8 = lp_s4 = NaN; cyc8 = cyc4 = 0; lp_ok = false
    try
        lp_setup = best(() -> setup(L; options = o8), reps)
        h = setup(L; options = o8)
        x8, i8 = solve(h, b; options = o8); cyc8 = i8.cycles
        x4, i4 = solve(h, b; options = o4); cyc4 = i4.cycles
        lp_s8 = best(() -> solve(h, b; options = o8), reps)
        lp_s4 = best(() -> solve(h, b; options = o4), reps)
        lp_ok = relres(L, x8, b) <= 1e-7
    catch e; @warn "lamg $f" e; end

    ac_setup = ac_s8 = ac_s4 = NaN; ac_ok = false
    try
        ac_setup = best(() -> approxchol_lap(W), reps)
        fac = approxchol_lap(W)
        x8 = fac(b; tol = 1e-8); x4 = fac(b; tol = 1e-4)
        ac_s8 = best(() -> fac(b; tol = 1e-8), reps)
        ac_s4 = best(() -> fac(b; tol = 1e-4), reps)
        ac_ok = relres(L, x8, b) <= 1e-6
    catch e; @warn "approxchol $f" e; end

    ac2_setup = ac2_s8 = ac2_s4 = NaN; ac2_ok = false
    try
        ac2_setup = best(() -> approxchol_lap2(W), reps)
        f2 = approxchol_lap2(W)
        x8 = f2(b; tol = 1e-8); x4 = f2(b; tol = 1e-4)
        ac2_s8 = best(() -> f2(b; tol = 1e-8), reps)
        ac2_s4 = best(() -> f2(b; tol = 1e-4), reps)
        ac2_ok = relres(L, x8, b) <= 1e-6
    catch e; @warn "approxchol2 $f" e; end

    @printf(io, "%s,%d,%d,%.4f,%.4f,%d,%.4f,%d,%d,%.4f,%.4f,%.4f,%d,%.4f,%.4f,%.4f,%d\n",
            f, n, m, lp_setup, lp_s8, cyc8, lp_s4, cyc4, lp_ok ? 1 : 0,
            ac_setup, ac_s8, ac_s4, ac_ok ? 1 : 0, ac2_setup, ac2_s8, ac2_s4, ac2_ok ? 1 : 0)
    flush(io)
    global done += 1
    @printf("  [%d/%d] %-45s m=%d lp=%.2f+%.2f(%d)/%.2f(%d) ac=%.2f+%.2f ac2=%.2f+%.2f\n",
            idx, length(cands), f, m, lp_setup, lp_s8, cyc8, lp_s4, cyc4, ac_setup, ac_s8, ac2_setup, ac2_s8)
    flush(stdout)
    GC.gc()
end
close(io)
println("DONE: $done graphs -> $out")
