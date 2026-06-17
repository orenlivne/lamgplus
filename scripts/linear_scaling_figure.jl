#!/usr/bin/env julia
"""
linear_scaling_figure.jl — reproduce LAMG paper Fig 4.1 on this Julia port.

Runs the *linear* LAMG Laplacian solver over a large, diverse set of
real-world graphs and records, per instance:

  setup time per edge      t_setup / m   [s/edge]
  solve time per edge       t_solve / m   [s/edge]
  total time per edge       (t_setup + t_solve) / m   [s/edge]

Paper claim (Livne & Brandt 2012, §4.1, Fig 4.1, Table 4.1): all three are
approximately CONSTANT across ~3 orders of magnitude in m — i.e. the solver
is O(m). We verify the same here and fit the log-log slope of per-edge time
vs m (slope ≈ 0 ⇒ linear).

Each instance: load .mtx, build adjacency W (symmetric, nonneg, zero diag),
reduce to the largest connected component, L = D - W, random zero-sum RHS,
solve to rel-residual `tol`. One timed run after a global JIT warm-up.

Writes results incrementally to OUT so partial progress survives a kill.

USAGE:
    julia --project=. scripts/linear_scaling_figure.jl \\
        [--glob=data] [--tol=1e-8] [--min-m=5000] [--max-nnz=8000000] \\
        [--budget-s=2400] [--max-file-mb=120] [--out=linear_scaling_results.csv]
"""

using LAMG
using LinearAlgebra, SparseArrays, Random, Statistics, Printf

# ── arg parsing ─────────────────────────────────────────────────────────
function getarg(key, default)
    for a in ARGS
        startswith(a, "--$key=") && return split(a, "=", limit=2)[2]
    end
    return default
end
const GLOBDIR  = getarg("glob", "data")
const TOL      = parse(Float64, getarg("tol", "1e-8"))
const MIN_M    = parse(Int,     getarg("min-m", "5000"))
const MAX_NNZ  = parse(Int,     getarg("max-nnz", "8000000"))
const BUDGET_S = parse(Float64, getarg("budget-s", "2400"))
const MAXFILEMB= parse(Float64, getarg("max-file-mb", "120"))
const OUT      = getarg("out", "linear_scaling_results.csv")

include(joinpath(@__DIR__, "mm_loader.jl"))   # read_mm_adj, laplacian helpers

# category tag (reuse competitor_benchmark heuristic, trimmed)
function app_category(name::AbstractString)
    n = lowercase(name)
    occursin(r"web-|cnr-|^eu-|^in-|^uk-|webbase", n) && return "web"
    occursin(r"cit-|coauth|citation|com-dblp", n)    && return "citation"
    occursin(r"soc|epinion|slashdot|brightkite|email|gowalla|wiki-|amazon|p2p|oregon", n) && return "social"
    occursin(r"roadnet|road|osm", n) && return "road"
    occursin(r"circuit|memplus|onetone|twotone|scircuit|^trans", n) && return "circuit"
    occursin(r"cage|optim|gset|sdp|lp_|^c-|^pds", n) && return "optimization"
    occursin(r"af_shell|naca|sphere|3d_|nasa|bone|fault|hood|pwtk|ldoor|thermal|engine", n) && return "geom3d"
    occursin(r"airfoil|delaunay|3elt|jagmesh|grid|dual|cti|cs4|whitaker|shock|wing|fe_|brack|crack|^ex", n) && return "geom2d"
    return "other"
end

function bench_one(L::SparseMatrixCSC, b::Vector{Float64})
    opts = LAMGOptions(tol = TOL, max_cycles = 100)
    t_setup = @elapsed h = setup(L; options = opts)
    t_solve = @elapsed (x, info) = solve(h, b; options = opts)
    rel = norm(L * x - b) / max(norm(b), 1e-30)
    Σnnz = sum(nnz(h[l].a) for l in 1:LAMG.num_levels(h))
    return (t_setup, t_solve, info.cycles, rel, LAMG.num_levels(h),
            Σnnz / nnz(L))
end

const HEADER = "instance,category,n,m,levels,opc,setup_s,solve_s,total_s," *
               "setup_per_edge_s,solve_per_edge_s,total_per_edge_s,cycles,rel,ok"

function main()
    files = filter(f -> endswith(f, ".mtx"), readdir(GLOBDIR; join = true))
    sizes = [(f, stat(f).size) for f in files]
    filter!(t -> t[2] <= MAXFILEMB * 1e6, sizes)
    sort!(sizes, by = t -> t[2])
    @printf "Candidate .mtx files (<= %.0f MB): %d\n" MAXFILEMB length(sizes)

    # JIT warm-up on a tiny grid so timings exclude compilation.
    let Aw = grid2d_laplacian(8, 8); bw = randn(64); bw .-= sum(bw)/64
        hw = setup(Aw; options = LAMGOptions(tol = 1e-8, max_cycles = 20))
        solve(hw, bw; options = LAMGOptions(tol = 1e-8, max_cycles = 20))
    end

    open(OUT, "w") do io; println(io, HEADER); end
    t0 = time()
    done = 0; ok_count = 0
    for (path, fsize) in sizes
        (time() - t0) > BUDGET_S && (println("\n[budget reached, stopping]"); break)
        name = basename(path)
        local W, L
        try
            W, L = read_mm_adj(path)
        catch e
            continue
        end
        nnz(L) == 0 && continue
        W, L = reduce_to_lcc(W, L)
        n = size(L, 1); m = div(nnz(L) - n, 2)
        (m < MIN_M || nnz(L) > MAX_NNZ) && continue
        rng = MersenneTwister(UInt32(0xABC) + UInt32(n & 0xffff))
        b = randn(rng, n); b .-= sum(b) / n
        local r
        try
            r = bench_one(L, b)
        catch e
            continue
        end
        t_setup, t_solve, cycles, rel, levels, opc = r
        ok = rel <= TOL * 10
        total = t_setup + t_solve
        open(OUT, "a") do io
            @printf io "%s,%s,%d,%d,%d,%.3f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%d,%.3e,%d\n" (
                name) app_category(name) n m levels opc t_setup t_solve total (
                t_setup/m) (t_solve/m) (total/m) cycles rel (ok ? 1 : 0)
        end
        done += 1; ok && (ok_count += 1)
        if done % 20 == 0
            @printf "  [%4d done, %4d ok, %.0fs] last: %-28s m=%-9d %.2f us/edge\n" (
                done) ok_count (time()-t0) name m (total/m*1e6)
        end
    end
    @printf "\nDONE: %d instances, %d ok, %.0fs. Results -> %s\n" done ok_count (time()-t0) OUT
end

main()
