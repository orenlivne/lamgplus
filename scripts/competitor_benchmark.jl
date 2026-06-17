#!/usr/bin/env julia
"""
Competitor benchmark for graph-Laplacian solvers.

Compares LAMG.jl against:
  - Laplacians.jl::approxchol_lap (Kyng-Sachdeva 2016)
  - CHOLMOD direct (via `cholesky` on L + ε·I)
  - pyAMG smoothed-aggregation (via PyCall, optional)
  - HYPRE BoomerAMG (optional, may DNF on some instances)

On each instance we extract the largest connected component, build the
Laplacian L = D - A, draw a random zero-sum RHS, and time setup + solve
to relative tolerance `tol` (default 1e-8).

USAGE:
    julia --project=scripts/competitor_env scripts/competitor_benchmark.jl \\
        [--instances=PATH] [--tol=1e-8] [--max-nnz=20000000] \\
        [--per-solver-timeout=300] [--out=competitor_results.csv]

Defaults to data/competitor_instances.txt for the list of .mtx files.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "competitor_env"))

# Make LAMG visible (developed into this env).
using LAMG
using LinearAlgebra
using SparseArrays
using Random
using Statistics
using Printf

# ── competitor packages ─────────────────────────────────────────────────
const HAS_LAPLACIANS = try
    @eval using Laplacians
    true
catch e
    @warn "Laplacians.jl unavailable: $e"
    false
end

const HAS_PYAMG = try
    @eval using PyCall
    @eval const _pyamg_mod = pyimport("pyamg")
    @eval const _scipy_sparse = pyimport("scipy.sparse")
    true
catch e
    @warn "pyAMG unavailable: $e"
    false
end

const HAS_HYPRE = try
    @eval using HYPRE
    @eval HYPRE.Init()
    true
catch e
    @warn "HYPRE unavailable: $e"
    false
end

# ── Matrix Market loader ────────────────────────────────────────────────
# Returns (W_adj, L) where W_adj is the adjacency (symmetric, zero diag,
# non-negative weights) and L = D - W_adj is the graph Laplacian.
# Mirrors the preprocessing in scripts/benchmark.jl::read_mm.

function read_mm_adj(path::AbstractString)
    open(path, "r") do io
        header = readline(io)
        tokens = split(lowercase(header))
        @assert tokens[1] == "%%matrixmarket" && tokens[2] == "matrix"
        format = tokens[3]
        field = tokens[4]
        symmetry = tokens[5]
        @assert format == "coordinate"
        line = readline(io)
        while startswith(strip(line), "%")
            line = readline(io)
        end
        nrows, ncols, nentries = parse.(Int, split(line))
        rows = Vector{Int}(undef, 2 * nentries)
        cols = Vector{Int}(undef, 2 * nentries)
        vals = Vector{Float64}(undef, 2 * nentries)
        k = 0
        for _ in 1:nentries
            line = readline(io)
            parts = split(line)
            i = parse(Int, parts[1])
            j = parse(Int, parts[2])
            v = if field == "pattern"
                1.0
            elseif field == "complex"
                re = parse(Float64, parts[3])
                im = parse(Float64, parts[4])
                sqrt(re * re + im * im)
            else
                parse(Float64, parts[3])
            end
            k += 1
            rows[k] = i; cols[k] = j; vals[k] = v
            if symmetry in ("symmetric", "hermitian", "skew-symmetric") && i != j
                k += 1
                rows[k] = j; cols[k] = i; vals[k] = v
            end
        end
        W_raw = sparse(rows[1:k], cols[1:k], vals[1:k], nrows, ncols)
        upper = triu(W_raw, 1)
        W = upper + sparse(transpose(upper))
        had_diagonal = any(!iszero, diag(W_raw))
        if had_diagonal
            ii, jj, _ = findnz(W)
            m_n, nn_n = size(W)
            W = sparse(ii, jj, ones(Float64, length(ii)), m_n, nn_n)
        end
        if !had_diagonal && (nnz(W) > 0) && (minimum(nonzeros(W)) < 0)
            rows_W = rowvals(W); vals_W = nonzeros(W)
            nW = size(W, 1)
            row_neg_sum = zeros(nW); row_max_abs = zeros(nW)
            for j in 1:nW
                for r in nzrange(W, j)
                    i = rows_W[r]; v = vals_W[r]
                    v < 0 && (row_neg_sum[i] += v)
                    row_max_abs[i] = max(row_max_abs[i], abs(v))
                end
            end
            need_abs = false
            for i in 1:nW
                if row_neg_sum[i] < -1e-5 * row_max_abs[i]
                    need_abs = true; break
                end
            end
            need_abs && (W = abs.(W))
        end
        # Zero diagonal.
        for j in 1:size(W, 2)
            for r in nzrange(W, j)
                if W.rowval[r] == j
                    W.nzval[r] = 0.0
                end
            end
        end
        if nnz(W) > 0
            max_w = maximum(abs, nonzeros(W))
            threshold = sqrt(eps(Float64)) * max_w
            for r in 1:nnz(W)
                abs(W.nzval[r]) < threshold && (W.nzval[r] = 0.0)
            end
        end
        dropzeros!(W)
        L = laplacian(W)
        return W, L
    end
end

# ── Connected-component reduction returns both adjacency and Laplacian.
function reduce_to_lcc(W::SparseMatrixCSC, L::SparseMatrixCSC)
    n = size(L, 1)
    label = LAMG.connected_components(L)
    M = maximum(label)
    M == 1 && return W, L
    sizes = zeros(Int, M)
    for l in label
        sizes[l] += 1
    end
    biggest = argmax(sizes)
    retained = findall(==(biggest), label)
    Wcc = W[retained, retained]
    Lcc = L[retained, retained]
    return Wcc, Lcc
end

# ── App category for reporting.
function app_category(name::AbstractString)
    n = lowercase(name)
    occursin(r"web-|cnr-|^eu-|^in-|^uk-|gleich|webbase", n) && return "web"
    occursin(r"cit-|coauth|copap|citation|com-dblp", n) && return "citation"
    occursin(r"snap__soc|^socfb|soc-|epinion|slashdot|brightkite|email|gowalla|wiki-|amazon|p2p|oregon|reuters|roget|foldoc|smallworld|preferential|^newman|net__|pajek__|gset", n) && return "social"
    occursin(r"roadnet|road|osm", n) && return "road"
    occursin(r"af_shell|naca|^m6|as365|nlr|333sp|sphere|3d_|nasa|bcsstk2[5-9]|bcsstk3[0-7]|crystk|pwt|ct20stif|olafu|raefsky|wave|wing|auto|barth|bodyy|tandem_dual|onera_dual|commanche|nopoly|pesa|inline|bone|fault|hood|pwtk|chen|nd6k|nd12k|nd24k|nd3k|engine|gridgena|fcondp|ts-palko|halfb|fullb|x104|crankseg|s3dkq4m2|cant|consph|pkustk|ramage|shipsec|stomach|torso|smt|cfd2|ldoor|af_0|af_1|af_2|af_3|af_4|af_5|af_shell|thermal|^t2dah|cubeland", n) && return "geom3d"
    occursin(r"airfoil|delaunay|3elt|jagmesh|lshp|grid[12]|big_dual|cti|cs4|t60k|m14b|netz4504|whitaker|shock|wing_nodal|fe_|garon|cavity|fidap|venturi|brack|crack|airfoil1|^l-9|^l\b|monien|walshaw|^add|2cubes_sphere|^ex|memchip", n) && return "geom2d"
    occursin(r"circuit|memplus|hcircuit|bcircuit|onetone|twotone|scircuit|coupcons|^dc|^trans", n) && return "circuit"
    occursin(r"cage|ghs_indef|dtoc|laser|bates|optim|gset|sdp|lp_|qp_|^c-|^cqd|^pds|^cz", n) && return "optimization"
    return "other"
end

# ── Solver wrappers.
# Each wrapper returns (setup_us, solve_us, ok::Bool, residual_rel::Float64).

# LAMG.
function bench_lamg(L::SparseMatrixCSC, b::Vector{Float64}, tol::Float64, repeats::Int)
    opts = LAMGOptions(tol = tol, max_cycles = 100)
    # Warm-up.
    try
        h0 = setup(L; options = opts)
        _, _ = solve(h0, b; options = opts)
    catch e
        return (NaN, NaN, false, NaN, 0)
    end
    setup_ts = Float64[]; solve_ts = Float64[]; cycles = 0
    last_x = nothing
    for _ in 1:repeats
        local h, info, x
        ts = @elapsed h = setup(L; options = opts)
        push!(setup_ts, ts)
        ts2 = @elapsed (x, info) = solve(h, b; options = opts)
        push!(solve_ts, ts2)
        cycles = info.cycles
        last_x = x
    end
    rel = norm(L * last_x - b) / max(norm(b), 1e-30)
    return (median(setup_ts) * 1e6, median(solve_ts) * 1e6,
            rel <= tol * 10, rel, cycles)
end

# approxchol_lap.
function bench_approxchol(W::SparseMatrixCSC, b::Vector{Float64}, tol::Float64, repeats::Int)
    HAS_LAPLACIANS || return (NaN, NaN, false, NaN, 0)
    # The closure built by `approxchol_lap(W; tol=...)` runs PCG inside on
    # each call, so we time setup separately by re-running the factor
    # function. We use `approxchol_lap` to get the operator factory:
    # call it with a no-iter call to capture pure setup, then solve.
    try
        # Warm-up.
        f0 = Laplacians.approxchol_lap(W; tol = tol, verbose = false)
        _ = f0(b)
    catch e
        return (NaN, NaN, false, NaN, 0)
    end
    setup_ts = Float64[]; solve_ts = Float64[]; iters = 0
    last_x = nothing
    for _ in 1:repeats
        local f, x
        ts = @elapsed f = Laplacians.approxchol_lap(W; tol = tol, verbose = false)
        push!(setup_ts, ts)
        # Use pcgIts so we know how many PCG iters it took.
        pcgIts = [0]
        ts2 = @elapsed x = f(b; tol = tol, verbose = false, pcgIts = pcgIts)
        push!(solve_ts, ts2)
        iters = pcgIts[1]
        last_x = x
    end
    # Build L for residual check.
    L = LAMG.laplacian(W)
    rel = norm(L * last_x - b) / max(norm(b), 1e-30)
    return (median(setup_ts) * 1e6, median(solve_ts) * 1e6,
            rel <= tol * 100, rel, iters)
end

# Direct Cholesky on the SPSD Laplacian. Need to regularize: add ε·I and
# project out the constant vector (zero-sum b makes that null direction
# benign). Use `cholesky` from SuiteSparse.
function bench_cholesky(L::SparseMatrixCSC, b::Vector{Float64}, tol::Float64, repeats::Int)
    n = size(L, 1)
    # Regularization that preserves SPD without much spectral shift.
    # We pin one node to zero by deleting that row/col instead (Dirichlet) —
    # cleaner than tiny diagonal ε which can blow conditioning.
    keep = 2:n
    A = L[keep, keep]
    bcut = b[keep]
    try
        # Warm-up.
        F0 = cholesky(A)
        _ = F0 \ bcut
    catch e
        return (NaN, NaN, false, NaN, 0)
    end
    setup_ts = Float64[]; solve_ts = Float64[]
    last_x = zeros(n)
    for _ in 1:repeats
        local F, y
        ts = @elapsed F = cholesky(A)
        push!(setup_ts, ts)
        ts2 = @elapsed y = F \ bcut
        push!(solve_ts, ts2)
        last_x = vcat(0.0, y)
    end
    # Center for fair residual check.
    last_x .-= mean(last_x)
    rel = norm(L * last_x - b) / max(norm(b), 1e-30)
    return (median(setup_ts) * 1e6, median(solve_ts) * 1e6,
            rel <= tol * 100, rel, 1)
end

# pyAMG smoothed aggregation. We hand it the Laplacian + a regularization
# diagonal to make it SPD.
function bench_pyamg(L::SparseMatrixCSC, b::Vector{Float64}, tol::Float64, repeats::Int)
    HAS_PYAMG || return (NaN, NaN, false, NaN, 0)
    n = size(L, 1)
    try
        # Convert to scipy CSR.
        # Build CSR triple.
        Lcsr_rows = Int32[]; Lcsr_cols = Int32[]; Lcsr_vals = Float64[]
        # Build with a Dirichlet pin at node 1 (drop row/col 1) so we have SPD.
        # That matches the Cholesky baseline.
        keep = 2:n
        A = L[keep, keep]
        bcut = b[keep]
        # Convert A to scipy CSC (PyCall handles SparseMatrixCSC).
        Apy = PyCall.PyObject(A)
        Apy_csr = _scipy_sparse.csr_matrix(Apy)
        bpy = PyCall.PyObject(bcut)
        # Warm-up.
        ml0 = _pyamg_mod.smoothed_aggregation_solver(Apy_csr)
        _ = ml0.solve(bpy, tol = tol)
        setup_ts = Float64[]; solve_ts = Float64[]
        last_x = nothing
        iters = 0
        for _ in 1:repeats
            local ml, x
            ts = @elapsed ml = _pyamg_mod.smoothed_aggregation_solver(Apy_csr)
            push!(setup_ts, ts)
            residuals = PyCall.PyObject([])
            ts2 = @elapsed x = ml.solve(bpy, tol = tol, residuals = residuals)
            push!(solve_ts, ts2)
            last_x = convert(Vector{Float64}, x)
            iters = length(residuals)
        end
        # Lift back to full-size vector, center.
        xfull = vcat(0.0, last_x)
        xfull .-= mean(xfull)
        rel = norm(L * xfull - b) / max(norm(b), 1e-30)
        return (median(setup_ts) * 1e6, median(solve_ts) * 1e6,
                rel <= tol * 100, rel, iters)
    catch e
        @warn "pyAMG failed: $(sprint(showerror, e))"
        return (NaN, NaN, false, NaN, 0)
    end
end

# HYPRE BoomerAMG.
function bench_hypre(L::SparseMatrixCSC, b::Vector{Float64}, tol::Float64, repeats::Int)
    HAS_HYPRE || return (NaN, NaN, false, NaN, 0)
    n = size(L, 1)
    keep = 2:n
    A = L[keep, keep]
    bcut = b[keep]
    try
        # Warm-up.
        H0 = HYPRE.HYPREMatrix(A)
        bH0 = HYPRE.HYPREVector(bcut)
        xH0 = HYPRE.HYPREVector(zeros(length(bcut)))
        amg0 = HYPRE.BoomerAMG(; Tol = tol, MaxIter = 200)
        HYPRE.solve!(amg0, xH0, H0, bH0)
        setup_ts = Float64[]; solve_ts = Float64[]
        last_x = nothing
        for _ in 1:repeats
            local AH, bH, xH, amg
            ts = @elapsed begin
                AH = HYPRE.HYPREMatrix(A)
                bH = HYPRE.HYPREVector(bcut)
                xH = HYPRE.HYPREVector(zeros(length(bcut)))
                amg = HYPRE.BoomerAMG(; Tol = tol, MaxIter = 200)
            end
            push!(setup_ts, ts)
            ts2 = @elapsed HYPRE.solve!(amg, xH, AH, bH)
            push!(solve_ts, ts2)
            last_x = copy(HYPRE.copy!(zeros(length(bcut)), xH))
        end
        xfull = vcat(0.0, last_x)
        xfull .-= mean(xfull)
        rel = norm(L * xfull - b) / max(norm(b), 1e-30)
        return (median(setup_ts) * 1e6, median(solve_ts) * 1e6,
                rel <= tol * 100, rel, 0)
    catch e
        @warn "HYPRE failed: $(sprint(showerror, e))"
        return (NaN, NaN, false, NaN, 0)
    end
end

# ── Per-instance driver.
struct InstanceResult
    instance::String
    category::String
    n::Int
    m::Int
    # LAMG
    lamg_setup_us::Float64; lamg_solve_us::Float64; lamg_total_us::Float64
    lamg_per_edge_us::Float64; lamg_ok::Bool; lamg_cycles::Int; lamg_rel::Float64
    # approxchol
    apx_setup_us::Float64; apx_solve_us::Float64; apx_total_us::Float64
    apx_per_edge_us::Float64; apx_ok::Bool; apx_iters::Int; apx_rel::Float64
    # cholesky
    chol_setup_us::Float64; chol_solve_us::Float64; chol_total_us::Float64
    chol_per_edge_us::Float64; chol_ok::Bool; chol_rel::Float64
    # pyamg
    pya_setup_us::Float64; pya_solve_us::Float64; pya_total_us::Float64
    pya_per_edge_us::Float64; pya_ok::Bool; pya_iters::Int; pya_rel::Float64
    # hypre
    hyp_setup_us::Float64; hyp_solve_us::Float64; hyp_total_us::Float64
    hyp_per_edge_us::Float64; hyp_ok::Bool; hyp_rel::Float64
    winner::String
end

function run_instance(path::String; tol::Float64 = 1e-8, repeats::Int = 3,
                      max_nnz::Int = 20_000_000,
                      enable_pyamg::Bool = true,
                      enable_hypre::Bool = true,
                      enable_chol::Bool = true)
    name = basename(path)
    cat = app_category(name)
    # Load (may be expensive for big files).
    @info "── $name"
    W, L = try
        read_mm_adj(path)
    catch e
        @warn "load failed $name: $e"
        return nothing
    end
    # Extract LCC.
    W, L = reduce_to_lcc(W, L)
    n = size(L, 1)
    nnzL = nnz(L)
    m = div(nnzL - n, 2)
    if nnzL > max_nnz
        @warn "$name: nnz $nnzL exceeds cap $max_nnz, skipping."
        return nothing
    end
    # RHS.
    rng = MersenneTwister(0xff + n)
    x_true = randn(rng, n); x_true .-= sum(x_true) / n
    b = L * x_true

    # --- LAMG.
    lamg_su, lamg_so, lamg_ok, lamg_rel, lamg_cy = bench_lamg(L, b, tol, repeats)
    # --- approxchol_lap.
    apx_su, apx_so, apx_ok, apx_rel, apx_it = bench_approxchol(W, b, tol, repeats)
    # --- Cholesky (cap at n=300k or nnz<6M to be safe with memory).
    chol_su = NaN; chol_so = NaN; chol_ok = false; chol_rel = NaN
    if enable_chol && n <= 500_000 && nnzL <= 6_000_000
        chol_su, chol_so, chol_ok, chol_rel, _ = bench_cholesky(L, b, tol, repeats)
    end
    # --- pyAMG (skip on very big instances).
    pya_su = NaN; pya_so = NaN; pya_ok = false; pya_rel = NaN; pya_it = 0
    if enable_pyamg && n <= 1_000_000 && nnzL <= 15_000_000
        pya_su, pya_so, pya_ok, pya_rel, pya_it = bench_pyamg(L, b, tol, repeats)
    end
    # --- HYPRE.
    hyp_su = NaN; hyp_so = NaN; hyp_ok = false; hyp_rel = NaN
    if enable_hypre && n <= 1_000_000 && nnzL <= 15_000_000
        hyp_su, hyp_so, hyp_ok, hyp_rel, _ = bench_hypre(L, b, tol, repeats)
    end

    # Totals.
    lamg_tot = lamg_su + lamg_so
    apx_tot = apx_su + apx_so
    chol_tot = chol_su + chol_so
    pya_tot = pya_su + pya_so
    hyp_tot = hyp_su + hyp_so
    pe(x) = isnan(x) ? NaN : (x / m)

    # Winner among solvers that converged.
    candidates = Tuple{String, Float64}[]
    lamg_ok && push!(candidates, ("lamg", lamg_tot))
    apx_ok && push!(candidates, ("approxchol", apx_tot))
    chol_ok && push!(candidates, ("cholesky", chol_tot))
    pya_ok && push!(candidates, ("pyamg", pya_tot))
    hyp_ok && push!(candidates, ("hypre", hyp_tot))
    winner = if isempty(candidates)
        "none"
    else
        _, idx = findmin(c -> c[2], candidates)
        candidates[idx][1]
    end

    return InstanceResult(name, cat, n, m,
        lamg_su, lamg_so, lamg_tot, pe(lamg_tot), lamg_ok, lamg_cy, lamg_rel,
        apx_su, apx_so, apx_tot, pe(apx_tot), apx_ok, apx_it, apx_rel,
        chol_su, chol_so, chol_tot, pe(chol_tot), chol_ok, chol_rel,
        pya_su, pya_so, pya_tot, pe(pya_tot), pya_ok, pya_it, pya_rel,
        hyp_su, hyp_so, hyp_tot, pe(hyp_tot), hyp_ok, hyp_rel,
        winner)
end

# ── CSV writer.
const CSV_HEADER = "instance,category,n,m," *
    "lamg_setup_us,lamg_solve_us,lamg_total_us,lamg_per_edge_us,lamg_ok,lamg_cycles,lamg_rel," *
    "approxchol_setup_us,approxchol_solve_us,approxchol_total_us,approxchol_per_edge_us,approxchol_ok,approxchol_iters,approxchol_rel," *
    "cholesky_setup_us,cholesky_solve_us,cholesky_total_us,cholesky_per_edge_us,cholesky_ok,cholesky_rel," *
    "pyamg_setup_us,pyamg_solve_us,pyamg_total_us,pyamg_per_edge_us,pyamg_ok,pyamg_iters,pyamg_rel," *
    "hypre_setup_us,hypre_solve_us,hypre_total_us,hypre_per_edge_us,hypre_ok,hypre_rel," *
    "winner"

function write_row(io::IO, r::InstanceResult)
    @printf(io, "%s,%s,%d,%d,", r.instance, r.category, r.n, r.m)
    @printf(io, "%.3f,%.3f,%.3f,%.6f,%d,%d,%.3e,",
            r.lamg_setup_us, r.lamg_solve_us, r.lamg_total_us, r.lamg_per_edge_us,
            r.lamg_ok ? 1 : 0, r.lamg_cycles, r.lamg_rel)
    @printf(io, "%.3f,%.3f,%.3f,%.6f,%d,%d,%.3e,",
            r.apx_setup_us, r.apx_solve_us, r.apx_total_us, r.apx_per_edge_us,
            r.apx_ok ? 1 : 0, r.apx_iters, r.apx_rel)
    @printf(io, "%.3f,%.3f,%.3f,%.6f,%d,%.3e,",
            r.chol_setup_us, r.chol_solve_us, r.chol_total_us, r.chol_per_edge_us,
            r.chol_ok ? 1 : 0, r.chol_rel)
    @printf(io, "%.3f,%.3f,%.3f,%.6f,%d,%d,%.3e,",
            r.pya_setup_us, r.pya_solve_us, r.pya_total_us, r.pya_per_edge_us,
            r.pya_ok ? 1 : 0, r.pya_iters, r.pya_rel)
    @printf(io, "%.3f,%.3f,%.3f,%.6f,%d,%.3e,",
            r.hyp_setup_us, r.hyp_solve_us, r.hyp_total_us, r.hyp_per_edge_us,
            r.hyp_ok ? 1 : 0, r.hyp_rel)
    @printf(io, "%s\n", r.winner)
end

# ── Main.
function main()
    instances_path = joinpath(@__DIR__, "..", "data", "competitor_instances.txt")
    out_csv = joinpath(@__DIR__, "..", "competitor_results.csv")
    tol = 1e-8
    max_nnz = 20_000_000
    repeats = 3
    enable_pyamg = HAS_PYAMG
    enable_hypre = HAS_HYPRE
    enable_chol = true
    start_idx = 1
    end_idx = typemax(Int)
    for a in ARGS
        if startswith(a, "--instances=")
            instances_path = split(a, "=", limit = 2)[2]
        elseif startswith(a, "--tol=")
            tol = parse(Float64, split(a, "=")[2])
        elseif startswith(a, "--max-nnz=")
            max_nnz = parse(Int, split(a, "=")[2])
        elseif startswith(a, "--out=")
            out_csv = split(a, "=", limit = 2)[2]
        elseif startswith(a, "--repeats=")
            repeats = parse(Int, split(a, "=")[2])
        elseif a == "--no-pyamg"
            enable_pyamg = false
        elseif a == "--no-hypre"
            enable_hypre = false
        elseif a == "--no-chol"
            enable_chol = false
        elseif startswith(a, "--start=")
            start_idx = parse(Int, split(a, "=")[2])
        elseif startswith(a, "--end=")
            end_idx = parse(Int, split(a, "=")[2])
        end
    end
    data_dir = joinpath(@__DIR__, "..", "data")
    names = readlines(instances_path)
    names = [strip(n) for n in names if !isempty(strip(n)) && !startswith(strip(n), "#")]
    end_idx = min(end_idx, length(names))
    println("Running $((end_idx - start_idx + 1)) instances ($start_idx..$end_idx of $(length(names))) from $instances_path")
    println("Solvers enabled: lamg=true approxchol=$HAS_LAPLACIANS chol=$enable_chol pyamg=$enable_pyamg hypre=$enable_hypre")
    println("Output: $out_csv")
    # Append mode if file exists and start > 1.
    mode = (start_idx == 1 || !isfile(out_csv)) ? "w" : "a"
    open(out_csv, mode) do io
        if mode == "w"
            println(io, CSV_HEADER)
        end
        flush(io)
        for i in start_idx:end_idx
            name = names[i]
            path = joinpath(data_dir, name)
            isfile(path) || (@warn "missing $path"; continue)
            res = try
                run_instance(path; tol = tol, repeats = repeats,
                             max_nnz = max_nnz,
                             enable_pyamg = enable_pyamg,
                             enable_hypre = enable_hypre,
                             enable_chol = enable_chol)
            catch e
                @warn "$name: instance crashed: $(sprint(showerror, e))"
                nothing
            end
            if res !== nothing
                write_row(io, res)
                flush(io)
                @printf("  [%d/%d] %-50s  n=%d m=%d  lamg=%.0fμs apx=%.0fμs chol=%.0fμs winner=%s\n",
                        i, end_idx, res.instance, res.n, res.m,
                        res.lamg_total_us, res.apx_total_us, res.chol_total_us,
                        res.winner)
            end
            GC.gc()
        end
    end
    println("Done. Results: $out_csv")
end

main()
