# Fidelity spot-check: DRA-QC (our reimplementation of Napov–Notay 2017) vs LAMG+
# on real graphs, same (L,b). Validates the paper's reported pattern (DRA-QC takes
# ~2–3× fewer iterations than LAMG and is faster). Self-contained loader (no
# Laplacians dependency); run with the base project:
#   julia --project=. scripts/draqc_fidelity_spotcheck.jl data/GHS_psdef__bmwcra_1.mtx ...
using LAMG, LinearAlgebra, SparseArrays, Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "draqc", "src", "DRAQC.jl"))
using .DRAQC

function read_mm_adj(path)
    open(path, "r") do io
        header = readline(io); tok = split(lowercase(header))
        field = tok[4]; sym = tok[5]
        line = readline(io); while startswith(strip(line), "%"); line = readline(io); end
        nr, nc, ne = parse.(Int, split(line))
        rows = Vector{Int}(undef, 2ne); cols = similar(rows); vals = Vector{Float64}(undef, 2ne); k = 0
        for _ in 1:ne
            p = split(readline(io)); i = parse(Int, p[1]); j = parse(Int, p[2])
            v = field == "pattern" ? 1.0 : parse(Float64, p[3])
            k += 1; rows[k] = i; cols[k] = j; vals[k] = v
            if sym in ("symmetric", "hermitian", "skew-symmetric") && i != j
                k += 1; rows[k] = j; cols[k] = i; vals[k] = v
            end
        end
        Wr = sparse(rows[1:k], cols[1:k], vals[1:k], nr, nc)
        upper = triu(Wr, 1); W = upper + sparse(transpose(upper))
        if any(!iszero, diag(Wr))
            ii, jj, _ = findnz(W); W = sparse(ii, jj, ones(length(ii)), size(W)...)
        end
        nnz(W) > 0 && minimum(nonzeros(W)) < 0 && (W = abs.(W))
        for j in 1:size(W, 2), r in nzrange(W, j); W.rowval[r] == j && (W.nzval[r] = 0.0); end
        dropzeros!(W)
        return W, laplacian(W)
    end
end
function lcc(L)
    label = LAMG.connected_components(L); M = maximum(label); M == 1 && return L
    sizes = zeros(Int, M); for l in label; sizes[l] += 1; end
    keep = findall(==(argmax(sizes)), label); return L[keep, keep]
end

paths = ARGS
tol = 1e-8
@printf("%-22s %8s %10s | %-22s | %-26s\n", "graph", "n", "m", "LAMG+  (cyc, s)", "DRA-QC (iters, s, μ-cyc)")
println("-"^96)
for path in paths
    _, L0 = read_mm_adj(path); L = lcc(L0)
    n = size(L, 1); m = (nnz(L) - n) ÷ 2
    Random.seed!(1); xt = randn(n); xt .-= sum(xt)/n; b = L * xt

    # LAMG+
    opts = LAMGOptions(tol = tol, max_cycles = 100)
    h = setup(L; options = opts); solve(h, b; options = opts)               # warm-up
    ts = @elapsed h = setup(L; options = opts)
    tso = @elapsed ((x, info) = solve(h, b; options = opts))
    lrel = norm(L*x - b)/norm(b); lcyc = info.cycles

    # DRA-QC
    hd = DRAQC.draqc_setup(L); s = DRAQC.DRAQCSolver(hd)                    # warm-up
    DRAQC.draqc_solve(s, b; tol = tol, maxiter = 300)
    td = @elapsed (hd = DRAQC.draqc_setup(L); s = DRAQC.DRAQCSolver(hd))
    tdo = @elapsed ((xd, infod) = DRAQC.draqc_solve(s, b; tol = tol, maxiter = 300))
    drel = norm(L*xd - b)/norm(b)
    # iterations to 1e-6 (the paper's robustness metric)
    _, info6 = DRAQC.draqc_solve(s, b; tol = 1e-6, maxiter = 300)
    oc = DRAQC.operator_complexity(hd); nlev = DRAQC.num_levels(hd)

    @printf("%-22s %8d %10d | cyc=%-3d %5.2f+%5.2fs | it=%-3d(@1e-6=%d) %6.2f+%5.2fs OC=%.2f L=%d rel=%.1e\n",
        basename(path), n, m, lcyc, ts, tso, infod.iters, info6.iters, td, tdo, oc, nlev, drel)
end
