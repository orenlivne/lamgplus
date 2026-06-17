# GATE: re-verify on TODAY's LAMG+ that the original FE/structural win (Table 4.2 of
# lamg_plus.tex) still holds. One high-degree structural matrix, setup+solve to 1e-8,
# median of REPEATS, vs the fast approxchol_lap AND the robust approxchol_lap2 — the two
# competitors in the paper. Reports total time, us/nnz, and the LAMG+ speedup over each.
# Sanity check only: if LAMG+ is no longer faster than the fast variant, something
# regressed and we fall back to the simpler LAMG+ of that era.
#   julia -t1 --project=scripts/competitor_env scripts/gate_fe_check.jl [matrix.mtx]
import Pkg
Pkg.activate(joinpath(@__DIR__, "competitor_env"))
using LAMG, LinearAlgebra, SparseArrays, Statistics, Printf, Random
import Laplacians
include(joinpath(@__DIR__, "mm_loader.jl"))
const DATA = joinpath(@__DIR__, "..", "data")

const FILE = length(ARGS) >= 1 ? ARGS[1] : "GHS_psdef__bmwcra_1.mtx"
const TOL  = 1e-8
const REPEATS = 3

med(v) = median(v)
function timeit(f, repeats)
    f()  # warm-up (JIT)
    ts = Float64[]
    for _ in 1:repeats; push!(ts, @elapsed f()); end
    med(ts)
end

path = joinpath(DATA, FILE)
isfile(path) || error("missing $path")
@printf("Loading %s ...\n", FILE); flush(stdout)
W, L = read_mm_adj(path)
W, L = reduce_to_lcc(W, L)
n = size(L, 1); nzL = nnz(L)
Random.seed!(1)
xt = randn(n); xt .-= mean(xt)
b = L * xt; b .-= mean(b)
@printf("n=%d  nnz(L)=%d  avg deg=%.1f\n\n", n, nzL, nzL/n)

# LAMG+ (current code), setup + solve, to TOL.
opts = LAMGOptions(tol=TOL, max_cycles=100)
hwar = setup(L; options=opts); xw,_ = solve(hwar, b; options=opts)  # warm
lp_setup = timeit(() -> setup(L; options=opts), REPEATS)
h = setup(L; options=opts)
local lp_x, lp_info
lp_solve = timeit(() -> ((lp_x, lp_info) = solve(h, b; options=opts)), REPEATS)
(lp_x, lp_info) = solve(h, b; options=opts)
lp_rel = norm(L*lp_x - b)/norm(b)
lp_tot = lp_setup + lp_solve

# fast approxchol_lap, setup + solve.
ac_setup = timeit(() -> Laplacians.approxchol_lap(W; tol=TOL, verbose=false), REPEATS)
fac = Laplacians.approxchol_lap(W; tol=TOL, verbose=false)
acpcg = [0]
ac_solve = timeit(() -> fac(b; tol=TOL, verbose=false, pcgIts=acpcg), REPEATS)
ac_x = fac(b; tol=TOL, verbose=false, pcgIts=acpcg)
ac_rel = norm(L*ac_x - b)/norm(b)
ac_tot = ac_setup + ac_solve

# robust approxchol_lap2, setup + solve.
ac2_setup = timeit(() -> Laplacians.approxchol_lap2(W; tol=TOL, verbose=false), REPEATS)
fac2 = Laplacians.approxchol_lap2(W; tol=TOL, verbose=false)
ac2pcg = [0]
ac2_solve = timeit(() -> fac2(b; tol=TOL, verbose=false, pcgIts=ac2pcg), REPEATS)
ac2_x = fac2(b; tol=TOL, verbose=false, pcgIts=ac2pcg)
ac2_rel = norm(L*ac2_x - b)/norm(b)
ac2_tot = ac2_setup + ac2_solve

us(t) = t*1e6/nzL
@printf("%-22s %8s %8s %8s %9s %6s %s\n","solver","setup s","solve s","total s","us/nnz","iters","rel")
@printf("%-22s %8.3f %8.3f %8.3f %9.3f %6d %.1e\n","LAMG+ (current)",lp_setup,lp_solve,lp_tot,us(lp_tot),lp_info.cycles,lp_rel)
@printf("%-22s %8.3f %8.3f %8.3f %9.3f %6d %.1e\n","approxchol_lap (fast)",ac_setup,ac_solve,ac_tot,us(ac_tot),acpcg[1],ac_rel)
@printf("%-22s %8.3f %8.3f %8.3f %9.3f %6d %.1e\n","approxchol_lap2 (robust)",ac2_setup,ac2_solve,ac2_tot,us(ac2_tot),ac2pcg[1],ac2_rel)
println()
@printf("LAMG+ speedup vs fast   approxchol_lap : %.2fx  (paper Table 4.2 reported ~2.5x on bmwcra_1)\n", ac_tot/lp_tot)
@printf("LAMG+ speedup vs robust approxchol_lap2: %.2fx  (paper reported ~7.7x)\n", ac2_tot/lp_tot)
println(ac_tot/lp_tot > 1.0 ? "GATE PASS: LAMG+ still faster than the fast variant." :
                              "GATE FAIL: LAMG+ no longer faster — investigate / fall back.")
