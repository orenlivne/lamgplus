# Same-code A/B: how much faster is the LAMG+ algorithm than the original LAMG it re-derives?
# Both run in THIS optimized Julia implementation (so cache/allocation optimizations cancel out and
# only the ALGORITHMIC delta is measured, no language confound). Baseline LAMG = the original
# configuration: K=8 test vectors, no strength-of-connection veto, caliber-1 only (the setup.jl
# comment calls {agg_K=8, agg_soc_τ=0, caliber2_1d=false} the bit-exact MATLAB-LAMG comparison).
# LAMG+ = defaults (lean K=4, SoC veto, selective caliber-2). Reports per-graph and geometric-mean
# setup+solve speedup over a class-stratified set (real SuiteSparse/SNAP + GKS synthetic families,
# incl. the grid-aligned-anisotropic grids where the caliber-2 refinement rescues convergence).
#   julia -t1 scripts/ab_lamg_vs_lamgplus.jl
const ROOT = normpath(joinpath(@__DIR__, ".."))
import Pkg; Pkg.activate(joinpath(ROOT, "scripts", "competitor_env"))
using LAMG, LinearAlgebra, SparseArrays, Random, Printf, Statistics
import Laplacians; const Lap = Laplacians
include(joinpath(ROOT, "scripts", "mm_loader.jl"))
const TOL = 1e-8; const MAXIT = 300
BASE() = LAMGOptions(agg_K=8, agg_soc_τ=0.0, caliber2_1d=false, tol=TOL, max_cycles=MAXIT)  # original LAMG
PLUS() = LAMGOptions(tol=TOL, max_cycles=MAXIT)                                              # LAMG+

function timeit(L, b, o)
    h = setup(L; options=o); solve(h, b; options=o)                       # warm
    t = @elapsed h2 = setup(L; options=o); sol = solve(h2, b; options=o)
    ts = @elapsed solve(h2, b; options=o)
    x = sol[1]; (t + ts, norm(L*x .- b)/max(norm(b),1e-30))
end
function one(name, cls, W, L)
    n = size(L,1); rng = MersenneTwister(0xab+n); xt = randn(rng, n); xt .-= sum(xt)/n; b = L*xt
    tb, rb = timeit(L, b, BASE()); tp, rp = timeit(L, b, PLUS())
    okb = rb ≤ TOL*50; okp = rp ≤ TOL*50; sp = tb/tp
    @printf("%-26s %-12s LAMG %6.3fs%s  LAMG+ %6.3fs%s  speedup %5.2fx\n",
            name, cls, tb, okb ? " " : "*", tp, okp ? " " : "*", sp)
    (cls, sp, okb, okp)
end

# Representative real graphs (one per class) + synthetic families.
reals = [("HB__bcsstk29.mtx","FE/structural"), ("Chen__pkustk07.mtx","FE/structural"),
         ("DIMACS10__rgg_n_2_16_s0.mtx","mesh/grid"), ("SNAP__soc-sign-Slashdot081106.mtx","social"),
         ("Williams__webbase-1M.mtx","web"), ("Gleich__usroads.mtx","road")]
res = Tuple{String,Float64,Bool,Bool}[]
for (f,cls) in reals
    p = joinpath(ROOT,"data",f); isfile(p) || (@warn "missing $f"; continue)
    W,L = try; reduce_to_lcc(read_mm_adj(p)...); catch e; (@warn "load $f" e; continue); end
    push!(res, one(f, cls, W, L))
end
# Synthetic: chimera, uniform 3-D grid, and grid-aligned anisotropic grids (the refinement target).
let W=Lap.chimera(200000,1); push!(res, one("chimera-1","chimera", W, LAMG.laplacian(W))); end
let W=Lap.grid3(80,80,40);   push!(res, one("grid3-80x80x40","grid-3D", W, LAMG.laplacian(W))); end
for (N,ε) in [(256,1e-4),(450,1e-4)]
    W=Lap.grid2(N,N; isotropy=ε); push!(res, one("aniso-$(N)x$N-$ε","aniso-grid", W, LAMG.laplacian(W)))
end

sps = [s for (_,s,okb,okp) in res if okp]            # geomean over graphs LAMG+ solved
gm  = exp(mean(log.(sps)))
@printf("\nGEOMEAN speedup (LAMG / LAMG+) over %d graphs = %.2fx   [range %.2f .. %.2f]\n",
        length(sps), gm, minimum(sps), maximum(sps))
for cls in unique(c for (c,_,_,_) in res)
    v=[s for (c,s,_,okp) in res if c==cls && okp]; isempty(v) && continue
    @printf("   %-14s geomean %.2fx (n=%d)\n", cls, exp(mean(log.(v))), length(v))
end
