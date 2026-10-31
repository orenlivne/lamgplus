# rhs_study.jl — Referee-1 point 3: does the choice of RHS favour LAMG+?
# For each system, solve with three RHS types to rel-residual 1e-8:
#   LX   : b = L*x_true, x_true ~ N(0,1) zero-mean      (paper + GKS's own choice: M*g normalised)
#   RAND : b ~ N(0,1), projected to zero mean           (referee's suggested "just remove the null space")
#   DIP  : b = e_s - e_t, s,t random                    (s-t dipole: long-wavelength, hard for MG)
# Solvers: LAMG+ (cycles, ACF = geomean of last 5 per-cycle factors, solve s), approxChol & AC (PCG its, solve s).
using LAMG, Laplacians, SparseArrays, LinearAlgebra, Random, Statistics, Printf, DelimitedFiles
include("/Users/oren/code/aier/lamgplus/scripts/mm_loader.jl")
const TOL = 1e-8
const DATA = "/Users/oren/code/aier/data/suitesparse"
const GKS  = "/Users/oren/code/aier/lamgplus/results/gks_cmp"
out = get(ENV, "OUT", "rhs_study.csv")

function read_ijv_lap(path)
    d = readdlm(path, ','); A = sparse(Int.(d[:,1]), Int.(d[:,2]), Float64.(d[:,3]))
    A = (A + A') ./ 2
    W = -A; W = W - spdiagm(0 => diag(W)); dropzeros!(W)
    for k in 1:nnz(W); W.nzval[k] < 0 && (W.nzval[k] = 0.0); end; dropzeros!(W)
    W, lap(W)
end
function load_mtx(name)
    W, L = read_mm_adj(joinpath(DATA, name)); W, L = reduce_to_lcc(W, L); W, L
end
systems = Any[
    ("grid2_aniso (GKS)",  () -> read_ijv_lap(joinpath(GKS, "grid2_aniso_A.txt"))),
    ("grid2_uniform (GKS)",() -> read_ijv_lap(joinpath(GKS, "grid2_uniform_A.txt"))),
    ("grid3_hicon (GKS)",  () -> read_ijv_lap(joinpath(GKS, "grid3_hicon_A.txt"))),
    ("chimera (GKS)",      () -> read_ijv_lap(joinpath(GKS, "chimera_A.txt"))),
    ("wted_chimera (GKS)", () -> read_ijv_lap(joinpath(GKS, "wted_chimera_A.txt"))),
    ("star (GKS)",         () -> read_ijv_lap(joinpath(GKS, "star_A.txt"))),
    ("spe10 (GKS)",        () -> read_ijv_lap(joinpath(GKS, "spe10_A.txt"))),
    ("delaunay_n18",       () -> load_mtx("DIMACS10__delaunay_n18.mtx")),
    ("cage12",             () -> load_mtx("vanHeukelum__cage12.mtx")),
    ("com-DBLP",           () -> load_mtx("SNAP__com-DBLP.mtx")),
    ("roadNet-PA",         () -> load_mtx("SNAP__roadNet-PA.mtx")),
    ("web-Stanford",       () -> load_mtx("SNAP__web-Stanford.mtx")),
    ("bmwcra_1",           () -> load_mtx("GHS_psdef__bmwcra_1.mtx")),
    ("pwtk",               () -> load_mtx("Boeing__pwtk.mtx")),
]
function rhs(kind, L, rng)
    n = size(L,1)
    if kind == "LX";   x = randn(rng, n); x .-= mean(x); return L*x
    elseif kind == "RAND"; b = randn(rng, n); b .-= mean(b); return b
    else; s, t = randperm(rng, n)[1:2]; b = zeros(n); b[s] = 1; b[t] = -1; return b; end
end
acf(cf) = isempty(cf) ? NaN : exp(mean(log.(cf[max(1,end-4):end])))
relres(L,x,b) = norm(L*x - b)/norm(b)

io = open(out, "w")
println(io, "system,n,m,rhs,seed,lp_cycles,lp_acf,lp_solve_s,lp_relres,apx_its,apx_solve_s,apx_relres,ac_its,ac_solve_s,ac_relres")
opts = LAMGOptions(tol=TOL, max_cycles=200)
for (name, build) in systems
    W, L = build(); n = size(L,1); m = div(nnz(L)-n, 2)
    h = setup(L; options=opts)
    fa = approxchol_lap(W; tol=TOL, verbose=false)
    fc = approxchol_lap2(W; tol=TOL, verbose=false)
    # warm-up all three
    b0 = rhs("LX", L, MersenneTwister(1)); solve(h, b0; options=opts); fa(b0; tol=TOL, maxits=1000, verbose=false); fc(b0; tol=TOL, maxits=1000, verbose=false)
    for kind in ("LX","RAND","DIP"), seed in 1:3
        b = rhs(kind, L, MersenneTwister(100*seed + 7))
        x, info = solve(h, b; options=opts)
        ita=[0]; ta = @elapsed xa = fa(b; tol=TOL, maxits=1000, verbose=false, pcgIts=ita)
        itc=[0]; tc = @elapsed xc = fc(b; tol=TOL, maxits=1000, verbose=false, pcgIts=itc)
        @printf(io, "%s,%d,%d,%s,%d,%d,%.4f,%.4f,%.2e,%d,%.4f,%.2e,%d,%.4f,%.2e\n", name, n, m, kind, seed,
                info.cycles, acf(info.conv_factors), info.solve_time, relres(L,x,b), ita[1], ta, relres(L,xa,b), itc[1], tc, relres(L,xc,b))
        flush(io)
        @printf("%-20s %-4s s%d  LAMG+ cyc=%3d acf=%.3f %.3fs | apx its=%3d %.3fs | AC its=%3d %.3fs\n", name, kind, seed, info.cycles, acf(info.conv_factors), info.solve_time, ita[1], ta, itc[1], tc)
    end
end
close(io)
