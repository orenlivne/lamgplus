# diag.jl — where does the slow asymptotic factor come from on web-Stanford / delaunay_n18 (RAND rhs)?
using LAMG, SparseArrays, LinearAlgebra, Random, Statistics, Printf
include("/Users/oren/code/aier/lamgplus/scripts/mm_loader.jl")
const DATA = "/Users/oren/code/aier/data/suitesparse"
acfw(rh) = (idx=[k for k in 2:length(rh) if 1e-11 <= rh[k] <= 1e-3 && rh[k-1] <= 1e-3]; isempty(idx) ? NaN : exp(mean(log.(rh[k]/rh[k-1]) for k in idx)))
function run(h, b, o; label="")
    x, info = solve(h, b; options=o)
    rh = info.residual_history ./ info.residual_history[1]
    @printf("  %-34s cyc=%3d acf=%.3f esc=%d  last5=%s\n", label, info.cycles, acfw(rh), info.gamma_escalated,
            join([@sprintf("%.2f",c) for c in info.conv_factors[max(1,end-4):end]], " "))
    info
end
function hier(h)
    for (i,l) in enumerate(h.levels)
        n = size(l.a,1); τ = i>1 ? n/size(h.levels[i-1].a,1) : 1.0
        @printf("   L%-2d %-12s n=%9d nnz=%10d nnz/n=%6.1f τ=%.2f%s\n", i, l.level_type, n, nnz(l.a), nnz(l.a)/n, τ,
                l.level_type==:elimination ? " stages=$(length(l.elim_stages))" : "")
    end
end
names = split(get(ENV,"GRAPHS","SNAP__web-Stanford.mtx DIMACS10__delaunay_n17.mtx DIMACS10__delaunay_n18.mtx"))
for f in names
    W,L = reduce_to_lcc(read_mm_adj(joinpath(DATA,f))...); n=size(L,1)
    println("\n=== $f n=$n nnz=$(nnz(L))")
    bR = randn(MersenneTwister(12), n); bR .-= mean(bR)
    xt = randn(MersenneTwister(11), n); xt .-= mean(xt); bLX = L*xt
    base = LAMGOptions(tol=1e-11, max_cycles=150)
    h = setup(L; options=base); hier(h)
    println(" γ_vec = ", round.(LAMG._build_gamma_vec(base.γ, base.γ_coarse, base.γ_coarse_growth, length(h), h); digits=2))
    run(h, bLX, base; label="default LX")
    run(h, bR, base; label="default RAND")
    run(h, bR, LAMGOptions(tol=1e-11, max_cycles=150, γ=1.5, γ_coarse=1.5); label="γ=1.5 RAND")
    run(h, bR, LAMGOptions(tol=1e-11, max_cycles=150, γ=2.0, γ_coarse=2.0); label="γ=2 (W) RAND")
    run(h, bR, LAMGOptions(tol=1e-11, max_cycles=150, ν_pre=2, ν_post=2); label="ν=(2,2) RAND")
    run(h, bR, LAMGOptions(tol=1e-11, max_cycles=150, history_size=8); label="history 8 RAND")
    run(h, bR, LAMGOptions(tol=1e-11, max_cycles=150, rhs_correction=4/3); label="flat 4/3 corr RAND")
    run(h, bR, LAMGOptions(tol=1e-11, max_cycles=150, do_recomb=false); label="no recomb RAND")
    LAMG._MINRES_NO_COARSE[] = true; run(h, bR, base; label="finest-only recomb RAND"); LAMG._MINRES_NO_COARSE[] = false
    # level truncation: first k levels, direct solve at level k
    println(" level-truncation (RAND, default cycle):")
    for k in 2:length(h)
        size(h.levels[k].a,1) > 60000 && continue   # direct solve too big
        hk = LAMG.Multilevel(h.levels[1:k], h.perm)
        run(hk, bR, base; label="  levels 1..$k (n_k=$(size(h.levels[k].a,1)))")
    end
    # setup variants
    for (lab, o) in (("K=8", LAMGOptions(tol=1e-11, max_cycles=150, agg_K=8)),
                     ("K=16", LAMGOptions(tol=1e-11, max_cycles=150, agg_K=16)),
                     ("Q=1.5", LAMGOptions(tol=1e-11, max_cycles=150, agg_Q=1.5)),
                     ("maxagg=4", LAMGOptions(tol=1e-11, max_cycles=150, agg_max_size=4)),
                     ("no cal2", LAMGOptions(tol=1e-11, max_cycles=150, caliber2_1d=false)),
                     ("no veto", LAMGOptions(tol=1e-11, max_cycles=150, agg_soc_τ=0.0)),
                     ("no hub iso", LAMGOptions(tol=1e-11, max_cycles=150, agg_hub_threshold=0.0)),
                     ("no reorder", LAMGOptions(tol=1e-11, max_cycles=150, reorder=false)))
        h2 = setup(L; options=o); print("  [setup $lab: $(length(h2)) levels, cx=$(round(sum(nnz(l.a) for l in h2.levels)/nnz(L);digits=2))]\n")
        run(h2, bR, o; label="setup $lab RAND")
    end
end
