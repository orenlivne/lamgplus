# true_acf2.jl — asymptotic convergence factor measured properly: run to 1e-12 (above the roundoff floor),
# report log10 relative residual per cycle, and the geomean factor over the window 1e-4..1e-11 (the asymptotic
# regime), for LX and RAND RHS. Also approxChol PCG iterations to 1e-8 and 1e-12 for comparison.
using LAMG, Laplacians, SparseArrays, LinearAlgebra, Random, Statistics, Printf
include("/Users/oren/code/aier/lamgplus/scripts/mm_loader.jl")
const DATA = "/Users/oren/code/aier/data/suitesparse"
function grid_aniso2(P; wx=1.0, wy=1e-4)
    idx(i,j)=(j-1)*P+i; I=Int[];J=Int[];V=Float64[]
    for j in 1:P, i in 1:P
        i<P && (push!(I,idx(i,j));push!(J,idx(i+1,j));push!(V,wx))
        j<P && (push!(I,idx(i,j));push!(J,idx(i,j+1));push!(V,wy))
    end
    W=sparse(I,J,V,P*P,P*P); W+W'
end
gs = Any[("grid2d_512", () -> grid2(512,512)), ("aniso2d_256", () -> grid_aniso2(256)), ("grid3d_64", () -> grid3(64)),
         ("chimera_1e5", () -> chimera(100_000, 1)),
         ("delaunay_n16", "DIMACS10__delaunay_n16.mtx"), ("delaunay_n18", "DIMACS10__delaunay_n18.mtx"), ("delaunay_n20", "DIMACS10__delaunay_n20.mtx"),
         ("roadNet-PA", "SNAP__roadNet-PA.mtx"), ("com-DBLP", "SNAP__com-DBLP.mtx"), ("web-Stanford", "SNAP__web-Stanford.mtx"),
         ("bmwcra_1", "GHS_psdef__bmwcra_1.mtx"), ("cage12", "vanHeukelum__cage12.mtx"), ("thermal2", "Schmid__thermal2.mtx")]
out = open(get(ENV,"OUT","true_acf2.csv"), "w")
println(out, "graph,rhs,cycles_to_1e-8,cycles_to_1e-12,acf_window,escalated,apx_its_1e-8,apx_its_1e-12,log10_residuals")
for (nm, src) in gs
    if src isa String; W,L = reduce_to_lcc(read_mm_adj(joinpath(DATA,src))...); else; W = src(); W=(W+W')/2; L = lap(W); end
    n = size(L,1)
    xt = randn(MersenneTwister(11), n); xt .-= mean(xt); bLX = L*xt
    bR = randn(MersenneTwister(12), n); bR .-= mean(bR)
    o = LAMGOptions(tol=1e-12, max_cycles=60); h = setup(L; options=o)
    fa = approxchol_lap(W; tol=1e-12, verbose=false)
    for (kind,b) in (("LX",bLX),("RAND",bR))
        x, info = solve(h, b; options=o)
        rh = info.residual_history ./ info.residual_history[1]
        c8 = findfirst(<=(1e-8), rh); c12 = findfirst(<=(1e-12), rh)
        idx = [k for k in 2:length(rh) if 1e-11 <= rh[k] <= 1e-4 && rh[k-1] <= 1e-4]
        aw = isempty(idx) ? NaN : exp(mean(log.(rh[k]/rh[k-1]) for k in idx))
        i8=[0]; fa(b; tol=1e-8, maxits=2000, verbose=false, pcgIts=i8); i12=[0]; fa(b; tol=1e-12, maxits=2000, verbose=false, pcgIts=i12)
        lr = join([@sprintf("%.1f", log10(r)) for r in rh[2:end]], " ")
        @printf(out, "%s,%s,%s,%s,%.3f,%d,%d,%d,%s\n", nm, kind, c8===nothing ? "-" : string(c8-1), c12===nothing ? "-" : string(c12-1), aw, info.gamma_escalated ? 1 : 0, i8[1], i12[1], lr); flush(out)
        @printf("%-14s %-4s to1e-8=%s to1e-12=%s ACF(1e-4..1e-11)=%.3f esc=%d | apx its %d / %d | log10 r: %s\n", nm, kind,
                c8===nothing ? "-" : string(c8-1), c12===nothing ? "-" : string(c12-1), aw, info.gamma_escalated ? 1 : 0, i8[1], i12[1], lr)
    end
    W=nothing; L=nothing; h=nothing; fa=nothing; GC.gc()
end
close(out)
