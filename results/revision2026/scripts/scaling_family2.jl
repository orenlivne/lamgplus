# scaling_family.jl — Referee-1 point 6: show O(m) the standard way, on fixed families of growing size.
# Families: 2D uniform grid, 2D grid-aligned anisotropic grid (eps=1e-4), 3D uniform grid, chimera(n),
#           SuiteSparse delaunay_n10..n20, rgg_n_2_15..19, road networks (PA/TX/CA).
# Per instance: n, m, levels, operator complexity, cycles, ACF, setup s, solve s, per-edge µs. Warm timed run.
using LAMG, Laplacians, SparseArrays, LinearAlgebra, Random, Statistics, Printf
include("/Users/oren/code/aier/lamgplus/scripts/mm_loader.jl")
const TOL = 1e-8
const DATA = "/Users/oren/code/aier/data/suitesparse"
out = get(ENV, "OUT", "scaling_family.csv")
maxm = parse(Int, get(ENV, "MAXM", "20000000"))

function grid_aniso2(P; wx=1.0, wy=1e-4)
    idx(i,j)=(j-1)*P+i; I=Int[];J=Int[];V=Float64[]
    for j in 1:P, i in 1:P
        i<P && (push!(I,idx(i,j));push!(J,idx(i+1,j));push!(V,wx))
        j<P && (push!(I,idx(i,j));push!(J,idx(i,j+1));push!(V,wy))
    end
    W=sparse(I,J,V,P*P,P*P); W+W'
end
acf(cf) = isempty(cf) ? NaN : exp(mean(log.(cf[max(1,end-4):end])))
opc(h) = sum(nnz(l.a) for l in h.levels) / nnz(h.levels[1].a)

fams = Any[]
for P in (128, 256, 512, 1024, 2048); push!(fams, ("grid2d", "grid2d_$(P)", () -> grid2(P, P))); end
for P in (128, 256, 512, 1024, 2048); push!(fams, ("aniso2d", "aniso2d_$(P)", () -> grid_aniso2(P; wy=1e-4))); end
for P in (16, 32, 64, 128); push!(fams, ("grid3d", "grid3d_$(P)", () -> grid3(P))); end
for n in (10_000, 100_000, 1_000_000); for i in 1:2; push!(fams, ("chimera", "chimera_$(n)_$(i)", () -> chimera(n, i))); end; end
for k in 10:20; push!(fams, ("delaunay", "delaunay_n$k", () -> read_mm_adj(joinpath(DATA, "DIMACS10__delaunay_n$k.mtx")))); end
for k in 15:19; push!(fams, ("rgg", "rgg_n_2_$k", () -> read_mm_adj(joinpath(DATA, "DIMACS10__rgg_n_2_$(k)_s0.mtx")))); end
for s in 2:4; push!(fams, ("delaunay", "delaunay_n18_seed$s", () -> (Random.seed!(s); read_mm_adj(joinpath(DATA, "DIMACS10__delaunay_n18.mtx"))))); end
for nm in ("SNAP__roadNet-PA.mtx","SNAP__roadNet-TX.mtx","SNAP__roadNet-CA.mtx","DIMACS10__great-britain_osm.mtx","DIMACS10__germany_osm.mtx")
    push!(fams, ("road", replace(nm, ".mtx"=>""), () -> read_mm_adj(joinpath(DATA, nm))))
end

io = open(out, "w")
println(io, "family,instance,n,m,levels,opc,cycles,acf,setup_s,solve_s,setup_per_edge_us,solve_per_edge_us,total_per_edge_us,relres,apx_setup_s,apx_solve_s,apx_its,apx_total_per_edge_us")
opts = LAMGOptions(tol=TOL, max_cycles=200)
# JIT warm-up
let W = grid2(64,64); L = lap(W); b = L*randn(size(L,1)); solve(L, b; options=opts); end
for (fam, name, build) in fams
    local W, L
    r = try build() catch e; println("skip $name: $e"); continue; end
    if r isa Tuple; W, L = r; W, L = reduce_to_lcc(W, L)
    else; W = (r + r') ./ 2; L = lap(W); end
    n = size(L,1); m = div(nnz(L)-n, 2)
    m > maxm && (println("skip $name (m=$m > MAXM)"); continue)
    rng = MersenneTwister(2026 + (occursin("seed",name) ? parse(Int, name[end]) : 0)); x = randn(rng, n); x .-= mean(x); b = L*x
    ts = @elapsed h = setup(L; options=opts)
    xx, info = solve(h, b; options=opts)
    rel = norm(L*xx - b)/norm(b)
    f0 = approxchol_lap(W; tol=TOL, verbose=false); f0(b)
    ta = @elapsed f = approxchol_lap(W; tol=TOL, verbose=false); its=[0]; tb = @elapsed f(b; tol=TOL, maxits=1000, verbose=false, pcgIts=its)
    @printf(io, "%s,%s,%d,%d,%d,%.3f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2e,%.4f,%.4f,%d,%.4f\n", fam, name, n, m, length(h.levels), opc(h),
            info.cycles, acf(info.conv_factors), ts, info.solve_time, 1e6*ts/m, 1e6*info.solve_time/m, 1e6*(ts+info.solve_time)/m, rel, ta, tb, its[1], 1e6*(ta+tb)/m)
    @printf("         approxChol setup=%.2fs solve=%.2fs its=%d  %.2f us/edge total\n", ta, tb, its[1], 1e6*(ta+tb)/m)
    flush(io)
    @printf("%-8s %-28s n=%9d m=%10d lev=%2d opc=%.2f cyc=%3d acf=%.3f setup=%.2fs solve=%.2fs  %.2f/%.2f us/edge\n",
            fam, name, n, m, length(h.levels), opc(h), info.cycles, acf(info.conv_factors), ts, info.solve_time, 1e6*ts/m, 1e6*info.solve_time/m)
    W = nothing; L = nothing; h = nothing; GC.gc()
end
close(io)
