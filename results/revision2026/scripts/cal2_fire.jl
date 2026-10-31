# cal2_fire.jl — Brandt falsifier: does caliber-2 fire on locally-1-D graphs other than anisotropic grids (road
# networks, meshes), and does it help there? Also a diagonal (45°) anisotropy boundary check.
# For each graph: fraction of fine rows of each aggregation level's P with 2 parents; ACF/cycles/time with and
# without caliber-2 (veto on in both).
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
# strong coupling along the (1,1) diagonal, weak along x and y: anisotropy not aligned with the grid axes
function grid_diag2(P; ws=1.0, ww=1e-4)
    idx(i,j)=(j-1)*P+i; I=Int[];J=Int[];V=Float64[]
    for j in 1:P, i in 1:P
        i<P && (push!(I,idx(i,j));push!(J,idx(i+1,j));push!(V,ww))
        j<P && (push!(I,idx(i,j));push!(J,idx(i,j+1));push!(V,ww))
        (i<P && j<P) && (push!(I,idx(i,j));push!(J,idx(i+1,j+1));push!(V,ws))
    end
    W=sparse(I,J,V,P*P,P*P); W+W'
end
acf(cf) = isempty(cf) ? NaN : exp(mean(log.(cf[max(1,end-4):end])))
function two_parent_fraction(h)
    tot=0; two=0; per=String[]
    for l in h.levels
        l.p === nothing && continue; l.level_type == :agg || continue
        Pt = sparse(l.p')  # columns = fine rows
        cnt = diff(Pt.colptr); t2 = count(==(2), cnt); tot += length(cnt); two += t2
        push!(per, @sprintf("%.1f%%", 100t2/length(cnt)))
    end
    (tot==0 ? 0.0 : two/tot, join(per, "/"))
end
graphs = Any[("aniso256 (control)", () -> grid_aniso2(256)), ("iso256 (control)", () -> grid2(256,256)),
             ("diag-aniso256 (45°)", () -> grid_diag2(256)),
             ("luxembourg_osm", () -> read_mm_adj(joinpath(DATA,"DIMACS10__luxembourg_osm.mtx"))),
             ("roadNet-PA", () -> read_mm_adj(joinpath(DATA,"SNAP__roadNet-PA.mtx"))),
             ("delaunay_n17", () -> read_mm_adj(joinpath(DATA,"DIMACS10__delaunay_n17.mtx"))),
             ("fe_ocean", () -> read_mm_adj(joinpath(DATA,"DIMACS10__fe_ocean.mtx"))),
             ("thermal1", () -> read_mm_adj(joinpath(DATA,"Schmid__thermal1.mtx")))]
for (nm, build) in graphs
    r = build(); if r isa Tuple; W,L = reduce_to_lcc(r...); else; L = lap((r+r')/2); end
    n = size(L,1); b = L*(randn(MersenneTwister(3), n) |> v -> v .- mean(v))
    for c2 in (true, false)
        o = LAMGOptions(tol=1e-8, max_cycles=100, caliber2_1d=c2)
        solve(L, b; options=o)   # warm
        ts = @elapsed h = setup(L; options=o); x, info = solve(h, b; options=o)
        fr, per = two_parent_fraction(h)
        @printf("%-22s cal2=%-5s 2-parent rows: %5.1f%% (per agg level %s)  cyc=%3d ACF=%.3f setup=%.2fs solve=%.2fs\n",
                nm, c2, 100fr, per, info.cycles, acf(info.conv_factors), ts, info.solve_time)
    end
end
