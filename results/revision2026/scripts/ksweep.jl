# ksweep.jl — Referee-1 point 2 / Brandt falsifiers: (a) does the affinity inversion on the anisotropic grid
# disappear with more test vectors K, WITHOUT the SOC veto? (b) seed variance of LAMG+ ACF with the default config.
using LAMG, Laplacians, SparseArrays, LinearAlgebra, Random, Statistics, Printf
function grid_aniso2(P; wx=1.0, wy=1e-4)
    idx(i,j)=(j-1)*P+i; I=Int[];J=Int[];V=Float64[]
    for j in 1:P, i in 1:P
        i<P && (push!(I,idx(i,j));push!(J,idx(i+1,j));push!(V,wx))
        j<P && (push!(I,idx(i,j));push!(J,idx(i,j+1));push!(V,wy))
    end
    W=sparse(I,J,V,P*P,P*P); W+W'
end
acf(cf) = isempty(cf) ? NaN : exp(mean(log.(cf[max(1,end-4):end])))
println("=== (a) K-sweep on 256^2 aniso grid eps=1e-4, veto OFF vs ON, caliber-2 on; 3 seeds each ===")
L = lap(grid_aniso2(256; wy=1e-4)); n = size(L,1)
for K in (4, 8, 16, 32), veto in (0.0, 0.05)
    res = Float64[]; cyc = Int[]
    for seed in 1:3
        Random.seed!(seed)
        o = LAMGOptions(tol=1e-8, max_cycles=60, agg_K=K, agg_soc_τ=veto, caliber2_1d=true)
        b = L*(randn(MersenneTwister(seed), n) |> v -> v .- mean(v))
        x, info = solve(L, b; options=o)
        push!(res, acf(info.conv_factors)); push!(cyc, info.cycles)
    end
    @printf("K=%2d veto=%s  ACF per seed: %s   cycles: %s\n", K, veto>0 ? "ON " : "OFF", join([@sprintf("%.3f",r) for r in res], " "), join(cyc, " "))
end
println("=== (b) seed variance (default config K=4, veto on) on 128^2 aniso, 128^2 iso grid, and a 3D 32^3 grid: 8 seeds ===")
for (nm, W) in (("aniso128", grid_aniso2(128; wy=1e-4)), ("iso128", grid2(128,128)), ("grid3d_32", grid3(32)))
    L = lap(W); n = size(L,1); res = Float64[]; cyc = Int[]
    for seed in 1:8
        Random.seed!(seed)
        b = L*(randn(MersenneTwister(100+seed), n) |> v -> v .- mean(v))
        x, info = solve(L, b; options=LAMGOptions(tol=1e-8, max_cycles=60))
        push!(res, acf(info.conv_factors)); push!(cyc, info.cycles)
    end
    @printf("%-10s ACF min/median/max = %.4f / %.4f / %.4f   cycles %s\n", nm, minimum(res), median(res), maximum(res), join(cyc," "))
end
