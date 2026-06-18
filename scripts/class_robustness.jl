# Is DRA-QC robust across ALL graph classes (like LAMG+ and AC), or are there
# classes where it fails / blows up — the same robustness test the paper applies to
# PETSc/hypre/BoomerAMG/CMG? We measure, per class, work-per-edge (operation count)
# for LAMG+ (taken as the robust reference) and our faithful DRA-QC reimplementation,
# and flag convergence + order-of-magnitude work outliers.
#
# Run under the comparison env (has Laplacians.jl for the GKS chimera families):
#   julia --project=examples/repro_env scripts/class_robustness.jl
using Laplacians
include(joinpath(@__DIR__, "work_per_edge.jl"))   # reuses lamg_work/draqc_work/loaders (driver guarded)

# ---- class instance builders ----
lap_from_W(W) = (W = (W + W')/2; for k in 1:size(W,2), r in nzrange(W,k); W.rowval[r]==k && (W.nzval[r]=0.0); end;
                 dropzeros!(W); sparse(Diagonal(vec(sum(W;dims=2)))) - W)
function grid3d_iso(m)
    lin(i,j,k)=i+m*(j-1)+m*m*(k-1); I=Int[];J=Int[];V=Float64[]
    for k in 1:m,j in 1:m,i in 1:m
        i<m&&(push!(I,lin(i,j,k));push!(J,lin(i+1,j,k));push!(V,1.0))
        j<m&&(push!(I,lin(i,j,k));push!(J,lin(i,j+1,k));push!(V,1.0))
        k<m&&(push!(I,lin(i,j,k));push!(J,lin(i,j,k+1));push!(V,1.0)); end
    W=sparse(I,J,V,m^3,m^3); lap_from_W(W)
end
function ba_lap(n, m0; seed=1)   # Barabási–Albert scale-free (social/citation proxy)
    rng=MersenneTwister(seed); edges=Set{Tuple{Int,Int}}(); rep=Int[]
    for i in 1:m0, j in i+1:m0; push!(edges,(i,j)); push!(rep,i); push!(rep,j); end
    for v in m0+1:n
        chosen=Set{Int}()
        while length(chosen) < min(m0, v-1); push!(chosen, rep[rand(rng,1:length(rep))]); end
        for t in chosen; push!(edges,(min(v,t),max(v,t))); push!(rep,v); push!(rep,t); end
    end
    I=Int[];J=Int[]; for (a,b) in edges; push!(I,a);push!(J,b);push!(I,b);push!(J,a); end
    lap_from_W(sparse(I,J,ones(length(I)),n,n))
end

dd = joinpath(@__DIR__,"..","data")
realgraph(name) = (p=joinpath(dd,name*".mtx"); isfile(p) ? lcc(read_mm_adj(p)) : nothing)

# (class label, builder). Representative instance per class (real where shipped, else generated).
specs = Any[
    ("FE (bmwcra)",            ()->realgraph("GHS_psdef__bmwcra_1")),
    ("structural (pwtk)",      ()->realgraph("Boeing__pwtk")),
    ("web (web-Stanford)",     ()->realgraph("SNAP__web-Stanford")),
    ("SPE10 (reservoir)",      ()->realgraph("SPE__spe10_2_nz20")),
    ("social/scale-free (BA)", ()->ba_lap(50_000, 5)),
    ("mesh/iso-grid 256²",     ()->aniso2d(256,256,1.0)),
    ("grid-3D 40³",            ()->grid3d_iso(40)),
    ("star 50k",               ()->(W=spzeros(50_001,50_001); for j in 2:50_001; W[1,j]=1.0;W[j,1]=1.0; end; lap_from_W(W))),
    ("chimera",                ()->lap_from_W(chimera(50_000, 35))),
    ("wtd-chimera",            ()->lap_from_W(wted_chimera(50_000, 1))),
    ("aniso-grid ε=1e-4",      ()->aniso2d(128,128,1e-4)),
    ("hi-contrast grid",       ()->hicontrast2d(128,128)),
]

tol=1e-8; OUTLIER=3.0
@printf("%-24s %9s | %-17s | %-21s | %s\n","class","n","LAMG+ cyc OC WPE","DRA-QC it OC WPE","verdict")
println("-"^96)
results=Tuple{String,Float64,Bool}[]
for (label, build) in specs
    L = build()
    L === nothing && (println(rpad(label,24), "   (no local instance — skipped)"); continue)
    n=size(L,1); Random.seed!(1); xt=randn(n); xt.-=sum(xt)/n; b=L*xt
    lw=lamg_work(L,b,tol); dw=draqc_work(L,b,tol)
    ratio = dw.wpe/lw.wpe
    robust = dw.ok && ratio < OUTLIER
    verdict = !dw.ok ? "✗ NON-CONV" : (ratio < OUTLIER ? "robust ($(round(ratio,digits=1))×)" : "OUTLIER $(round(ratio,digits=1))× more work")
    push!(results,(label,ratio,robust))
    @printf("%-24s %9d | %3d %4.2f %7.0f | %4d %4.2f %8.0f | %s\n",
        label,n, lw.cyc,lw.oc,lw.wpe, dw.it,dw.oc,dw.wpe, verdict)
end
nbad = count(r->!r[3], results)
println("\nDRA-QC robust on $(length(results)-nbad)/$(length(results)) classes tested; ",
        "outliers/failures: ", join([r[1] for r in results if !r[3]], ", "))
println("(robust = converges to 1e-8 AND work-per-edge within $(OUTLIER)× of LAMG+. ",
        "A higher constant on easy classes is generic-method overhead, not non-robustness.)")
