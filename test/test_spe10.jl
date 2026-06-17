# Validate that the SPE10 TPFA construction (scripts/build_spe10.jl) yields a TRUE graph Laplacian:
# a symmetric, non-negatively weighted adjacency whose Laplacian L = D - W is symmetric PSD with the
# constant null vector and non-positive off-diagonals. We test the construction recipe on a tiny
# random permeability field (with exact dense spectrum), and -- if the full artifact has been built --
# the actual data/SPE__spe10_2_nz20.mtx on disk (with sparse-safe checks only).
using Test, SparseArrays, LinearAlgebra, Random

# Tiny cell-centred TPFA build, identical recipe to scripts/build_spe10.jl (harmonic-mean directional
# transmissibility x geometric factor) on an nx*ny*nz grid with given per-cell permeabilities.
_harm(a,b) = (s=a+b; s <= 0 ? 0.0 : 2a*b/s)
function tpfa_adj(KX,KY,KZ, nx,ny,nz; gx=1.0, gy=4.0, gz=100.0)
    lin(i,j,k) = i + nx*(j-1) + nx*ny*(k-1)
    I=Int[]; J=Int[]; V=Float64[]
    add!(a,b,w) = (w>0 && (push!(I,a);push!(J,b);push!(V,w); push!(I,b);push!(J,a);push!(V,w)))
    for k in 1:nz, j in 1:ny, i in 1:nx
        i<nx && add!(lin(i,j,k), lin(i+1,j,k), gx*_harm(KX[i,j,k],KX[i+1,j,k]))
        j<ny && add!(lin(i,j,k), lin(i,j+1,k), gy*_harm(KY[i,j,k],KY[i,j+1,k]))
        k<nz && add!(lin(i,j,k), lin(i,j,k+1), gz*_harm(KZ[i,j,k],KZ[i,j,k+1]))
    end
    sparse(I,J,V, nx*ny*nz, nx*ny*nz)
end
_lap(W) = spdiagm(0 => vec(sum(W, dims=2))) - W

# Sparse-safe Laplacian identities (no dense allocation): valid on graphs of any size.
function check_laplacian_sparse(W)
    n = size(W,1)
    @test W == permutedims(W)                         # symmetric adjacency
    @test isempty(nonzeros(W)) || minimum(nonzeros(W)) ≥ 0   # non-negative edge weights
    @test all(iszero, diag(W))                         # no self-loops
    L = _lap(W)
    @test L ≈ permutedims(L)                           # symmetric Laplacian
    @test norm(L * ones(n)) < 1e-8 * sqrt(n)           # constant vector in null space (zero row sums)
    rows = rowvals(L); vals = nonzeros(L); maxoff = -Inf
    for j in 1:n, k in nzrange(L,j); i=rows[k]; i!=j && (maxoff = max(maxoff, vals[k])); end
    @test maxoff ≤ 1e-9                                # off-diagonals ≤ 0 (M-matrix / SDDM)
    @test minimum(diag(L)) ≥ -1e-9                     # non-negative diagonal
    rng = MersenneTwister(11)                          # PSD via the energy quadratic form x'Lx ≥ 0
    for _ in 1:5; x = randn(rng, n); @test dot(x, L*x) ≥ -1e-6 * dot(x,x); end
    L
end

@testset "SPE10 TPFA build is a true graph Laplacian" begin
    rng = MersenneTwister(7)
    nx,ny,nz = 4,5,3
    KX = 10.0 .^ (7 .* rand(rng, nx,ny,nz) .- 4)      # high-contrast (~7-decade) positive perms
    KY = 10.0 .^ (7 .* rand(rng, nx,ny,nz) .- 4)
    KZ = 10.0 .^ (7 .* rand(rng, nx,ny,nz) .- 4)
    W = tpfa_adj(KX,KY,KZ, nx,ny,nz)
    @test size(W,1) == nx*ny*nz
    L = check_laplacian_sparse(W)
    ev = eigvals(Symmetric(Matrix(L)))                # tiny => exact spectrum
    @test ev[1] > -1e-8                                # PSD
    @test ev[2] > 1e-10                                # connected field => simple zero eigenvalue

    # The actual built artifact, if present (data/SPE__spe10_2_nz20.mtx from scripts/build_spe10.jl).
    f = joinpath(@__DIR__, "..", "data", "SPE__spe10_2_nz20.mtx")
    if isfile(f)
        I=Int[]; J=Int[]; V=Float64[]; n=0
        open(f) do io
            readline(io); l=readline(io); while startswith(strip(l),"%"); l=readline(io); end
            n = parse(Int, split(l)[1])
            for ln in eachline(io)
                p=split(ln); isempty(p) && continue
                i=parse(Int,p[1]); j=parse(Int,p[2]); v=length(p)≥3 ? parse(Float64,p[3]) : 1.0
                push!(I,i);push!(J,j);push!(V,v); i!=j && (push!(I,j);push!(J,i);push!(V,v))
            end
        end
        Wreal = sparse(I,J,V,n,n)
        @test size(Wreal,1) == 60*220*20
        check_laplacian_sparse(Wreal)                 # sparse-safe checks on the real 264k-node matrix
    end
end
