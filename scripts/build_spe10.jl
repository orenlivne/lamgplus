# Build graph-Laplacian adjacency matrices from the SPE10 (Tenth SPE Comparative Solution Project,
# Christie & Blunt 2001) Model 2 permeability field --- the canonical "SPE" family used as a hard
# high-contrast, anisotropic test in Gao-Kyng-Spielman 2023. We discretise the single-phase pressure
# equation -div(K grad p)=f on the 60x220x85 Cartesian reservoir grid with a cell-centred two-point
# flux approximation (TPFA): each cell is a node, each interior face an edge whose weight is the
# harmonic-mean transmissibility of the two cells' directional permeabilities, scaled by the geometric
# factor A_face/d. The result is a genuine weighted 7-point graph Laplacian (an SDDM/M-matrix), the
# same operator type LAMG+ and every competitor consume. We emit a small size ladder by taking the
# first nz z-layers. Input: SPE10MODEL2_PERM.INC (OPM mirror). Output: data/SPE__spe10_2_nz<NZ>.mtx.
#   julia scripts/build_spe10.jl [perm.INC]   (auto-downloads the public OPM mirror if absent)
using SparseArrays, Printf, Downloads
const PERM_URL = "https://raw.githubusercontent.com/OPM/opm-data/master/spe10model2/SPE10MODEL2_PERM.INC"
const NX, NY, NZ = 60, 220, 85            # SPE10 Model 2 grid
const DX, DY, DZ = 20.0, 10.0, 2.0        # cell sizes (ft): 1200x2200x170 ft over 60x220x85
const GX = DY*DZ/DX                        # face-area / distance geometric factors per direction
const GY = DX*DZ/DY
const GZ = DX*DY/DZ
harm(a,b) = (s=a+b; s <= 0 ? 0.0 : 2a*b/s) # harmonic mean (=0 if both perms vanish -> no flux)

function parse_perm(path)
    kx=Float64[]; ky=Float64[]; kz=Float64[]; cur=nothing
    sizehint!(kx, NX*NY*NZ); sizehint!(ky, NX*NY*NZ); sizehint!(kz, NX*NY*NZ)
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s,"--")) && continue
        s=="PERMX" && (cur=kx; continue)
        s=="PERMY" && (cur=ky; continue)
        s=="PERMZ" && (cur=kz; continue)
        cur===nothing && continue
        for tok in split(s)
            tok=="/" && (cur=nothing; break)
            if occursin('*',tok)                      # Eclipse run-length: N*value
                a,b=split(tok,'*'); append!(cur, fill(parse(Float64,b), parse(Int,a)))
            else
                push!(cur, parse(Float64,tok))
            end
        end
    end
    kx,ky,kz
end

function build_adj(KX,KY,KZ, nz)
    lin(i,j,k) = i + NX*(j-1) + NX*NY*(k-1)
    I=Int[]; J=Int[]; V=Float64[]
    @inline add!(a,b,w) = (w>0 && (push!(I,a);push!(J,b);push!(V,w); push!(I,b);push!(J,a);push!(V,w)))
    for k in 1:nz, j in 1:NY, i in 1:NX
        i<NX && add!(lin(i,j,k), lin(i+1,j,k), GX*harm(KX[i,j,k],KX[i+1,j,k]))   # x-faces
        j<NY && add!(lin(i,j,k), lin(i,j+1,k), GY*harm(KY[i,j,k],KY[i,j+1,k]))   # y-faces
        k<nz && add!(lin(i,j,k), lin(i,j,k+1), GZ*harm(KZ[i,j,k],KZ[i,j,k+1]))   # z-faces
    end
    n = NX*NY*nz
    sparse(I,J,V,n,n)
end

function write_mtx_sym(path, W)
    n=size(W,1); rows=rowvals(W); vals=nonzeros(W)
    cnt=0; for j in 1:n, t in nzrange(W,j); rows[t]>j && (cnt+=1); end
    open(path,"w") do io
        println(io,"%%MatrixMarket matrix coordinate real symmetric"); println(io,"$n $n $cnt")
        for j in 1:n, t in nzrange(W,j); i=rows[t]; i>j && println(io,"$i $j $(vals[t])"); end
    end
end

permpath = length(ARGS)>=1 ? ARGS[1] : joinpath(@__DIR__, "..", "data", "SPE10MODEL2_PERM.INC")
outdir   = joinpath(@__DIR__, "..", "data")
if !isfile(permpath)
    println("perm field not found; downloading SPE10 Model 2 (public domain, Christie-Blunt) ...")
    Downloads.download(PERM_URL, permpath)
end
println("parsing $permpath ...")
kx,ky,kz = parse_perm(permpath)
@printf("  PERMX=%d PERMY=%d PERMZ=%d (expected %d each)\n", length(kx),length(ky),length(kz), NX*NY*NZ)
@assert length(kx)==NX*NY*NZ && length(ky)==NX*NY*NZ && length(kz)==NX*NY*NZ "unexpected perm count"
KX=reshape(kx,NX,NY,NZ); KY=reshape(ky,NX,NY,NZ); KZ=reshape(kz,NX,NY,NZ)
@printf("  perm range: kx[%.3g,%.3g] ky[%.3g,%.3g] kz[%.3g,%.3g]  (contrast kx %.1g)\n",
        minimum(kx),maximum(kx), minimum(ky),maximum(ky), minimum(kz),maximum(kz), maximum(kx)/max(minimum(kx),1e-30))
for nz in (20, 43, 85)
    W = build_adj(KX,KY,KZ, nz)
    n=size(W,1); m=div(nnz(W),2)
    f = joinpath(outdir, "SPE__spe10_2_nz$(nz).mtx")
    write_mtx_sym(f, W)
    @printf("  nz=%2d -> n=%d m=%d  ->  %s\n", nz, n, m, f)
end
println("done")
