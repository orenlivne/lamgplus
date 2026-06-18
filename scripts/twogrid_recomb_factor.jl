# twogrid_recomb_factor.jl — reproduces the caliber-1/caliber-2 "LFA", "2-level", and
# "+recomb" columns of Table A.1, all with the SOLVER's smoother: forward Gauss-Seidel
# (omega=1), the (nu_pre,nu_post)=(1,2) cycle. Self-contained (Julia stdlib only).
#   (a) forward-GS two-grid LFA factor from the Fourier symbol;
#   (b) the same factor measured on a PERIODIC grid (the LFA setting, no boundary), with the
#       production selective caliber-2 + unweighted least-squares weight, with/without
#       min-residual iterate recombination.
# The LFA and the measured periodic 2-level agree (caliber-1 0.48 vs 0.47; caliber-2 ~0.09 vs
# 0.08-0.12), validating the analysis. Run: julia scripts/twogrid_recomb_factor.jl
using SparseArrays, LinearAlgebra, Printf, Statistics, Random

# Periodic anisotropic grid (torus) — matches the LFA's infinite-grid assumption (no boundary).
function grid_per(P; wx=1.0, wy=1.0)
    idx(i,j)=(j-1)*P+i; I=Int[]; J=Int[]; V=Float64[]
    for j in 1:P, i in 1:P
        ip = i<P ? i+1 : 1; push!(I,idx(i,j)); push!(J,idx(ip,j)); push!(V,wx)
        jp = j<P ? j+1 : 1; push!(I,idx(i,j)); push!(J,idx(i,jp)); push!(V,wy)
    end
    W=sparse(I,J,V,P*P,P*P); W=W+W'; spdiagm(0=>vec(sum(W,dims=2)))-W
end
gauge(x)=x .- sum(x)/length(x)

function tvs(L; K=4, sw=8, seed=11)
    n=size(L,1); rows=rowvals(L); vals=nonzeros(L); d=diag(L); X=zeros(n,K)
    for c in 1:K
        x=gauge(randn(MersenneTwister(seed+c),n))
        for _ in 1:sw
            for i in 1:n; s=0.0; for k in nzrange(L,i); r=rows[k]; r!=i && (s-=vals[k]*x[r]); end; x[i]=s/d[i]; end
            x=gauge(x)
        end
        X[:,c]=x
    end
    X
end

function agg_clean(P)
    n=P*P; idx(i,j)=(j-1)*P+i; aggid(i,j)=(j-1)*(P÷2)+((i+1)÷2); label=zeros(Int,n)
    for j in 1:P, i in 1:P; label[idx(i,j)]=aggid(i,j); end
    nc=(P÷2)*P; seeds=zeros(Int,nc); for i in 1:n; a=label[i]; seeds[a]==0 && (seeds[a]=i); end
    label, seeds, nc
end
function strong(L,i,label; th=0.5)
    rows=rowvals(L); vals=nonzeros(L); rmax=0.0
    for k in nzrange(L,i); r=rows[k]; r!=i && (rmax=max(rmax,abs(vals[k]))); end
    S=Int[]; for k in nzrange(L,i); r=rows[k]; r==i && continue; abs(vals[k])>=th*rmax && push!(S,label[r]); end
    unique(S)
end
P1b(label,nc)=sparse(collect(1:length(label)),[label[i] for i in 1:length(label)],ones(length(label)),length(label),nc)
function P2(L,X,label,seeds,nc; delta=1e-3)
    n=size(L,1); Ip=Int[]; Jp=Int[]; Vp=Float64[]; upg=0; iss=falses(n); for a in 1:nc; iss[seeds[a]]=true; end
    for i in 1:n
        if iss[i]; push!(Ip,i); push!(Jp,label[i]); push!(Vp,1.0); continue; end
        S=strong(L,i,label)
        if length(S)==2
            A,B=S[1],S[2]; a=@view X[seeds[A],:]; b=@view X[seeds[B],:]; xi=@view X[i,:]
            dn=0.0; nu=0.0; for k in 1:size(X,2); dn+=(a[k]-b[k])^2; nu+=(a[k]-b[k])*(xi[k]-b[k]); end
            w = dn>1e-30 ? nu/dn : 1.0
            if w<0||w>1; push!(Ip,i); push!(Jp,label[i]); push!(Vp,1.0)
            elseif w<=delta; push!(Ip,i); push!(Jp,B); push!(Vp,1.0)
            elseif w>=1-delta; push!(Ip,i); push!(Jp,A); push!(Vp,1.0)
            else push!(Ip,i); push!(Jp,A); push!(Vp,w); push!(Ip,i); push!(Jp,B); push!(Vp,1-w); upg+=1; end
        else push!(Ip,i); push!(Jp,label[i]); push!(Vp,1.0); end
    end
    sparse(Ip,Jp,Vp,n,nc), upg
end
# forward GS (nu_pre,nu_post)=(1,2) two-grid factor, exact coarse, +/- min-residual recombination
function tg(L,Pr; nupre=1, nupost=2, recomb=false, kap=4, ncyc=80)
    n=size(L,1); Ac=Pr'*L*Pr; Acf=lu(Ac+1e-10*I); rows=rowvals(L); vals=nonzeros(L); d=diag(L)
    fwd!(e)=begin for i in 1:n; s=0.0; for k in nzrange(L,i); r=rows[k]; r!=i && (s-=vals[k]*e[r]); end; e[i]=s/d[i]; end; e end
    e=gauge(randn(MersenneTwister(7),n)); rs=[norm(e)]; hist=Vector{Vector{Float64}}()
    for c in 1:ncyc
        for _ in 1:nupre; fwd!(e); e.=gauge(e); end
        rc=Pr'*(-(L*e)); ec=gauge(Acf\rc); e.+=Pr*ec
        for _ in 1:nupost; fwd!(e); e.=gauge(e); end; e=gauge(e)
        if recomb && !isempty(hist); E=reduce(hcat,(h .- e for h in hist)); LE=L*E; al=LE\(-(L*e)); e.=gauge(e .+ E*al); end
        push!(hist,copy(e)); length(hist)>kap && popfirst!(hist); push!(rs,norm(e))
    end
    rr=[rs[i+1]/rs[i] for i in 1:length(rs)-1 if rs[i]>1e-11]
    isempty(rr) ? NaN : exp(mean(log.(max.(rr[max(end-6,1):end],1e-30))))
end
# forward-GS two-grid LFA factor (analytic; numeric sup over the Fourier domain), nu=nupre+nupost=3
function lfa(eps; calib=2, nu=3, n=500)
    best=0.0
    for tx in range(1e-4,pi/2;length=n), ty in range(0.0,pi;length=n)
        sx=sin(tx/2)^2; cx=cos(tx/2)^2; sy=sin(ty/2)^2; l1=4sx+4eps*sy; l2=4cx+4eps*sy
        Sa = 1 - l1/((2+2eps)-exp(-im*tx)-eps*exp(-im*ty))
        Sb = 1 - l2/((2+2eps)-exp(-im*(tx+pi))-eps*exp(-im*ty))
        q1,q2 = calib==1 ? (cx,sx) : (cx^2,sx^2); be=q1*l1/(q1*l1+q2*l2)
        rho=abs(Sa^nu*(1-be)+Sb^nu*be); rho>best && (best=rho)
    end
    best
end

P=64
@printf("%-7s | LFA c1 | LFA c2 | 2lvl c1 | 2lvl c2 | 2lvl c2+rec | up%%  (periodic, fwd-GS (1,2))\n","eps")
for eps in (1e-1, 1e-2, 1e-4)
    L=grid_per(P; wy=eps); X=tvs(L); label,seeds,nc=agg_clean(P)
    Pc1=P1b(label,nc); Pc2,upg=P2(L,X,label,seeds,nc)
    @printf("%-7.0e | %6.3f | %6.3f | %7.3f | %7.3f | %11.3f | %.0f\n",
        eps, lfa(eps;calib=1), lfa(eps;calib=2), tg(L,Pc1), tg(L,Pc2), tg(L,Pc2;recomb=true), 100*upg/(P*P-nc))
end
println("DONE")
