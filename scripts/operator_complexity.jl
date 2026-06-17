const ROOT = normpath(joinpath(@__DIR__, ".."))
import Pkg; Pkg.activate(joinpath(ROOT,"scripts","competitor_env"))
using LAMG, LinearAlgebra, SparseArrays, Random
import Laplacians; const Lap=Laplacians
include(joinpath(ROOT,"scripts","mm_loader.jl"))
const TOL=1e-8
using HYPRE; HYPRE.Init()
using PETSc; const plib=PETSc.petsclibs[1]; PETSc.initialize(plib); const PCOMM=PETSc.LibPETSc.PETSC_COMM_SELF
for g in ["DNVS__troll.mtx","Boeing__pwtk.mtx","GHS_psdef__bmwcra_1.mtx","Oberwolfach__bone010.mtx"]
    W,L=reduce_to_lcc(read_mm_adj(joinpath(ROOT,"data",g))...); n=size(L,1)
    xt=randn(MersenneTwister(1),n); xt.-=sum(xt)/n; b=L*xt; A=L[2:n,2:n]; bc=b[2:n]
    o=LAMGOptions(tol=TOL,max_cycles=300); h=setup(L;options=o)
    ocL=sum(nnz(h[k].a) for k in 1:length(h.levels))/nnz(h[1].a)
    println("=== GRAPH $g  LAMG+op=$(round(ocL,digits=3)) ===")
    amg=HYPRE.BoomerAMG(;Tol=TOL,MaxIter=300,PrintLevel=2)
    HYPRE.solve!(amg,HYPRE.HYPREVector(zeros(n-1)),HYPRE.HYPREMatrix(A),HYPRE.HYPREVector(bc))
    ksp=PETSc.KSP(plib,PCOMM,A; ksp_type="cg",pc_type="gamg",ksp_rtol=TOL,ksp_max_it=300,ksp_view=true); ksp\bc
end
