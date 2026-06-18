"""
    DRAQCHybrid

Prototype hybrid graph-Laplacian solver: Napov–Notay DRA-QC (quality-controlled
aggregation + K-cycle, reused verbatim from the `DRAQC` package) with the LAMG
**strength-of-connection veto** grafted onto aggregation. The veto supplies the
directional information DRA lacks, targeting DRA-QC's slow regime — grid-aligned
anisotropy and high-contrast coefficients — without disturbing its behavior on the
graphs it already handles well.

Only the aggregation differs from DRA-QC; δ, quality control, Galerkin, the K-cycle,
and the FCG(1) solve are DRAQC's, unchanged.
"""
module DRAQCHybrid

using LinearAlgebra
using SparseArrays

# bring in the faithful DRA-QC implementation (reused wholesale)
include(joinpath(@__DIR__, "..", "..", "draqc", "src", "DRAQC.jl"))
using .DRAQC

include("soc_aggregate.jl")
include("hybrid_setup.jl")
include("elim.jl")

end # module
