"""
    DRAQC

Faithful Julia reimplementation of the Degree-aware Rooted Aggregation with
quality control (DRA-QC) graph-Laplacian multigrid solver of

    A. Napov and Y. Notay, "An Efficient Multigrid Method for Graph Laplacian
    Systems II: Robust Aggregation", SIAM J. Sci. Comput. 39(5), S379–S403, 2017.

Built bottom-up with a unit test per component (see `test/`). This is a
*competitor reimplementation* for an honest, same-machine comparison against
LAMG+ and the other solvers — it is independent of the LAMG+ solver in `src/`.

Equation/algorithm numbers in the source comments refer to that paper.
"""
module DRAQC

using LinearAlgebra
using SparseArrays

include("graph_utils.jl")
include("delta.jl")
include("quality.jl")
include("aggregate.jl")
include("filter.jl")
include("setup.jl")
include("fcg.jl")
include("cycle.jl")
include("solve.jl")

end # module
