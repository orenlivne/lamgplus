# DRA-QC reimplementation test suite.
# Run: julia --project=. draqc/test/runtests.jl   (from the repo root)
using Test
include(joinpath(@__DIR__, "..", "src", "DRAQC.jl"))

@testset "DRAQC" begin
    include("test_delta.jl")
    include("test_quality.jl")
    include("test_aggregate.jl")
    include("test_filter.jl")
    include("test_setup.jl")
    include("test_solve.jl")
end
