"""
    LAMG

LAMG+ — a lean, parameter-free algebraic multigrid solver for graph-Laplacian
linear systems `Lφ=b`. A re-derivation of Lean Algebraic Multigrid (Livne–Brandt
2012) with a strength-of-connection veto and a selective caliber-2 interpolation.

Quickstart:

    using LAMG, SparseArrays
    A = grid2d_laplacian(64, 64)          # any graph Laplacian L = D - W
    b = randn(size(A,1)); b .-= sum(b)/length(b)
    h = setup(A)
    x, info = solve(h, b)                  # x ⟂ 1, ‖Lx-b‖ ≤ 1e-8‖b‖
    info.cycles
"""
module LAMG

using LinearAlgebra
using SparseArrays
using Random
using Statistics
using Printf
using Base.Threads: @threads, nthreads

include("graph.jl")
include("relaxer.jl")
include("elimination.jl")
include("weight_aware_elimination.jl")
include("level.jl")
include("multilevel.jl")
include("iterate_recomb.jl")
include("processor.jl")
include("cycle.jl")
include("shrinkage.jl")
include("mock_cycle.jl")
include("relax_cycle.jl")
include("solve_cycle.jl")
include("interpolation.jl")
include("coarsen.jl")
include("setup.jl")

export
    # graph
    laplacian, path_laplacian, grid2d_laplacian, grid3d_laplacian,
    random_graph_laplacian,
    is_laplacian, is_graph_laplacian, null_vector,
    connected_components, largest_component,
    # relaxers
    Relaxer, GaussSeidelRelaxer, JacobiRelaxer, relax!, update_fas!,
    # elimination
    EliminationStage, low_degree_nodes, elimination_operators, eliminate_once,
    WAEliminationLevel, weight_aware_eliminate, wae_select, wae_strong_degree,
    wae_restrict, wae_interpolate,
    # hierarchy
    Level, Multilevel, create_finest_level, create_agg_level,
    create_elimination_level, finest_level, num_levels,
    is_elimination, is_finest, operator,
    restrict_op, coarsen_op, interpolate_op,
    restrict_elimination, interpolate_elimination, coarse_type,
    # cycles
    Processor, Cycle, run_cycle!,
    RelaxCycleProcessor, SolveCycleProcessor, DryRunProcessor, MockCycleProcessor,
    relax_cycle, solve_cycle, mock_cycle, dry_cycle,
    # diagnostics
    shrinkage_factor, ShrinkageResult,
    # iterate recombination
    IterateHistory, save_iterate!, clear_history!, min_res!,
    # interpolation
    piecewise_constant_interpolation, caliber2_interpolation, galerkin_coarse_operator,
    aggregation_from_partition,
    # coarsen
    Aggregation, affinity, aggregate, energy_ratio,
    # setup / solve
    LAMGOptions, LAMGHierarchy, setup, solve

end # module
