"""
    RelaxCycleProcessor(multilevel, ν_pre, ν_post, ν_coarsest)

Multilevel relaxation cycle for Ax = 0 (used for measuring shrinkage on the
full hierarchy). Mirrors `helmholtz.solve.relax_cycle.RelaxCycleProcessor`.
"""
mutable struct RelaxCycleProcessor <: Processor
    mlh::Multilevel
    ν_pre::Int
    ν_post::Int
    ν_coarsest::Int
    debug::Bool
    # per-level state, allocated in `initialize!`
    x::Vector{Vector{Float64}}
    b::Vector{Vector{Float64}}
    x_initial::Vector{Vector{Float64}}
end

function RelaxCycleProcessor(mlh::Multilevel; ν_pre::Int = 2, ν_post::Int = 2,
                             ν_coarsest::Int = 4, debug::Bool = false)
    n = num_levels(mlh)
    empty = Vector{Vector{Float64}}(undef, n)
    RelaxCycleProcessor(mlh, ν_pre, ν_post, ν_coarsest, debug,
                        empty, copy(empty), copy(empty))
end

function initialize!(p::RelaxCycleProcessor, l::Int, _num_levels::Int, x)
    n = num_levels(p.mlh)
    p.x = Vector{Vector{Float64}}(undef, n)
    p.b = Vector{Vector{Float64}}(undef, n)
    p.x_initial = Vector{Vector{Float64}}(undef, n)
    p.x[l] = collect(Float64.(x))
    p.b[l] = zeros(Float64, size(p.mlh[l]))
end

function process_coarsest!(p::RelaxCycleProcessor, l::Int)
    lv = p.mlh[l]
    for _ in 1:p.ν_coarsest
        relax!(lv, p.x[l], p.b[l]; sweeps = 1)
    end
end

function pre_process!(p::RelaxCycleProcessor, l::Int)
    lv = p.mlh[l]
    _relax(p, l, p.ν_pre)
    # FAS coarsening
    lc = l + 1
    coarse = p.mlh[lc]
    x = p.x[l]
    xc0 = coarsen_op(coarse, x)
    p.x_initial[lc] = copy(xc0)
    p.x[lc] = copy(xc0)
    # b^c = R(b - A x) + A^c x^c0
    p.b[lc] = restrict_op(coarse, p.b[l] .- operator(lv, x)) .+ operator(coarse, xc0)
end

function post_process!(p::RelaxCycleProcessor, l::Int)
    lc = l + 1
    coarse = p.mlh[lc]
    p.x[l] .+= interpolate_op(coarse, p.x[lc] .- p.x_initial[lc])
    _relax(p, l, p.ν_post)
end

function _relax(p::RelaxCycleProcessor, l::Int, n::Int)
    n <= 0 && return
    lv = p.mlh[l]
    for _ in 1:n
        relax!(lv, p.x[l], p.b[l]; sweeps = 1)
    end
end

result(p::RelaxCycleProcessor, l::Int) = p.x[l]

"""
    relax_cycle(mlh; γ=1.0, ν_pre=2, ν_post=2, ν_coarsest=4, num_levels=length(mlh)) -> Cycle
"""
function relax_cycle(mlh::Multilevel; γ::Real = 1.0,
                     ν_pre::Int = 2, ν_post::Int = 2, ν_coarsest::Int = 4,
                     num_levels::Int = length(mlh))
    proc = RelaxCycleProcessor(mlh; ν_pre = ν_pre, ν_post = ν_post,
                               ν_coarsest = ν_coarsest)
    Cycle(proc, γ, num_levels)
end
