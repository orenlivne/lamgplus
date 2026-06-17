"""
    Processor

Abstract base type for cycle business-logic delegates. Mirrors
`helmholtz.hierarchy.processor.Processor`.

The `Cycle` runner calls into a Processor at well-defined hooks:

    initialize(p, l, num_levels, x)   - at the start of a cycle on the finest
                                         level l with initial guess x
    pre_process(p, l)                 - at level l, just before descending to l+1
    process_coarsest(p, l)            - at the coarsest level l
    post_process(p, l)                - at level l, just after ascending from l+1
    post_cycle(p, l)                  - at the finest level l, end of cycle
    result(p, l)                      - returns the final iterate at level l

Default implementations are no-ops. Concrete processors override what they need.
Level indices are 1-based (Julia convention): finest = 1, coarsest = num_levels.
"""
abstract type Processor end

initialize!(::Processor, l::Int, num_levels::Int, x) = nothing
pre_process!(::Processor, l::Int) = nothing
process_coarsest!(::Processor, l::Int) = nothing
post_process!(::Processor, l::Int) = nothing
post_cycle!(::Processor, l::Int) = nothing
result(::Processor, l::Int) = nothing

"""
    DryRunProcessor

A processor that records the sequence of hook calls *without performing any
numerical work*. Used to verify `Cycle` visitation logic independently of
operators and relaxers.

`p.calls` is a Vector{Tuple{Symbol,Int}} of `(hook_name, level)` calls in
the order they were issued.
"""
mutable struct DryRunProcessor <: Processor
    calls::Vector{Tuple{Symbol,Int}}
end
DryRunProcessor() = DryRunProcessor(Tuple{Symbol,Int}[])

initialize!(p::DryRunProcessor, l::Int, _num_levels::Int, _x) =
    (push!(p.calls, (:initialize, l)); nothing)
pre_process!(p::DryRunProcessor, l::Int) =
    (push!(p.calls, (:pre_process, l)); nothing)
process_coarsest!(p::DryRunProcessor, l::Int) =
    (push!(p.calls, (:process_coarsest, l)); nothing)
post_process!(p::DryRunProcessor, l::Int) =
    (push!(p.calls, (:post_process, l)); nothing)
post_cycle!(p::DryRunProcessor, l::Int) =
    (push!(p.calls, (:post_cycle, l)); nothing)
result(p::DryRunProcessor, l::Int) = p.calls

"""
    dry_cycle(γ, num_levels; finest=1) -> Cycle

Convenience constructor: a `Cycle` whose processor records all hook calls.
After `run_cycle!`, inspect `cycle.processor.calls` for the visitation order.
"""
dry_cycle(γ, num_levels::Int; finest::Int = 1) =
    Cycle(DryRunProcessor(), γ, num_levels, finest)
