"""
    Cycle(processor, cycle_index, num_levels; finest=1)

A generic multilevel cycle. Owns visitation control; delegates business logic
to `processor`. Mirrors `helmholtz.hierarchy.cycle.Cycle`.

Levels are 1-based: relative position 1 = finest, num_levels = coarsest.
`finest` is the absolute level index at which to start; usually 1, but >1 for
sub-cycles starting at a coarser level.

`cycle_index` is γ. Either:
- a scalar (uniform across levels), or
- a Vector{<:Real} of length `num_levels - 1`, where `cycle_index[i]` is the
  number of descents from relative position `i` (so `cycle_index[1]` is the
  rate from the finest level — always effectively 1 since the finest is
  visited exactly once per cycle; meaningful entries start at i=2).

Convention: V-cycle ↔ γ = 1.0; W-cycle ↔ γ = 2.0. LAMG's fractional cycle
runs at γ ≈ 1.5. Non-integer γ is supported via truncated `num_visits`.

Run with `run_cycle!(cycle, x)`.
"""
struct Cycle{P<:Processor, G<:Union{Real,AbstractVector{<:Real}}}
    processor::P
    cycle_index::G
    num_levels::Int
    finest::Int
end
Cycle(p::Processor, γ, num_levels::Int; finest::Int = 1) =
    Cycle(p, γ, num_levels, finest)

"""
    run_cycle!(cycle::Cycle, x)

Execute one cycle starting from the finest level. Returns whatever the
processor's `result(...)` hook returns.

Algorithm: starting at the finest level, descend until the coarsest is
reached, processing each level via `pre_process!`. At the coarsest, run
`process_coarsest!`. Then ascend, processing each level via `post_process!`.
The order is parameterized by γ: at relative position `i` (1 < i < L), the
maximum number of descents from this level per ascent from its parent is
`γ[i] * num_visits[i-1]`, where `num_visits[i-1]` is the count of descents
already made from the parent level (relative position `i-1`).
"""
function run_cycle!(c::Cycle, x)
    f = c.finest
    L = c.num_levels
    coarsest_abs = f + L - 1
    γ = _expand_gamma(c.cycle_index, L)   # length L-1, indices 1..L-1
    num_visits = zeros(Int, L - 1)        # length L-1, indices 1..L-1

    initialize!(c.processor, f, L, x)

    l = f
    while true
        i = l - f + 1                     # relative position, 1..L

        if l == coarsest_abs
            k = l - 1
        else
            max_visits = (i == 1) ? 1 : γ[i] * num_visits[i - 1]
            k = (num_visits[i] < max_visits) ? (l + 1) : (l - 1)
        end

        if l == coarsest_abs
            process_coarsest!(c.processor, l)
        end

        if k < f
            break
        elseif k > l
            num_visits[i] += 1
            pre_process!(c.processor, l)
        else
            post_process!(c.processor, k)
        end
        l = k
    end

    post_cycle!(c.processor, f)
    return result(c.processor, f)
end

# Internal: expand a scalar γ to a per-level vector. Returns a vector of length
# num_levels - 1, indexed 1..num_levels-1 by relative position. γ[1] is unused
# at the finest level (max_visits=1 is hard-coded there).
_expand_gamma(γ::Real, num_levels::Int) = fill(float(γ), num_levels - 1)
function _expand_gamma(γ::AbstractVector, num_levels::Int)
    @assert length(γ) == num_levels - 1 "γ must have length num_levels-1 = $(num_levels-1)"
    return collect(float.(γ))
end
