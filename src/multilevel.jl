"""
    Multilevel

An ordered hierarchy of `Level` objects. `mlh[1]` is the finest level,
`mlh[end]` is the coarsest. Mirrors `helmholtz.hierarchy.multilevel.Multilevel`.

Use `push!(mlh, level)` to append a coarser level during setup.
"""
mutable struct Multilevel
    levels::Vector{Level}
    perm::Vector{Int}   # RCM reordering of the finest input (empty = identity / no reorder)
end

Multilevel() = Multilevel(Level[], Int[])
Multilevel(finest::Level) = Multilevel(Level[finest], Int[])
Multilevel(levels::Vector{Level}) = Multilevel(levels, Int[])

Base.length(mlh::Multilevel) = length(mlh.levels)
Base.iterate(mlh::Multilevel, state::Int = 1) =
    state > length(mlh.levels) ? nothing : (mlh.levels[state], state + 1)
Base.getindex(mlh::Multilevel, i::Int) = mlh.levels[i]
Base.lastindex(mlh::Multilevel) = length(mlh.levels)
Base.push!(mlh::Multilevel, level::Level) = (push!(mlh.levels, level); mlh)

"""
    finest_level(mlh::Multilevel) -> Level
"""
finest_level(mlh::Multilevel) = mlh.levels[1]

"""
    num_levels(mlh::Multilevel) -> Int
"""
num_levels(mlh::Multilevel) = length(mlh.levels)
