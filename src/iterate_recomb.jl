"""
Iterate recombination (a.k.a. min-res Krylov-style enhancement) — LAMG §6.

After several cycles, we have a history of iterates `{x_i}` at a level.
Iterate recombination replaces the current iterate `x` with the
energy/residual-minimizing combination

    y = x + sum_i α_i (x_i - x)

choosing `α` so that `‖b - A y‖_2` is minimized. This is a small least-squares
problem of size n × N where N is the history depth (typically 2-4).

Port of `ProcessorSolve.m::minRes`.

Used at every level above the coarsest AGG, and at the finest level at end
of cycle. Skipped above ELIMINATION levels (no error to compensate).
"""

"""
    IterateHistory(n::Int, capacity::Int)

A bounded circular buffer of (x, r) iterate pairs, used for iterate
recombination. Capacity is typically 2-4.
"""
mutable struct IterateHistory
    X::Matrix{Float64}              # n × capacity, columns are saved iterates
    R::Matrix{Float64}              # n × capacity, columns are corresponding residuals
    E::Matrix{Float64}             # n × capacity work buffer for (X[:,i] − x)
    AE::Matrix{Float64}            # n × capacity work buffer for (r − R[:,i]) (= A·E)
    C::Matrix{Float64}             # n × capacity copy of AE for in-place QR (qr!)
    wx::Vector{Float64}            # n work buffer for E·α
    wr::Vector{Float64}            # n work buffer for AE·α
    capacity::Int
    latest::Int                     # 1..capacity; 0 means empty
    n_active::Int                   # current number of valid entries
end

IterateHistory(n::Int, capacity::Int) =
    IterateHistory(zeros(n, capacity), zeros(n, capacity), zeros(n, capacity),
                   zeros(n, capacity), zeros(n, capacity), zeros(n), zeros(n), capacity, 0, 0)

"""
    save_iterate!(h::IterateHistory, x::AbstractVector, r::AbstractVector)

Append the (x, r) pair to the circular history.
"""
function save_iterate!(h::IterateHistory, x::AbstractVector,
                       r::AbstractVector)
    h.capacity == 0 && return
    i = h.latest + 1
    i > h.capacity && (i = 1)
    h.latest = i
    if h.n_active < h.capacity
        h.n_active += 1
    end
    @views h.X[:, i] = x
    @views h.R[:, i] = r
    return
end

"""
    clear_history!(h::IterateHistory)
"""
function clear_history!(h::IterateHistory)
    h.latest = 0
    h.n_active = 0
end

"""
    min_res!(h::IterateHistory, x::AbstractVector, r::AbstractVector) -> (x, r)

Replace `(x, r)` with the energy/residual-minimizing recombination over
the history `h`. Modifies `x` and `r` in place AND returns them.

Solves `α = (R - r)⁻¹ᴸˢ * r` where the columns of `R - r` are `r - r_i`
(equivalently, `AE` where `E[:, i] = x_i - x`). Then updates
`x ← x + E α`, `r ← r - AE α`.

This is the LAMG `minRes` step, restated in terms of saved residuals so
we don't need to apply A again.
"""
# DEBUG toggle: when true, the (A,b) min_res! method recomputes the current and
# history residuals from scratch before recombining, so AE = A·E holds exactly
# regardless of any stale stored residual. Used to test whether a residual-
# consistency bug is behind "recomb hurts on web graphs".
const _MINRES_RECOMPUTE = Ref(false)
# DEBUG toggle: skip coarse-level recombination (recombine only at the finest)
# to isolate whether the coarse-level min-res is what degrades multilevel μ.
const _MINRES_NO_COARSE = Ref(false)

# DEBUG: when set to a Vector{Int}, each min_res! call pushes its n_active
# (number of history iterates available to recombine). Used to diagnose why
# recombination is far less effective in the port than in MATLAB.
const _MINRES_NACTIVE_LOG = Ref{Union{Nothing,Vector{Int}}}(nothing)

function min_res!(h::IterateHistory, x::AbstractVector, r::AbstractVector)
    N = h.n_active
    _MINRES_NACTIVE_LOG[] !== nothing && push!(_MINRES_NACTIVE_LOG[], N)
    N == 0 && return x, r
    # In-place into preallocated buffers (was: 2 fresh n×N matrices + 2 product vectors).
    # Identical arithmetic — the least-squares α = AE \ r uses the SAME QR, so the result is
    # bit-for-bit unchanged; only the storage is reused.
    Ev = @view h.E[:, 1:N]; AEv = @view h.AE[:, 1:N]
    @views Ev .= h.X[:, 1:N] .- x          # E[:,i] = x_i − x
    @views AEv .= r .- h.R[:, 1:N]          # AE[:,i] = r − r_i  (= A·E[:,i])
    Cv = @view h.C[:, 1:N]; copyto!(Cv, AEv) # qr! destroys its input → factor a reusable copy
    α = qr!(Cv) \ r                          # same QR least-squares (no n×N copy alloc)
    mul!(h.wx, Ev, α); x .+= h.wx           # x ← x + E·α
    mul!(h.wr, AEv, α); r .-= h.wr           # r ← r − AE·α
    return x, r
end

# Debug variant: recompute r = b − A·x AND every history residual
# R_i = b − A·X_i before recombining (forces AE = A·E exactly).
function min_res!(h::IterateHistory, x::AbstractVector, r::AbstractVector, A, b)
    N = h.n_active
    N == 0 && return x, r
    @inbounds for i in 1:N
        @views h.R[:, i] .= b .- A * h.X[:, i]
    end
    r .= b .- A * x
    return min_res!(h, x, r)
end
