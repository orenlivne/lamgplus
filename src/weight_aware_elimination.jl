"""
Weight-aware low-degree elimination — a LAMG extension for *weighted* anisotropic
operators.

Standard low-degree elimination (`elimination.jl`) selects an independent set `F` over
*all* edges, so `L_FF` is diagonal and the Schur complement is trivially cheap. On a
strongly anisotropic operator (a few strong couplings per node plus several very weak
ones) that selection sees full degree and can only peel a red-black half before the
Schur fill stops it.

Weight-aware elimination instead counts degree and tests independence over the
**strong** edges only (|w| ≥ τ · row-max), permitting *weak* edges between members of
`F`. `L_FF` is then merely **near-diagonal** (its off-diagonals are the tiny weak
couplings), still cheap to factor, and the Schur complement `L_C = L_CC −
L_CF L_FF⁻¹ L_FC` is **exact**. On a strong-x/weak-y grid this collapses the strong
chains exactly — algebraic semicoarsening with *zero* energy inflation.

Safety: on an unweighted graph every edge is "strong" (|w| = row-max), so the selection
reduces *exactly* to the standard low-degree independent set — a verified no-op. The
strength threshold `τ` is small by default, so the operator is inert unless a genuine
weight disparity exists.
"""

"""
    wae_strong_degree(A; τ=0.01) -> (sdeg::Vector{Int}, rowmax::Vector{Float64})

Per-node count of incident **strong** edges (`|w| ≥ τ · row-max`), and the per-node
maximum incident `|w|`.
"""
function wae_strong_degree(A::SparseMatrixCSC; τ::Real = 0.01)
    n = size(A, 1); rows = rowvals(A); vals = nonzeros(A)
    rowmax = zeros(n)
    @inbounds for j in 1:n, k in nzrange(A, j)
        i = rows[k]
        i != j && (rowmax[i] = max(rowmax[i], abs(vals[k])))
    end
    sdeg = zeros(Int, n)
    @inbounds for j in 1:n, k in nzrange(A, j)
        i = rows[k]
        (i != j && rowmax[i] > 0 && abs(vals[k]) >= τ * rowmax[i]) && (sdeg[i] += 1)
    end
    sdeg, rowmax
end

"""
    wae_select(A; τ=0.01, dmax=4) -> (F::Vector{Int}, C::Vector{Int})

Greedy maximal independent set (over **strong** edges) of nodes whose strong-degree is
in `1..dmax` and whose diagonal is nonzero. Weak edges between members of `F` are
permitted. On an unweighted graph this returns the same `F` as the standard
`low_degree_nodes` independent set.
"""
function wae_select(A::SparseMatrixCSC; τ::Real = 0.01, dmax::Int = 4)
    n = size(A, 1); rows = rowvals(A); vals = nonzeros(A)
    sdeg, rowmax = wae_strong_degree(A; τ = τ)
    # 0 = undecided candidate, 1 = chosen (in F), 2 = ineligible / blocked
    status = fill(0, n)
    @inbounds for i in 1:n
        (sdeg[i] == 0 || sdeg[i] > dmax || abs(A[i, i]) <= 1e-300) && (status[i] = 2)
    end
    @inbounds for i in 1:n
        status[i] != 0 && continue
        ok = true
        for k in nzrange(A, i)
            j = rows[k]; j == i && continue
            if abs(vals[k]) >= τ * rowmax[i] && status[j] == 1
                ok = false; break
            end
        end
        if ok
            status[i] = 1
            for k in nzrange(A, i)
                j = rows[k]; j == i && continue
                (abs(vals[k]) >= τ * rowmax[j] && status[j] == 0) && (status[j] = 2)
            end
        end
    end
    findall(==(1), status), findall(!=(1), status)
end

"""
A single weight-aware elimination level: the strong-independent set `F`, its complement
`C`, the near-diagonal block `Aff = A[F,F]`, the coupling `Afc = A[F,C]`, and the exact
Schur complement `Ac` on `C`.
"""
struct WAEliminationLevel
    F::Vector{Int}
    C::Vector{Int}
    Aff::SparseMatrixCSC{Float64,Int}
    Afc::SparseMatrixCSC{Float64,Int}
    Ac::SparseMatrixCSC{Float64,Int}
    n::Int
end

"""
    weight_aware_eliminate(A; τ=0.01, dmax=4) -> Union{Nothing, WAEliminationLevel}

Exactly Schur-eliminate a strong-independent low-degree set from the symmetric operator
`A`. Returns `nothing` (a no-op) when no node qualifies. The reduced operator
`level.Ac` is the exact Schur complement; `wae_restrict` / `wae_interpolate` carry the
right-hand side down and the solution back up so that, for any `b`, solving
`level.Ac x_C = wae_restrict(level, b)` and `wae_interpolate(level, x_C, b)` reproduces
the solution of `A x = b` exactly.
"""
function weight_aware_eliminate(A::SparseMatrixCSC; τ::Real = 0.01, dmax::Int = 4)
    F, C = wae_select(A; τ = τ, dmax = dmax)
    isempty(F) && return nothing
    Aff = A[F, F]; Afc = A[F, C]; Acf = A[C, F]; Acc = A[C, C]
    Ac = Acc - Acf * (Aff \ Matrix(Afc))
    WAEliminationLevel(F, C, Aff, Afc, sparse(Symmetric(Ac)), size(A, 1))
end

"""
    wae_restrict(level, b) -> b_C

Coarse right-hand side `b_C − A_CF A_FF⁻¹ b_F` (`A_CF = A_FC^⊤`).
"""
wae_restrict(L::WAEliminationLevel, b::AbstractVector) =
    b[L.C] .- (L.Afc' * (L.Aff \ b[L.F]))

"""
    wae_interpolate(level, x_C, b) -> x

Reconstruct the full solution: `x_C` on the coarse set and
`x_F = A_FF⁻¹ (b_F − A_FC x_C)` on the eliminated set.
"""
function wae_interpolate(L::WAEliminationLevel, xC::AbstractVector, b::AbstractVector)
    x = zeros(L.n)
    x[L.C] = xC
    x[L.F] = L.Aff \ (b[L.F] .- L.Afc * xC)
    x
end
