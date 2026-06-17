"""
    MockCycleProcessor

A "mock" cycle that uses an ideal coarse-level correction to gauge the quality
of a coarse-variable set BEFORE designing the interpolation operator. Mirrors
`helmholtz.solve.mock_cycle.MockCycle` from `mg/amgplus/src/`.

The mock cycle runs:

    1. `num_steps` relaxation sweeps on Ax = 0 at the fine level.
    2. `num_corrector_steps` ideal coarse-level corrections that exactly
       zero out the coarse-variable values:
           x ← x − Qᵀ (QQᵀ)⁻¹ (Qx − q₀)
       where Q is the coarsening matrix and q₀ are baseline (target) coarse
       values (here, 0 for the homogeneous problem Ax = 0).

This is the "habituated compatible relaxation" in AMG literature (cf. amgplus
§"Numerical quantitative performance predictors" and Brandt's MG guide §5).
A small mock-cycle convergence factor predicts good two-level convergence
once an interpolation is chosen.
"""
struct MockCycleProcessor <: Processor
    relaxer::Relaxer
    q::SparseMatrixCSC{Float64,Int}
    qqt_factor::Any                        # Cholesky factorization of Q*Qᵀ
    num_steps::Int
    num_corrector_steps::Int
    ω::Float64
    # mutable state per cycle:
    x::Ref{Vector{Float64}}
end

function MockCycleProcessor(relaxer::Relaxer, q::SparseMatrixCSC;
                            num_steps::Int = 1,
                            num_corrector_steps::Int = 1,
                            ω::Real = 1.0)
    qqt = Matrix(q * q')                   # n_c × n_c, usually small
    fact = cholesky(Symmetric(qqt))
    MockCycleProcessor(relaxer, q, fact, num_steps, num_corrector_steps,
                       Float64(ω), Ref(Float64[]))
end

function initialize!(p::MockCycleProcessor, l::Int, num_levels::Int, x)
    p.x[] = collect(Float64.(x))
end

function pre_process!(p::MockCycleProcessor, l::Int)
    # Relax `num_steps` times on Ax = 0.
    b = zeros(length(p.x[]))
    for _ in 1:p.num_steps
        relax!(p.relaxer, p.x[], b; sweeps = 1)
    end
end

function process_coarsest!(p::MockCycleProcessor, l::Int)
    # Ideal coarse-level correction: project out the coarse-variable component.
    for _ in 1:p.num_corrector_steps
        xc = p.q * p.x[]                   # residual coarse value (target = 0)
        y = p.qqt_factor \ xc              # solve (Q Qᵀ) y = Q x
        p.x[] .-= p.ω .* (p.q' * y)
    end
end

result(p::MockCycleProcessor, l::Int) = p.x[]

"""
    mock_cycle(relaxer, q; num_steps=1, num_corrector_steps=1, ω=1.0) -> Cycle

Convenience constructor for a 2-level mock cycle.
"""
function mock_cycle(relaxer::Relaxer, q::SparseMatrixCSC;
                    num_steps::Int = 1, num_corrector_steps::Int = 1,
                    ω::Real = 1.0)
    proc = MockCycleProcessor(relaxer, q; num_steps = num_steps,
                              num_corrector_steps = num_corrector_steps, ω = ω)
    Cycle(proc, 1.0, 2)                    # 2 levels, V-cycle
end
