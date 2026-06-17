"""
    ShrinkageResult

Result struct from `shrinkage_factor`.

Fields:
- `factor`              :: μ, the shrinkage factor (geometric-mean residual
                           reduction over the first `podr_sweeps` sweeps).
- `podr_sweeps`         :: Point Of Diminishing Returns index (i.e., the
                           sweep count after which efficiency stops improving).
- `residual_history`    :: Matrix of size (num_sweeps+1, num_examples) with
                           residual norms per sweep per starting vector.
- `conv_history`        :: Vector of geometric-mean per-sweep convergence
                           factors.
- `conv_factor`         :: asymptotic per-sweep convergence factor (last sweep).
"""
struct ShrinkageResult
    factor::Float64
    podr_sweeps::Int
    residual_history::Matrix{Float64}
    conv_history::Vector{Float64}
    conv_factor::Float64
end

"""
    shrinkage_factor(operator, method!, n::Int; ...) -> ShrinkageResult

Estimate the shrinkage factor μ of an iterative method on Ax = 0, the
geometric-mean per-sweep residual reduction over the first `podr_sweeps`
sweeps, where `podr_sweeps` is the Point Of Diminishing Returns.

Arguments:
- `operator(x)` returns Ax (residual = b - Ax = -Ax for the homogeneous case).
- `method!(x, b)` performs one iteration step in place; the implementation
  should not assume b == 0 (we pass b = 0 here).
- `n` is the vector size.

Keyword arguments:
- `num_examples` :: number of random starts to average over (default 5).
- `max_sweeps`   :: cap on sweep count (default 100).
- `slow_conv_factor` :: stop early when per-sweep factor exceeds this (default 1.3).
- `leeway_factor` :: efficiency inflation factor used to find the PODR (1.2).
- `min_residual_reduction` :: ensure reduction by at least this fraction (0.2).
- `rng` :: RNG for initial vectors.

Mirrors `helmholtz.solve.smoothing.shrinkage_factor` from `mg/amgplus/src/`.
"""
function shrinkage_factor(operator::Function, method!::Function, n::Int;
                          num_examples::Int = 5,
                          max_sweeps::Int = 100,
                          slow_conv_factor::Real = 1.3,
                          leeway_factor::Real = 1.2,
                          min_residual_reduction::Real = 0.2,
                          rng = Random.default_rng())
    # Initial random vectors, columns of X.
    X = 2 .* rand(rng, n, num_examples) .- 1.0
    X ./= norm(X)
    b = zeros(n, num_examples)
    residual_history = Vector{Vector{Float64}}()
    r_norm = [norm(operator(view(X, :, j))) for j in 1:num_examples]
    push!(residual_history, copy(r_norm))
    conv_factor = 0.0
    i = 0
    while conv_factor < slow_conv_factor && i < max_sweeps
        i += 1
        r_old = copy(r_norm)
        for j in 1:num_examples
            xj = view(X, :, j)
            bj = view(b, :, j)
            method!(xj, bj)
        end
        for j in 1:num_examples
            r_norm[j] = norm(operator(view(X, :, j)))
        end
        push!(residual_history, copy(r_norm))
        conv_factor = mean(r_norm ./ max.(r_old, 1e-30))
    end

    H = reduce(hcat, residual_history)'    # rows = sweeps, cols = examples
    H = Matrix{Float64}(H)
    num_iters = size(H, 1) - 1

    # PODR detection: examine the geometric-mean efficiency curve.
    reduction = vec(mean(H ./ H[1:1, :]; dims = 2))
    sweep_count = max.(collect(0:num_iters), 1e-2)
    efficiency = reduction .^ (1 ./ sweep_count)

    min_eff = minimum(efficiency)
    last_within_leeway = findlast(efficiency .< leeway_factor * min_eff)
    podr = last_within_leeway === nothing ? num_iters : last_within_leeway - 1
    has_min_red = findfirst(reduction .< min_residual_reduction)
    if has_min_red !== nothing
        podr = max(podr, has_min_red - 1)
    end
    podr = max(podr, 1)
    factor = efficiency[podr + 1]

    # Per-sweep convergence factors (history of geometric means).
    conv_history = vec(mean(exp.(diff(log.(H); dims = 1)); dims = 2))
    asym = isempty(conv_history) ? 1.0 : conv_history[end]

    return ShrinkageResult(factor, podr, H, conv_history, asym)
end
