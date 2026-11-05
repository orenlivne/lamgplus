# Homogeneous-problem convergence factor (Brandt / LAMG §5.1 protocol): the exact solution is
# known (b = L x_true), start from x = 0 so the initial error is -x_true (random, zero mean),
# run a fixed number of cycles and report the geometric mean of the per-cycle ERROR reduction
# (energy norm) over the stationary tail. Independent of the residual window artefacts.
using LAMG, LinearAlgebra, Statistics, Random
function acf_homog(h, L, o::LAMGOptions; ncyc::Int=30, tail::Int=10, seed::Int=5)
    n = size(L,1); xt = randn(MersenneTwister(seed), n); xt .-= mean(xt); b = L*xt
    p = h.perm; bp = isempty(p) ? b : b[p]; xtp = isempty(p) ? xt : xt[p]; A = LAMG.finest_level(h).a
    proc = LAMG.SolveCycleProcessor(h, bp; ν_pre=o.ν_pre, ν_post=o.ν_post, ν_coarsest=o.ν_coarsest,
                                    do_recomb=o.do_recomb, history_size=o.history_size,
                                    use_direct_coarsest=(o.ν_coarsest == -1), rhs_correction=o.rhs_correction)
    g = (o.γ_coarse_growth <= 1.0 && length(h) >= 18) ? 1.15 : o.γ_coarse_growth
    cyc = LAMG.Cycle(proc, LAMG._build_gamma_vec(o.γ, o.γ_coarse, g, length(h), h), length(h))
    x = zeros(n); e = x .- xtp; en = [sqrt(dot(e, A*e))]
    for k in 1:ncyc
        LAMG.run_cycle!(cyc, x); x = copy(LAMG.result(proc, 1)); e = x .- xtp
        push!(en, sqrt(max(dot(e, A*e), 0.0)))
        en[end] < 1e-13*en[1] && break
    end
    f = en[2:end] ./ en[1:end-1]
    # Roundoff floor: on graphs with a wide degree range the energy-norm error bottoms out at
    # ~1e-8..1e-12 of its initial value and the factors after that are noise (often > 1). The
    # stationary tail is taken from the cycles whose error is still above 100× the attained
    # minimum; the last `tail` of those enter the geometric mean.
    # A floor has been hit only if the error ever INCREASED (impossible for the exact iteration);
    # a monotone slow crawl (NLR: 0.84 for 30 cycles) is a genuine tail and is kept whole.
    hit = any(>(1.0), f)
    floor = hit ? minimum(en) : 0.0
    valid = [k for k in 1:length(f) if en[k+1] > 100*floor]
    isempty(valid) && (valid = [1])
    t = f[valid[max(1,end-tail+1):end]]
    (acf = exp(mean(log.(max.(t,1e-300)))), factors = f, cycles = length(f), valid = length(valid))
end
