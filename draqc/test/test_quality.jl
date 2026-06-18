# Unit tests for the aggregate quality measure μ(G) and the eq.14 Cholesky test.
# Gold references: Napov–Notay Examples 3.1 (clique, exact) and 3.2 (two cliques,
# closed-form bounds), plus a cross-check that quality_ok_factor (eq.14) agrees
# with the exact generalized-eigenvalue definition of μ(G) (Thm 3.3).
using Test, SparseArrays, LinearAlgebra, Random

# Clique Laplacian A_G = a(n_g I − 11ᵀ): off-diagonal −a, internal degree (n_g−1)a.
clique_AG(ng, a) = a .* (ng .* Matrix(1.0I, ng, ng) .- ones(ng, ng))

@testset "quality μ(G)" begin

    @testset "Example 3.1 (clique): μ = 1 + γ/(a·n_g) exactly" begin
        for ng in (2, 3, 5, 8), a in (1.0, 0.5, 3.0), γ in (0.7, 2.0, 11.0)
            AG = clique_AG(ng, a)
            XG = AG + γ * I
            μ = DRAQC.mu_exact(AG, Matrix(XG))
            @test μ ≈ 1 + γ / (a * ng) rtol = 1e-10
        end
    end

    @testset "Example 3.2 (two cliques share 1 vertex): closed-form bounds" begin
        # Two cliques of size n1+1 and n2+1 sharing exactly one common vertex,
        # all weights a, Γ_G = γI. Eq.(12): lower ≤ μ(G) ≤ upper.
        function two_clique_AG(n1, n2, a)
            ng = n1 + n2 + 1
            v1 = 1:(n1+1)            # clique 1 (vertex 1 is the shared one)
            v2 = vcat(1, (n1+2):ng)  # clique 2 (shares vertex 1)
            W = zeros(ng, ng)
            for vs in (v1, v2), i in vs, j in vs
                i != j && (W[i, j] = a)
            end
            return Diagonal(vec(sum(W; dims=2))) - W |> Matrix
        end
        for (n1, n2, a, γ) in ((3, 4, 1.0, 1.0), (5, 6, 2.0, 3.0), (4, 9, 1.0, 5.0))
            AG = two_clique_AG(n1, n2, a)
            XG = AG + γ * I
            μ = DRAQC.mu_exact(AG, Matrix(XG))
            lower = 1 + γ / (a * min(n1, n2)) * (1 - 1 / (n1 + n2 - 1))
            upper = 1 + γ / a
            @test lower - 1e-9 <= μ <= upper + 1e-9
        end
    end

    @testset "eq.14 Cholesky test ⟺ μ(G) ≤ κ̄ (cross-check)" begin
        rng = MersenneTwister(11)
        for trial in 1:40
            ng = rand(rng, 3:8)
            # random connected small graph Laplacian as A_G (zero row sum, null=1)
            W = zeros(ng, ng)
            for i in 1:ng-1
                w = rand(rng) + 0.1; W[i, i+1] = w; W[i+1, i] = w
            end
            for i in 1:ng, j in i+1:ng
                if rand(rng) < 0.4
                    w = rand(rng) + 0.1; W[i, j] = w; W[j, i] = w
                end
            end
            AG = Diagonal(vec(sum(W; dims=2))) - W |> Matrix
            γ = rand(rng, ng) .* 5 .+ 0.05           # positive ⇒ X_G PD
            XG = AG + Diagonal(γ)
            μ = DRAQC.mu_exact(AG, XG)
            # test κ̄ clearly below and clearly above μ (avoid the fuzzy boundary)
            @test DRAQC.quality_ok_factor(AG, XG, 1.1 * μ) == true
            @test DRAQC.quality_ok_factor(AG, XG, 0.9 * μ) == false
        end
    end

    @testset "aggregate_quality_matrices assembly (A_G·1=0, γ=δ_G+2·ext)" begin
        # Build a known sparse Laplacian and pull an interior aggregate.
        rng = MersenneTwister(3)
        n = 16
        W = spzeros(n, n)
        for i in 1:n-1
            w = rand(rng) + 0.5; W[i, i+1] = w; W[i+1, i] = w
        end
        for i in 1:n, j in i+1:n
            if rand(rng) < 0.3
                w = rand(rng) + 0.5; W[i, j] = w; W[j, i] = w
            end
        end
        A = sparse(Diagonal(vec(sum(W; dims=2)))) - W
        δ = DRAQC.delta_vector(A)
        G = [3, 4, 5, 6]
        AG, γ, XG = DRAQC.aggregate_quality_matrices(A, G, δ)
        @test AG * ones(length(G)) ≈ zeros(length(G)) atol = 1e-12   # zero row sum
        @test issymmetric(AG)
        # ext_p = a_jj − internal degree; γ_p = δ_{G[p]} + 2·ext_p
        _, ext = DRAQC.subgraph_laplacian(A, G)
        @test γ ≈ [δ[G[p]] + 2 * ext[p] for p in 1:length(G)]
        @test XG ≈ AG + Diagonal(γ)
        # external degree consistency: a_jj = internal + external
        for p in 1:length(G)
            @test A[G[p], G[p]] ≈ AG[p, p] + ext[p]
        end
    end
end
