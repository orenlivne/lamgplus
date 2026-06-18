# Unit tests for aggregate filtering (§4.3): bad-vertex removal and subgroup extraction.
using Test, SparseArrays, LinearAlgebra

lap_from_edges(edges, n) = (W = spzeros(n, n);
    for (i, j, w) in edges; W[i, j] = w; W[j, i] = w; end;
    sparse(Diagonal(vec(sum(W; dims=2)))) - W)

@testset "aggregate filtering (§4.3)" begin

    @testset "bad-vertex removal drops weakly-attached high-γ vertices" begin
        # core clique {1,2,3,4} (w=1); vertices 5,6 attach to the core by a weak
        # edge (ε) but carry large external degree ⇒ must be removed.
        ε = 0.01
        edges = [(1,2,1.0),(1,3,1.0),(1,4,1.0),(2,3,1.0),(2,4,1.0),(3,4,1.0),
                 (1,5,ε),(1,6,ε),
                 (5,7,1.0),(5,8,1.0),(5,9,1.0),
                 (6,10,1.0),(6,11,1.0),(6,12,1.0)]
        A = lap_from_edges(edges, 12)
        δ = DRAQC.delta_vector(A)
        S = DRAQC.bad_vertex_removal(A, [1,2,3,4,5,6], 1, δ)
        @test sort(S) == [1,2,3,4]
        # and the refined aggregate is the good core, well below the threshold
        Sref = DRAQC.refine_aggregate(A, [1,2,3,4,5,6], 1, δ)
        @test sort(Sref) == [1,2,3,4]
        AG, γ, XG = DRAQC.aggregate_quality_matrices(A, Sref, δ)
        @test DRAQC.mu_exact(AG, XG) <= 10.0
    end

    @testset "subgroup extraction (Example-4.1 mechanism)" begin
        # Two cliques {1,2,3,4},{1,5,6,7} sharing root 1; each non-root core vertex
        # has 3 external leaves. Local removal keeps all (each passes (19)), but the
        # global μ exceeds κ̄ ⇒ factorization fails ⇒ subgroup extraction splits off
        # one clique, dropping μ — exactly the paper's 14.4 → 4.2 behavior.
        edges = Tuple{Int,Int,Float64}[]
        for cl in ([1,2,3,4], [1,5,6,7]), i in cl, j in cl
            i < j && push!(edges, (i, j, 1.0))
        end
        nid = 8
        for v in 2:7, _ in 1:3
            push!(edges, (v, nid, 1.0)); nid += 1
        end
        A = lap_from_edges(edges, nid - 1)
        δ = DRAQC.delta_vector(A)
        G = collect(1:7)

        AGf, γf, XGf = DRAQC.aggregate_quality_matrices(A, G, δ)
        μfull = DRAQC.mu_exact(AGf, XGf)
        @test μfull > 10.0                                   # combined aggregate is bad
        @test sort(DRAQC.bad_vertex_removal(A, G, 1, δ)) == G  # removal alone keeps all

        Sref = DRAQC.refine_aggregate(A, G, 1, δ)
        @test 1 in Sref                                      # root retained
        @test issubset(Set(Sref), Set(G))                   # subset of original
        @test length(Sref) < length(G)                      # extraction shrank it
        AGr, γr, XGr = DRAQC.aggregate_quality_matrices(A, Sref, δ)
        @test DRAQC.mu_exact(AGr, XGr) <= 10.0              # now acceptable quality
    end

    @testset "refine_aggregate contract on benign aggregates" begin
        # On a 2D grid, DRA aggregates are already good: refine keeps them intact and
        # the result always passes the quality bound, retains the root, and ⊆ input.
        function grid2d_lap(nx, ny)
            n = nx * ny; idx(i, j) = (j - 1) * nx + i
            I, J, V = Int[], Int[], Float64[]
            for j in 1:ny, i in 1:nx, (di, dj) in ((1,0),(-1,0),(0,1),(0,-1))
                ii, jj = i + di, j + dj
                if 1 <= ii <= nx && 1 <= jj <= ny
                    push!(I, idx(i,j)); push!(J, idx(ii,jj)); push!(V, -1.0)
                end
            end
            W = sparse(I, J, V, n, n); sparse(Diagonal(-vec(sum(W; dims=2)))) + W
        end
        A = grid2d_lap(10, 10)
        δ = DRAQC.delta_vector(A)
        agg, nc = DRAQC.dra_aggregate(A)
        for a in 1:nc
            members = findall(==(a), agg)
            root = members[1]
            S = DRAQC.refine_aggregate(A, members, root, δ)
            @test root in S
            @test issubset(Set(S), Set(members))
            if 2 <= length(S) <= 1024
                AG, γ, XG = DRAQC.aggregate_quality_matrices(A, S, δ)
                @test DRAQC.mu_exact(AG, XG) <= 10.5
            end
        end
    end
end
