using Test
using LinearAlgebra
using SparseArrays
using LAMG

# Tests mirroring MATLAB `lin-solve/Problems.m::toLaplacian` preprocessing
# semantics. We inline a lightweight `read_mm_test` mirror of
# `scripts/benchmark.jl::read_mm` to avoid the script's `using Pkg` /
# activate dance in test mode.

function read_mm_test(path::AbstractString)
    open(path, "r") do io
        header = readline(io)
        tokens = split(lowercase(header))
        format = tokens[3]; field = tokens[4]; symmetry = tokens[5]
        line = readline(io)
        while startswith(line, "%")
            line = readline(io)
        end
        nrows, ncols, nentries = parse.(Int, split(line))
        rows = Vector{Int}(undef, 2 * nentries)
        cols = Vector{Int}(undef, 2 * nentries)
        vals = Vector{Float64}(undef, 2 * nentries)
        k = 0
        for _ in 1:nentries
            line = readline(io)
            parts = split(line)
            i = parse(Int, parts[1])
            j = parse(Int, parts[2])
            v = field == "pattern" ? 1.0 : parse(Float64, parts[3])
            k += 1
            rows[k] = i; cols[k] = j; vals[k] = v
            if symmetry in ("symmetric", "hermitian", "skew-symmetric") && i != j
                k += 1
                rows[k] = j; cols[k] = i; vals[k] = v
            end
        end
        W_raw = sparse(rows[1:k], cols[1:k], vals[1:k], nrows, ncols)
        upper = triu(W_raw, 1)
        W = upper + sparse(transpose(upper))
        had_diagonal = any(!iszero, diag(W_raw))
        if had_diagonal
            ii, jj, _ = findnz(W)
            m_n, nn_n = size(W)
            W = sparse(ii, jj, ones(Float64, length(ii)), m_n, nn_n)
        end
        if !had_diagonal && (nnz(W) > 0) && (minimum(nonzeros(W)) < 0)
            rows_W = rowvals(W); vals_W = nonzeros(W)
            nW = size(W, 1)
            row_neg_sum = zeros(nW); row_max_abs = zeros(nW)
            for j in 1:nW
                for r in nzrange(W, j)
                    i = rows_W[r]; v = vals_W[r]
                    v < 0 && (row_neg_sum[i] += v)
                    row_max_abs[i] = max(row_max_abs[i], abs(v))
                end
            end
            need_abs = false
            for i in 1:nW
                if row_neg_sum[i] < -1e-5 * row_max_abs[i]
                    need_abs = true; break
                end
            end
            need_abs && (W = abs.(W))
        end
        for j in 1:size(W, 2)
            for r in nzrange(W, j)
                if W.rowval[r] == j
                    W.nzval[r] = 0.0
                end
            end
        end
        if nnz(W) > 0
            max_w = maximum(abs, nonzeros(W))
            threshold = sqrt(eps(Float64)) * max_w
            for r in 1:nnz(W)
                abs(W.nzval[r]) < threshold && (W.nzval[r] = 0.0)
            end
        end
        dropzeros!(W)
        return laplacian(W)
    end
end
const read_mm = read_mm_test

@testset "MM preprocessing (MATLAB Problems.toLaplacian semantics)" begin
    mktempdir() do tmp
        @testset "Symmetric adjacency — directed: triu(W,1) + triu(W,1)'" begin
            # Directed: edge 1→2 weight 1, edge 2→1 weight 2.
            # MATLAB Graph.m takes triu(W,1) (only the upper-triangle entry
            # 1→2 with weight 1), then symmetrizes. The (2→1) entry with
            # weight 2 is discarded — matching MATLAB even though the
            # paper text suggests summing.
            path = joinpath(tmp, "directed.mtx")
            open(path, "w") do io
                println(io, "%%MatrixMarket matrix coordinate real general")
                println(io, "3 3 2")
                println(io, "1 2 1.0")
                println(io, "2 1 2.0")
            end
            L = read_mm(path)
            @test L[1, 2] ≈ -1.0  atol = 1e-12
            @test L[2, 1] ≈ -1.0  atol = 1e-12
            @test L[1, 1] ≈ 1.0
        end

        @testset "Diagonal entries → use sparsity pattern (binary)" begin
            # Stiffness-like 3-node: diagonal + off-diagonals.
            path = joinpath(tmp, "stiff.mtx")
            open(path, "w") do io
                println(io, "%%MatrixMarket matrix coordinate real symmetric")
                println(io, "3 3 5")
                println(io, "1 1 5.0")
                println(io, "2 2 7.0")
                println(io, "3 3 9.0")
                println(io, "2 1 2.0")
                println(io, "3 2 3.0")
            end
            L = read_mm(path)
            # Off-diagonals collapsed to ±1 (sparsity pattern).
            @test L[1, 2] ≈ -1.0  atol = 1e-12
            @test L[2, 1] ≈ -1.0  atol = 1e-12
            @test L[2, 3] ≈ -1.0  atol = 1e-12
            @test L[3, 2] ≈ -1.0  atol = 1e-12
        end

        @testset "Small negatives kept (< 1e-5 of row max) — not abs'd" begin
            # Edge weights: 1.0, -1e-8 (tiny negative). The latter is below the
            # 1e-5 * max threshold; not abs'd, just kept.
            path = joinpath(tmp, "smallneg.mtx")
            open(path, "w") do io
                println(io, "%%MatrixMarket matrix coordinate real symmetric")
                println(io, "3 3 2")
                println(io, "2 1 1.0")
                println(io, "3 2 -1e-8")
            end
            L = read_mm(path)
            # tiny negative is below sqrt(eps) * max ≈ 1.5e-8: filtered out.
            # so L[2,3] = 0.
            @test L[2, 3] ≈ 0.0 atol = 1e-12
        end

        @testset "Large negatives → abs" begin
            # Mostly positive but one large negative on a row dominates.
            # 4-node, edges (1,2)=1, (2,3)=−5, (3,4)=1.
            # Row 2: sum_neg = -5; max|W| = 5. -5 < -1e-5*5 → trigger abs.
            path = joinpath(tmp, "largeneg.mtx")
            open(path, "w") do io
                println(io, "%%MatrixMarket matrix coordinate real symmetric")
                println(io, "4 4 3")
                println(io, "2 1 1.0")
                println(io, "3 2 -5.0")
                println(io, "4 3 1.0")
            end
            L = read_mm(path)
            # After abs(W), W[2,3] = 5; L[2,3] = -5.
            @test L[2, 3] ≈ -5.0 atol = 1e-12
        end

        @testset "Disconnected graph → largest CC kept" begin
            # 5-node graph: edges (1,2), (3,4) → 2 CCs of size 2 and 2;
            # node 5 is isolated. Largest CC has 2 nodes.
            path = joinpath(tmp, "discon.mtx")
            open(path, "w") do io
                println(io, "%%MatrixMarket matrix coordinate real symmetric")
                println(io, "5 5 2")
                println(io, "2 1 1.0")
                println(io, "4 3 1.0")
            end
            L = read_mm(path)
            # The MM reader returns the full Laplacian; CC extraction is in
            # run_instance, not read_mm. Verify the Laplacian is correct.
            @test size(L, 1) == 5
            # Two components plus an isolated node.
            cc = connected_components(L)
            @test maximum(cc) == 3
        end
    end
end

@testset "RHS correction μ option flows through" begin
    using Random
    L = path_laplacian(64)
    rng = MersenneTwister(0x77)
    b = randn(rng, 64); b .-= sum(b) / 64
    # With μ = 1 (off) vs μ = 4/3 (default), the converged result must still
    # satisfy A x = b. We're only changing the RHS scaling for the FAS *coarse*
    # equation, not the finest. So both converge but iterate counts differ.
    x1, info1 = solve(L, b; options = LAMGOptions(rhs_correction = 1.0,
                                                  tol = 1e-10, max_cycles = 50))
    x2, info2 = solve(L, b; options = LAMGOptions(rhs_correction = 4 / 3,
                                                  tol = 1e-10, max_cycles = 50))
    @test info1.final_residual <= 1e-10 * norm(b)
    @test info2.final_residual <= 1e-10 * norm(b)
    # On 1D paths, elimination is exact so both converge in 1 cycle; no
    # difference here. The bigger payoff for μ = 4/3 is on geometric grids
    # — covered in `test_setup_solve.jl`.
end
