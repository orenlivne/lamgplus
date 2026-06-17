# Shared Matrix Market loader + LCC reduction for benchmark scripts.
# Extracted verbatim from competitor_benchmark.jl so all timing scripts
# preprocess instances identically. Returns (W_adj, L) where W_adj is the
# symmetric nonneg zero-diagonal adjacency and L = D - W_adj.

using LAMG, LinearAlgebra, SparseArrays

function read_mm_adj(path::AbstractString)
    open(path, "r") do io
        header = readline(io)
        tokens = split(lowercase(header))
        @assert tokens[1] == "%%matrixmarket" && tokens[2] == "matrix"
        format = tokens[3]; field = tokens[4]; symmetry = tokens[5]
        @assert format == "coordinate"
        line = readline(io)
        while startswith(strip(line), "%")
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
            i = parse(Int, parts[1]); j = parse(Int, parts[2])
            v = if field == "pattern"
                1.0
            elseif field == "complex"
                re = parse(Float64, parts[3]); im = parse(Float64, parts[4])
                sqrt(re * re + im * im)
            else
                parse(Float64, parts[3])
            end
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
                W.rowval[r] == j && (W.nzval[r] = 0.0)
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
        L = laplacian(W)
        return W, L
    end
end

function reduce_to_lcc(W::SparseMatrixCSC, L::SparseMatrixCSC)
    label = LAMG.connected_components(L)
    M = maximum(label)
    M == 1 && return W, L
    sizes = zeros(Int, M)
    for l in label
        sizes[l] += 1
    end
    biggest = argmax(sizes)
    retained = findall(==(biggest), label)
    return W[retained, retained], L[retained, retained]
end
