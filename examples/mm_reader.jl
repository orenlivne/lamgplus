# examples/mm_reader.jl
# Minimal MatrixMarket (.mtx) reader for graph Laplacians, shared by the examples.
# Returns the weighted adjacency W and its Laplacian L = D - W. Handles
# pattern/real/complex fields, symmetric/general symmetry, negative weights
# (takes |w| when a row's negative off-diagonals would break diagonal dominance),
# and drops the diagonal. `reduce_to_lcc` restricts to the largest connected
# component so the system is compatible (single null vector).
using SparseArrays, LinearAlgebra
import LAMG: laplacian

function read_mm_adj(path::AbstractString)
    open(path, "r") do io
        header = readline(io)
        tokens = split(lowercase(header))
        @assert tokens[1] == "%%matrixmarket" && tokens[2] == "matrix"
        format = tokens[3]; field = tokens[4]; symmetry = tokens[5]
        @assert format == "coordinate"
        line = readline(io)
        while startswith(strip(line), "%"); line = readline(io); end
        nrows, ncols, nentries = parse.(Int, split(line))
        rows = Vector{Int}(undef, 2 * nentries); cols = Vector{Int}(undef, 2 * nentries)
        vals = Vector{Float64}(undef, 2 * nentries); k = 0
        for _ in 1:nentries
            parts = split(readline(io))
            i = parse(Int, parts[1]); j = parse(Int, parts[2])
            v = field == "pattern" ? 1.0 :
                field == "complex" ? sqrt(parse(Float64, parts[3])^2 + parse(Float64, parts[4])^2) :
                parse(Float64, parts[3])
            k += 1; rows[k] = i; cols[k] = j; vals[k] = v
            if symmetry in ("symmetric", "hermitian", "skew-symmetric") && i != j
                k += 1; rows[k] = j; cols[k] = i; vals[k] = v
            end
        end
        W_raw = sparse(rows[1:k], cols[1:k], vals[1:k], nrows, ncols)
        upper = triu(W_raw, 1); W = upper + sparse(transpose(upper))
        had_diagonal = any(!iszero, diag(W_raw))
        if had_diagonal
            ii, jj, _ = findnz(W); W = sparse(ii, jj, ones(Float64, length(ii)), size(W)...)
        end
        if !had_diagonal && nnz(W) > 0 && minimum(nonzeros(W)) < 0
            rows_W = rowvals(W); vals_W = nonzeros(W); nW = size(W, 1)
            row_neg_sum = zeros(nW); row_max_abs = zeros(nW)
            for j in 1:nW, r in nzrange(W, j)
                i = rows_W[r]; v = vals_W[r]
                v < 0 && (row_neg_sum[i] += v); row_max_abs[i] = max(row_max_abs[i], abs(v))
            end
            any(row_neg_sum[i] < -1e-5 * row_max_abs[i] for i in 1:nW) && (W = abs.(W))
        end
        for j in 1:size(W, 2), r in nzrange(W, j)
            W.rowval[r] == j && (W.nzval[r] = 0.0)
        end
        if nnz(W) > 0
            thr = sqrt(eps(Float64)) * maximum(abs, nonzeros(W))
            for r in 1:nnz(W); abs(W.nzval[r]) < thr && (W.nzval[r] = 0.0); end
        end
        dropzeros!(W)
        return W, laplacian(W)
    end
end

function reduce_to_lcc(W::SparseMatrixCSC, L::SparseMatrixCSC)
    label = LAMG.connected_components(L); M = maximum(label)
    M == 1 && return W, L
    sizes = zeros(Int, M); for l in label; sizes[l] += 1; end
    keep = findall(==(argmax(sizes)), label)
    return W[keep, keep], L[keep, keep]
end
