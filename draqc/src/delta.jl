# Global δ vector used by the quality measure (Napov–Notay 2017, eq. 15).

"""
    delta_vector(A) -> δ

Global vector δ = (U−D) D⁻¹ (L−D) 1, with U = triu(A), D = diag(A), L = tril(A)
(Napov–Notay eq. 15; δ_j = ((U−D)D⁻¹(L−D)1)_j). Computed with one sparse
matvec. For a graph Laplacian all entries are ≥ 0.
"""
function delta_vector(A::SparseMatrixCSC)
    n = size(A, 1)
    D = diag(A)
    Ls = tril(A, -1)            # L − D  (strict lower triangle)
    Us = triu(A, 1)             # U − D  (strict upper triangle)
    return Us * ((Ls * ones(n)) ./ D)
end
