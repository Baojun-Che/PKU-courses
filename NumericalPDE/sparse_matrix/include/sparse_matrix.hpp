#ifndef SPARSE_MATRIX_HPP
#define SPARSE_MATRIX_HPP

#include <vector>

class SparseMatrixCSR {
private:
    int rows_;
    int cols_;
    int nnz_;
    std::vector<double> values_;
    std::vector<int> col_indices_;
    std::vector<int> row_ptrs_;

public:
    // Constrct CSR matrix
    SparseMatrixCSR(int rows, int cols);
    SparseMatrixCSR(int rows, int cols, 
                   const std::vector<int>& row_indices,
                   const std::vector<int>& col_indices,
                   const std::vector<double>& values);
    
    // Basic elements
    int GetRows() const { return rows_; }
    int GetCols() const { return cols_; }
    int GetNNZ() const { return nnz_; }
    
    // Core Functions
    void ConstructFromCOO(int rows, int cols,
                         const std::vector<int>& row_indices,
                         const std::vector<int>& col_indices,
                         const std::vector<double>& values);
    
    std::vector<double> Multiply(const std::vector<double>& x) const;
    SparseMatrixCSR GetUpperTriangular(bool include_diagonal = true) const;
    SparseMatrixCSR GetLowerTriangular(bool include_diagonal = true) const;
    std::vector<double> GetDiagonal() const;
    SparseMatrixCSR GetDiagonalMatrix() const;
    
    void PrintMatrix() const;
    std::vector<std::vector<double>> ToDenseMatrix() const;
    
};

#endif