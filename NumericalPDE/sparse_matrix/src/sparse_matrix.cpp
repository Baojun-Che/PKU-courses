#include "../include/sparse_matrix.hpp"
#include <iostream>
#include <algorithm>
#include <stdexcept>



SparseMatrixCSR::SparseMatrixCSR(int rows, int cols) 
    : rows_(rows), cols_(cols), nnz_(0) {
    row_ptrs_.resize(rows + 1, 0);
}

SparseMatrixCSR::SparseMatrixCSR(int rows, int cols, 
               const std::vector<int>& row_indices,
               const std::vector<int>& col_indices,
               const std::vector<double>& values) {
    ConstructFromCOO(rows, cols, row_indices, col_indices, values);
}

void SparseMatrixCSR::ConstructFromCOO(int rows, int cols,
                     const std::vector<int>& row_indices,
                     const std::vector<int>& col_indices,
                     const std::vector<double>& values) {
                        
    if (row_indices.size() != col_indices.size() || 
        row_indices.size() != values.size()) {
        throw std::invalid_argument("Illegal input: COO vectors must have same size");
    }
    
    rows_ = rows;
    cols_ = cols;
    nnz_ = values.size();
    
    // Step 1: Counting number of nonzero elements of each row
    row_ptrs_.resize(rows + 1, 0);
    for (int i = 0; i < nnz_; i++) {
        if (row_indices[i] < 0 || row_indices[i] >= rows) {
            throw std::out_of_range("Row index out of range");
        }
        row_ptrs_[row_indices[i] + 1]++;
    }
    
    // Step 2: Computing row_ptrs_ via prefix sum
    for (int i = 0; i < rows; i++) {
        row_ptrs_[i + 1] += row_ptrs_[i];
    }
    
    // Step 3: Filling values_, col_indices_
    values_.resize(nnz_);
    col_indices_.resize(nnz_);
    std::vector<int> current_pos(rows, 0); 
    for (int i = 0; i < nnz_; i++) {
        int row = row_indices[i];
        int pos = row_ptrs_[row] + current_pos[row];
        values_[pos] = values[i];
        col_indices_[pos] = col_indices[i];
        current_pos[row]++;
    }
}

// y = A * x
std::vector<double> SparseMatrixCSR::Multiply(const std::vector<double>& x) const {
    if ( static_cast<int>(x.size()) != cols_) {
        throw std::invalid_argument("Vector size must match matrix columns");
    }
    
    std::vector<double> y(rows_, 0.0);
    
    for (int i = 0; i < rows_; i++) {
        for (int j = row_ptrs_[i]; j < row_ptrs_[i + 1]; j++) {
            y[i] += values_[j] * x[col_indices_[j]];
        }
    }
    
    return y;
}

SparseMatrixCSR SparseMatrixCSR::GetUpperTriangular(bool include_diagonal) const {
    std::vector<int> row_indices, col_indices;
    std::vector<double> values;
    
    for (int i = 0; i < rows_; i++) {
        for (int j = row_ptrs_[i]; j < row_ptrs_[i + 1]; j++) {
            if (col_indices_[j] > i  || (include_diagonal && col_indices_[j] == i)) { 
                row_indices.push_back(i);
                col_indices.push_back(col_indices_[j]);
                values.push_back(values_[j]);
            }
        }
    }
    
    return SparseMatrixCSR(rows_, cols_, row_indices, col_indices, values);
}



SparseMatrixCSR SparseMatrixCSR::GetLowerTriangular(bool include_diagonal) const {
        std::vector<int> row_indices, col_indices;
        std::vector<double> values;
        
        for (int i = 0; i < rows_; i++) {
            for (int j = row_ptrs_[i]; j < row_ptrs_[i + 1]; j++) {
                if (col_indices_[j] < i || (include_diagonal && col_indices_[j] == i)) {
                    row_indices.push_back(i);
                    col_indices.push_back(col_indices_[j]);
                    values.push_back(values_[j]);
                }
            }
        }
        
        return SparseMatrixCSR(rows_, cols_, row_indices, col_indices, values);
    }
    
    // Get diagonal (as a vector)
std::vector<double> SparseMatrixCSR::GetDiagonal() const {
        std::vector<double> diag(rows_, 0.0);
        
        for (int i = 0; i < rows_; i++) {
            for (int j = row_ptrs_[i]; j < row_ptrs_[i + 1]; j++) {
                if (col_indices_[j] == i) {
                    diag[i] = values_[j];
                    break;
                }
            }
        }
        
        return diag;
    }
    
    // Get diagonal (as a CSR matrix)
    SparseMatrixCSR SparseMatrixCSR::GetDiagonalMatrix() const {
        std::vector<double> diag = GetDiagonal();
        std::vector<int> row_indices, col_indices;
        std::vector<double> values;
        
        for (int i = 0; i < rows_; i++) {
            if (diag[i] != 0.0) {
                row_indices.push_back(i);
                col_indices.push_back(i);
                values.push_back(diag[i]);
            }
        }
        
        return SparseMatrixCSR(rows_, cols_, row_indices, col_indices, values);
    }
    
    // 打印矩阵（用于调试）
    void SparseMatrixCSR::PrintMatrix() const {
        std::cout << "CSR Sparse Matrix " << rows_ << "x" << cols_ 
                  << " with " << nnz_ << " non-zero elements\n";
        
        std::cout << "Values: ";
        for (double v : values_) std::cout << v << " ";
        std::cout << "\nCol indices: ";
        for (int c : col_indices_) std::cout << c << " ";
        std::cout << "\nRow pointers: ";
        for (int r : row_ptrs_) std::cout << r << " ";
        std::cout << std::endl;
    }
    
    // 转换为稠密矩阵（用于验证）
    std::vector<std::vector<double>> SparseMatrixCSR::ToDenseMatrix() const {
        std::vector<std::vector<double>> dense(rows_, std::vector<double>(cols_, 0.0));
        
        for (int i = 0; i < rows_; i++) {
            for (int j = row_ptrs_[i]; j < row_ptrs_[i + 1]; j++) {
                dense[i][col_indices_[j]] = values_[j];
            }
        }
        
        return dense;
    }