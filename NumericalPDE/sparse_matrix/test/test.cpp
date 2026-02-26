#include "../include/solvers.hpp"
#include "../include/sparse_matrix.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

using namespace std;

SparseMatrixCSR Poisson_matrix_maker(int n) {
    int size = n * n;
    
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<double> values;
    
    double h = 1.0 / (n + 1);
    double h2 = 1.0 / (h * h);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;  // Current node index
            
            // Diagonal element: 4/h²
            row_indices.push_back(idx);
            col_indices.push_back(idx);
            values.push_back(4.0 * h2);
            
            // Left neighbor (i, j-1)
            if (j > 0) {
                row_indices.push_back(idx);
                col_indices.push_back(idx - 1);
                values.push_back(-1.0 * h2);
            }
            
            // Right neighbor (i, j+1)
            if (j < n - 1) {
                row_indices.push_back(idx);
                col_indices.push_back(idx + 1);
                values.push_back(-1.0 * h2);
            }
            
            // Top neighbor (i-1, j)
            if (i > 0) {
                row_indices.push_back(idx);
                col_indices.push_back(idx - n);
                values.push_back(-1.0 * h2);
            }
            
            // Bottom neighbor (i+1, j)
            if (i < n - 1) {
                row_indices.push_back(idx);
                col_indices.push_back(idx + n);
                values.push_back(-1.0 * h2);
            }
        }
    }
    
    return SparseMatrixCSR(size, size, row_indices, col_indices, values);
}


std::vector<double> generate_random_vector(int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    
    std::vector<double> vec(size);
    for (int i = 0; i < size; i++) {
        vec[i] = dis(gen);
    }
    return vec;
}

/**
 * Test Jacobi solver performance for different matrix sizes
 * and write results to result.txt
 */
void run_jacobi_performance_test() {
    std::ofstream result_file("result.txt");
    
    if (!result_file.is_open()) {
        std::cerr << "Error: Cannot open result.txt for writing!" << std::endl;
        return;
    }
    
    // Test different grid sizes
    std::vector<int> grid_sizes = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    // std::vector<int> grid_sizes = {10};

    // Write header to result file
    result_file << "Jacobi Solver Performance Results for Poisson Equation\n";
    result_file << "======================================================\n";
    result_file << "Grid Size | Matrix Size | NNZ | Density(%) | ";
    result_file << "Construction(ms) | Jacobi(ms) | Residual\n";
    result_file << "----------|-------------|-----|------------|";
    result_file << "-----------------|------------|---------\n";
    
    
    for (int n : grid_sizes) {
        int matrix_size = n * n;
        
        // Measure matrix construction time
        auto start_construction = std::chrono::high_resolution_clock::now();
        SparseMatrixCSR A = Poisson_matrix_maker(n);

        // std::vector<double> dA = A.GetDiagonal();
        
        // result_file << dA[1] << dA[5] <<"\n";

        auto end_construction = std::chrono::high_resolution_clock::now();
        auto construction_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_construction - start_construction);
        
        // Generate random right-hand side vector
        std::vector<double> b = generate_random_vector(matrix_size);
        
        // Test Jacobi solver
        auto start_jacobi = std::chrono::high_resolution_clock::now();
        auto x_jacobi = JacobiSolver(A, b, 5000, 1e-6);
        auto end_jacobi = std::chrono::high_resolution_clock::now();
        auto jacobi_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_jacobi - start_jacobi);
        
        // Compute residual to verify solution quality
        double residual_norm = vector_dis(A.Multiply(x_jacobi), b);
        
        // Calculate matrix density
        double density = (1.0 - static_cast<double>(A.GetNNZ()) / (matrix_size * matrix_size)) * 100.0;
        
        // Output to console
        std::cout << n << "x" << n << "     | " << matrix_size << "x" << matrix_size 
                  << " | " << construction_time.count() << "ms        | " 
                  << jacobi_time.count() << "ms        | " 
                  << "N/A" << std::endl;  // Iteration count would need to be modified in JacobiSolver
        
        // Write detailed results to file
        result_file << n << "x" << n << " | " << matrix_size << "x" << matrix_size 
                   << " | " << A.GetNNZ() << " | " << std::fixed << std::setprecision(2) << density
                   << " | " << construction_time.count() 
                   << " | " << jacobi_time.count();
        
        // Add residual information
        result_file << " | " << std::scientific << residual_norm << "\n";
    }
    
    result_file.close();
    std::cout << "\nResults written to result.txt" << std::endl;
}


void run_CG_performance_test() {

    int size = 3;

    std::vector<int> row_indices = {0,0,1,1,2,2};
    std::vector<int> col_indices = {0,1,1,2,0,2};
    std::vector<double> values = {-1,1,-1,1,-1,1};

    SparseMatrixCSR A = SparseMatrixCSR(size, size, row_indices, col_indices, values);

    std::vector<double> b = {1, 1, 2};

    std::vector<double> x_0(size, 0.0);
    auto start_CG = std::chrono::high_resolution_clock::now();

    auto x_CG = GMRES(A, b, x_0);
    auto end_CG = std::chrono::high_resolution_clock::now();
    auto CG_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_CG- start_CG);

    double residual_norm = vector_dis(A.Multiply(x_CG), b);

    std::cout<< "residual norm: " << residual_norm << std::endl << "slover run time: "<< CG_time.count() << "ms"<< std::endl;
}

int main() {
    
    // run_jacobi_performance_test();
    
    run_CG_performance_test();

    return 0;
}