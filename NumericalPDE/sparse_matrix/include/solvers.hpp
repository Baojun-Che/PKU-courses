#ifndef SOLVERS_HPP
#define SOLVERS_HPP

#include "sparse_matrix.hpp"
#include <vector>

std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b);

std::vector<double> operator-(const std::vector<double>& a, const std::vector<double>& b);

std::vector<double> operator*(double scalar, const std::vector<double>& vec);

double vector_dis(const std::vector<double>& a, const std::vector<double>& b);

double vector_dot(const std::vector<double>& a, const std::vector<double>& b);


// Jacobi solver
std::vector<double> JacobiSolver(const SparseMatrixCSR& A, 
                                const std::vector<double>& b,
                                const std::vector<double>& x0,
                                int max_iter = 10000, 
                                double tol = 1e-8);

// CG solver
std::vector<double> CGSolver(const SparseMatrixCSR& A,
                                     const std::vector<double>& b,
                                     int max_iter = 3000,
                                     double tol = 1e-8);

// GMRES solver
std::vector<double> GMRES(const SparseMatrixCSR& A, const std::vector<double>& b, const std::vector<double>& x0,
                   int restart = 100, int max_restarts = 100, double tol = 1e-8);
#endif