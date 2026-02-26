#include "../include/solvers.hpp"
#include <cmath>
#include <iostream>
#include <utility>

std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size for addition");
    }
    
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

std::vector<double> operator-(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size for subtraction");
    }
    
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

std::vector<double> operator*(double scalar, const std::vector<double>& vec) {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = scalar * vec[i];
    }
    return result;
}

double vector_dis(const std::vector<double>& a, const std::vector<double>& b){
    if (b.size() != a.size()) {
            throw std::invalid_argument("Vectors must have same size for calculating distance");
        }
    double dis = 0.0;
    for (int i = 0; i < static_cast<int>(a.size()); i++) {
            dis += (b[i] - a[i]) * (b[i] - a[i]);
        }
    return sqrt(dis);
}

double vector_dot(const std::vector<double>& a, const std::vector<double>& b){
    if (b.size() != a.size()) {
            throw std::invalid_argument("Vectors must have same size for inner dot");
        }
    double ans = 0.0;
    for (int i = 0; i < static_cast<int>(a.size()); i++) {
            ans += a[i]*b[i];
        }
    return ans;
}

std::vector<double> JacobiSolver(const SparseMatrixCSR& A, 
                                const std::vector<double>& b,
                                const std::vector<double>& x0,
                                int max_iter, 
                                double tol) {
    int cols = A.GetRows();
    if (static_cast<int>(b.size()) != cols) {
            throw std::invalid_argument("Vectors size must match matrix columns");
        }

    std::vector<double> x = x0;  // initial guess
    std::vector<double> x_new(cols, 0.0);
        
    SparseMatrixCSR D = A.GetDiagonalMatrix();
    std::vector<double> diag = D.GetDiagonal();
    
    SparseMatrixCSR L = A.GetLowerTriangular(false);  // strict lower triangular
    SparseMatrixCSR U = A.GetUpperTriangular(false);
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Jacobi iteration: x_new = D^{-1}(b - (L+U)x)

        std::vector<double> Ax = A.Multiply(x);
        double residual = vector_dis(Ax, b);

        if (residual < tol) {
            std::cout << "Jacobi converged after " << iter + 1 << " iterations\n";
            break;
        }

        if ((iter + 1) % 500 == 0) {
            std::cout << "Jacobi iteration " << iter + 1 << ", residual: " << residual << "\n";
        }

        std::vector<double> Lx = L.Multiply(x);
        std::vector<double> Ux = U.Multiply(x);

        for (int i = 0; i < cols; i++) {
            if (diag[i] != 0.0) {
                x_new[i] = (b[i]-Lx[i]-Ux[i]) / diag[i];
            }

            if (std::abs(diag[i]) < 1e-10) {
            std::cout << "Warning: low value in diagonal " << i << ", value=" << diag[i] << std::endl;
            }
        }        
        x = x_new; 
    }
    
    return x;
}

std::vector<double> CGSolver(const SparseMatrixCSR& A,
                                     const std::vector<double>& b,
                                     int max_iter,
                                     double tol) {
    int cols = A.GetRows();
    if (static_cast<int>(b.size()) != cols) {
            throw std::invalid_argument("Vector size must match matrix columns");
        }

    std::vector<double> x(cols, 0.0);
    std::vector<double> r = b;
    std::vector<double> p = r;
    
    for (int iter = 0; iter < max_iter; iter++) {

        double residual = sqrt(vector_dot(r, r));

        if (residual < tol) {
            std::cout << "CG converged after " << iter + 1 << " iterations\n";
            break;
        }

        if ((iter + 1) % 500 == 0) {
            std::cout << "CG iteration " << iter + 1 << ", residual: " << residual << "\n";
        }

        std::vector<double> Ap = A.Multiply(p);
        double alpha = vector_dot(r,p) / vector_dot(Ap,p); 
        x = x + alpha*p;
        r = r - alpha*Ap;
        double beta = - vector_dot(r,Ap) / vector_dot(Ap,p); 
        p = r + beta*p;
    }

    return x;

}

std::vector<double> GMRES(const SparseMatrixCSR& A, const std::vector<double>& b, const std::vector<double>& x0,
                   int restart, int max_restarts, double tol) {
    
    std::vector<double> x = x0;
    int total_iterations = 0;
    int total_restarts = 0;
    
    for (int restart_count = 0; restart_count < max_restarts; ++restart_count) {

        std::vector<double> r = b - A.Multiply(x);
        double r_norm = std::sqrt(vector_dot(r, r));
        
        if (restart_count%5==0){
            std::cout << "restart_count= "<< restart_count <<", residual="<<r_norm<<"\n";
        }
        
        if (r_norm < tol) {
            std::cout << "GMRES converged after " << total_iterations 
                      << " iterations (" << total_restarts << " restarts), "
                      << "final residual: " << r_norm << std::endl;
            return x;
        }
        
        // Arnoldi过程初始化
        int n = b.size();
        std::vector<std::vector<double>> V(restart + 1, std::vector<double>(n, 0.0));
        V[0] = (1.0 / r_norm) * r;  
        std::vector<std::vector<double>> H(restart + 1, std::vector<double>(restart, 0.0));
        std::vector<double> cs(restart), sn(restart);
        std::vector<double> g(restart + 1, 0.0);
        g[0] = r_norm;
        
        // 内部迭代
        int j = 0;
        for (; j < restart; ++j) {
            // Arnoldi步骤
            std::vector<double> w = A.Multiply(V[j]);
            
            // 修正的Gram-Schmidt
            for (int i = 0; i <= j; ++i) {
                H[i][j] = vector_dot(w, V[i]);
                w = w - H[i][j] * V[i] ; 
            }
            
            H[j+1][j] = std::sqrt(vector_dot(w, w));
            
            if (H[j+1][j] < 1e-14) break;
            
            V[j+1] = (1.0 / H[j+1][j]) * w; 
            
            // 应用之前的Givens旋转
            for (int i = 0; i < j; ++i) {
                double temp = cs[i] * H[i][j] + sn[i] * H[i+1][j];
                H[i+1][j] = -sn[i] * H[i][j] + cs[i] * H[i+1][j];
                H[i][j] = temp;
            }
            
            // 计算新的Givens旋转
            double a = H[j][j], b_val = H[j+1][j];
            double rho = std::sqrt(a*a + b_val*b_val);
            if (rho == 0.0) {
                cs[j] = 1.0;
                sn[j] = 0.0;
            } else {
                cs[j] = a / rho;
                sn[j] = b_val / rho;
            }
            
            // 应用Givens旋转
            H[j][j] = cs[j] * H[j][j] + sn[j] * H[j+1][j];
            H[j+1][j] = 0.0;
            
            // 更新右端项
            g[j+1] = -sn[j] * g[j];
            g[j] = cs[j] * g[j];
            
            // 检查收敛
            if (std::abs(g[j+1]) < tol) {
                j++;
                break;
            }
        }
        
        total_iterations += j;
        total_restarts = restart_count + 1;
        
        // 回代求解
        std::vector<double> y(j, 0.0);
        for (int i = j - 1; i >= 0; --i) {
            y[i] = g[i];
            for (int k = i + 1; k < j; ++k) {
                y[i] -= H[i][k] * y[k];
            }
            y[i] /= H[i][i];
        }
        
        // 更新解
        for (int i = 0; i < j; ++i) {
            x = x +  y[i] * V[i];  
        }
        
        // 检查内部迭代是否收敛
        std::vector<double> new_r = b - A.Multiply(x);
        double new_r_norm = std::sqrt(vector_dot(new_r, new_r));
        
        if (new_r_norm < tol) {
            std::cout << "GMRES converged after " << total_iterations 
                      << " iterations (" << total_restarts << " restarts), "
                      << "final residual: " << new_r_norm << std::endl;
            return x;
        }
    }
    
    // 计算最终残差
    std::vector<double> final_r = b - A.Multiply(x);
    double final_r_norm = std::sqrt(vector_dot(final_r, final_r));
    
    std::cout << "GMRES finished (may not converge) after " << total_iterations 
              << " iterations (" << total_restarts << " restarts), "
              << "final residual: " << final_r_norm << std::endl;
    
    return x;
}