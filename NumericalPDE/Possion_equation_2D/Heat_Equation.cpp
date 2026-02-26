#include "../sparse_matrix/include/solvers.hpp"
#include "../sparse_matrix/include/sparse_matrix.hpp"
#include "../2D_Domain/test_functions.hpp"


#include <iostream>
#include <math.h>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

using namespace std;

void fill_matrix(std::vector<int> &row_indices, std::vector<int> &col_indices, std::vector<double> &values,
    int k, int id_col, double v){

    row_indices.push_back(k);
    col_indices.push_back(id_col);
    values.push_back(v);
    return;
}

void result_output(grid_indexer grid_indexer_, std::vector<double> u, double t, double hx, double hy,
                   double x_min, double y_min, std::string file_name) {

    std::ofstream result_file(file_name);

    if (!result_file.is_open()) {
        std::cerr << "Error: Cannot open " << file_name << " for writing!" << std::endl;
        return;
    }

    double error_inf = 0.0;
    double error_2 = 0.0;

    int N_grid_points = grid_indexer_.N_grid_points;

    
    for (int k = 0; k < N_grid_points; k++) {
        double x = x_min + hx * grid_indexer_.x_grid[k], y = y_min + hy * grid_indexer_.y_grid[k];
        // result_file << x << " " << y << " " << u[k] << "\n";
        double u_ref = g_t(x, y, t);
        result_file << x << " " << y << " " << u[k] << " " << u_ref <<"\n";

        if ((x-1)*(x-1)+y*y>0.25) {
            error_inf = max(error_inf, std::abs(u_ref - u[k]));
            error_2 += (u_ref - u[k])*(u_ref - u[k]);

        }

    }

    result_file << "Maximum Error: " << error_inf << std::endl;
    result_file << "L2 Error: " << std::sqrt(error_2/N_grid_points) << std::endl;

    std::cout << "Maximum Error: " << error_inf << std::endl;
    std::cout << "L2 Error: " << std::sqrt(error_2/N_grid_points) << std::endl;

    result_file.close();

    return;
}


void explicit_solver(int N_x, int N_y, double dt, int N_t){

    double hx = 2.0/(N_x-1), hy = 4.0/(N_y-1);
    if( dt * (1/(hx*hx)+1/(hy*hy) ) > 0.5*RATE){
        std::cout << "Warning: unstable explicit scheme! " <<std::endl;
    }
    grid_indexer grid_indexer_ = grid_indexer_init(0.0, 2.0, -2.0, 2.0, N_x, N_y);
    double x_min = 0.0, y_min = -2.0;
    int N_grids = grid_indexer_.N_grid_points;
    std::vector<int> line_ind = grid_indexer_.line_ind;
    std::vector<double> u_0(N_grids, 0.0);
    std::vector<double> u_new(N_grids, 0.0);
 
    for(int k=0; k < N_grids; k++){
        int i = grid_indexer_.x_grid[k], j = grid_indexer_.y_grid[k];
        double x = x_min + i*hx, y = y_min + j*hy;
        u_0[k] = u0(x, y);
        // u_0[k] = 0;
    }
    
    for(int iter=1; iter <= N_t; iter++){
        double t_new = dt*iter;
        if(iter% max((N_t/10),1)==0){
            std::cout<<"Running explicit scheme to solve time t="<< t_new <<"\n";
        }
        double t_0 = t_new - dt;
        for(int k=0; k<N_grids; k++){
            
            int i = grid_indexer_.x_grid[k], j = grid_indexer_.y_grid[k];
            double x = x_min + i*hx, y = y_min + j*hy;

            std::vector<double> hh = grid_indexer_.h_news[k];
            double h_n, h_e, h_s, h_w;
            double u_n, u_e, u_s, u_w;
            if(hh[0]>hy*RATE){ // the North point is inner point
                h_n = hy;
                u_n = u_0[k+1];
            }
            else{
                h_n = hh[0];
                u_n = g_t(x, y+h_n, t_0);
            }

            if(hh[2]>hy*RATE){ // the Sorth point is inner point
                h_s = hy;
                u_s = u_0[k-1];
            }
            else{
                h_s = hh[2];
                u_s = g_t(x, y-h_s, t_0);
            }

            if(hh[1]>hx*RATE){ // the East point is inner point
                h_e = hx;
                u_e = u_0[ line_ind[(i+1)*N_y+j] ];
            }
            else{
                h_e = hh[1];
                u_e = g_t(x+h_e, y, t_0);
            }

            if(hh[3]>hx*RATE){ // the West point is inner point
                h_w = hx;
                u_w = u_0[ line_ind[(i-1)*N_y+j] ];
            }
            else{
                h_w = hh[3];
                u_w = g_t(x-h_w, y, t_0);
            }

            double delta = 0;
            if ((x-1)*(x-1) + y*y < (RATE-1)*(hx*hx+hy*hy)){
                // Delta函数项
                delta = rho(t_0)/(hx*hy);
            }
            double u_p = u_0[k];

            // Laplace 算子项
            double dxx = 2*( (u_e - u_p)/h_e - (u_p - u_w)/h_w )/(h_e + h_w);
            double dyy = 2*( (u_n - u_p)/h_w - (u_p - u_s)/h_s )/(h_n + h_s);

            u_new[k] = u_p + dt*(dxx + dyy + delta);


        }

        u_0 = u_new;
    }
    
    result_output(grid_indexer_, u_0, N_t*dt, hx, hy, 0.0, -2.0, "results/explicit_solver.txt");
    return;

}

void implicit_solver(int N_x, int N_y, double dt, int N_t){
    
    double hx = 2.0/(N_x-1), hy = 4.0/(N_y-1);
    grid_indexer grid_indexer_ = grid_indexer_init(0.0, 2.0, -2.0, 2.0, N_x, N_y);
    double x_min = 0.0, y_min = -2.0;
    int N_grids = grid_indexer_.N_grid_points;
    std::vector<int> line_ind = grid_indexer_.line_ind;
    std::vector<double> u_0(N_grids, 0.0);
    std::vector<double> u_new(N_grids, 0.0);


    // 初始化

    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<double> values;

    for(int k=0; k < N_grids; k++){  
        int i = grid_indexer_.x_grid[k], j = grid_indexer_.y_grid[k];
        double x = x_min + i*hx, y = y_min + j*hy;
        u_0[k] = u0(x, y);
        std::vector<double> hh = grid_indexer_.h_news[k];
        double h_n, h_e, h_s, h_w;

        if(hh[0]>hy*RATE){ // the North point is inner point
            h_n = hy;
            row_indices.push_back(k);
            col_indices.push_back(k+1);
            values.push_back( -dt/(2*hy*hy));
        }
        else{
            h_n = hh[0];
        }

        if(hh[2]>hy*RATE){ // the Sorth point is inner point
            h_s = hy;
            row_indices.push_back(k);
            col_indices.push_back(k-1);
            values.push_back( -dt/(2*hy*hy));
        }
        else{
            h_s = hh[2];
        }

        if(hh[1]>hx*RATE){ // the East point is inner point
            h_e = hx;
            row_indices.push_back(k);
            col_indices.push_back(line_ind[(i+1)*N_y+j]);
            values.push_back( -dt/(2*hx*hx));
        }
        else{
            h_e = hh[1];
        }

        if(hh[3]>hx*RATE){ // the West point is inner point
            h_w = hx;
            row_indices.push_back(k);
            col_indices.push_back(line_ind[(i-1)*N_y+j]);
            values.push_back( -dt/(2*hx*hx));
        }
        else{
            h_w = hh[3];
        }
        row_indices.push_back(k);
        col_indices.push_back(k);
        values.push_back( 1+ dt*(1.0/h_n + 1.0/h_s)/(2*hy) + dt*(1.0/h_e + 1.0/h_w)/(2*h_s) );
    }
    

    SparseMatrixCSR M = SparseMatrixCSR(N_grids, N_grids, row_indices, col_indices, values);

    for(int iter=1; iter <= N_t; iter++){
        double t_new = dt*iter;
        double t_0 = t_new - dt;

        if(iter% max(1,(N_t/10))==0){
            std::cout<<"Running implicit scheme to solve time t="<< t_new <<"\n";
        }

        std::vector<double> b(N_grids, 0);

        for(int k=0; k<N_grids; k++){
            
            int i = grid_indexer_.x_grid[k], j = grid_indexer_.y_grid[k];
            double x = x_min + i*hx, y = y_min + j*hy;

            std::vector<double> hh = grid_indexer_.h_news[k];
            double h_n, h_e, h_s, h_w;

            if(hh[0]>hy*RATE){ // the North point is inner point
                h_n = hy;
                b[k] += dt * u_0[k+1]/(2*hy*hy);
            }
            else{
                h_n = hh[0];
                b[k]+= dt * (g_t(x,y+h_n,t_new) + g_t(x,y+h_n,t_0)) / (2*hy*hh[0]);
            }

            if(hh[2]>hy*RATE){ // the Sorth point is inner point
                h_s = hy;
                b[k] += dt * u_0[k-1]/(2*hy*hy);
            }
            else{
                h_s = hh[2];
                b[k]+= dt * (g_t(x,y-h_s,t_new)+g_t(x,y-h_s,t_0)) / (2*hy*hh[2]);
            }

            if(hh[1]>hx*RATE){ // the East point is inner point
                h_e = hx;
                b[k] += dt * u_0[line_ind[(i+1)*N_y+j]]/(2*hx*hx);
            }
            else{
                h_e = hh[1];
                b[k]+= dt * (g_t(x+h_e,y,t_new)+g_t(x+h_e,y,t_new)) / (2*hx*hh[1]);
            }

            if(hh[3]>hx*RATE){ // the West point is inner point
                h_w = hx;
                b[k] += dt * u_0[line_ind[(i-1)*N_y+j]]/(2*hx*hx);
            }
            else{
                h_w = hh[3];
                b[k]+= dt * (g_t(x-h_w,y,t_new)+g_t(x-h_w,y,t_new)) / (2*hx*hh[3]);
            }

            if ((x-1)*(x-1) + y*y < (RATE-1)*(hx*hx+hy*hy)){
                // Delta函数项
                b[k] += dt * (rho(t_0)+rho(t_new))/(2*hx*hy);
            } 

            b[k] += u_0[k] * (1 - dt*(1.0/h_n + 1.0/h_s)/(2*hy) - dt*(1.0/h_e + 1.0/h_w)/(2*h_s));
        }

        u_new = CGSolver(M, b);
        u_0 = u_new;
    }
    
    result_output(grid_indexer_, u_0, N_t*dt, hx, hy, 0.0, -2.0, "results/implicit_solver.txt");
    return;

}

int main(){

    int N = 40;
    int N_x = 2*N+1, N_y = 4*N+1;

    /*
        implicit_solver
    */ 
    double h = 2.0/(N_x-1);
    double T = 2*PI;
    int N_t = N;
    double dt = T/N_t;

    std::cout<< "Solve heat equation with h= "<< h << ",dt = " << dt << "\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    implicit_solver(N_x, N_y, dt, N_t);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto run_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time- start_time);


    std::cout<< "Run time: "<< run_time.count() << " ms"<< std::endl;


    /*
        explicit_solver
    */ 
    // double h = 2.0/(N_x-1);
    // double T = 2*PI;
    // double dt = h*h/5;
    // int N_t = round(T/dt);

    // std::cout<< "Solve heat equation with h= "<< h << ",dt = " << dt << "\n";
    // auto start_time = std::chrono::high_resolution_clock::now();
    // explicit_solver(N_x, N_y, dt, N_t);
    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto run_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    //         end_time- start_time);


    // std::cout<< "Run time: "<< run_time.count() << " ms"<< std::endl;

    system("pause");
    return 0;
}
