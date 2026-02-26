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


void result_output(grid_indexer grid_indexer_, std::vector<double> u, double hx, double hy,
                   double x_min, double y_min, std::string file_name, bool demean = true) {
    std::ofstream result_file(file_name);

    if (!result_file.is_open()) {
        std::cerr << "Error: Cannot open " << file_name << " for writing!" << std::endl;
        return;
    }

    double error = 0.0;
    int N_grid_points = grid_indexer_.N_grid_points;

    if (demean) {
        double u_sum = 0;
        double A = 0, B = 0;

        for (int k = 0; k < N_grid_points; k++)
            u_sum += u[k];
        for (int k = 0; k < N_grid_points; k++)
            u[k] -= u_sum / N_grid_points;

        for (int k = 0; k < N_grid_points; k++) {
            double x = x_min + hx * grid_indexer_.x_grid[k], y = y_min + hy * grid_indexer_.y_grid[k];
            result_file << x << " " << y << " " << u[k] << "\n";
            double e = u[k] - g(x, y);
            if (e < A) A = e;
            if (e > B) B = e;
        }
        error = (B - A) / 2;
    } else {
        for (int k = 0; k < N_grid_points; k++) {
            double x = x_min + hx * grid_indexer_.x_grid[k], y = y_min + hy * grid_indexer_.y_grid[k];
            result_file << x << " " << y << " " << u[k] << "\n";
            if (std::abs(g(x, y) - u[k]) > error) {
                error = std::abs(g(x, y) - u[k]);
            }
        }
    }

    result_file.close();
    std::cout << "Maximum Error: " << error << std::endl;
}


void Dirichlet_Possion_solver(double x_min, double x_max, double y_min, double y_max, int N_x, int N_y){

    double hx = (x_max-x_min)/(N_x-1), hy=(y_max-y_min)/(N_y-1);

    grid_indexer grid_indexer_ = grid_indexer_init(x_min, x_max, y_min, y_max, N_x, N_y);
    int N_grids = grid_indexer_.N_grid_points;
    std::vector<int> line_ind = grid_indexer_.line_ind;

    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<double> values;
    std::vector<double> b(N_grids, 0);

    for (int k=0; k<N_grids; k++){
        
        int i = grid_indexer_.x_grid[k], j = grid_indexer_.y_grid[k];
        double x = x_min + i*hx, y = y_min + j*hy;


        b[k] = f(x,y);
        if ( k!= line_ind[i*N_y+j] ) {
            throw std::out_of_range("Wrong index!");
        }

        std::vector<double> hh = grid_indexer_.h_news[k];
        double h_n, h_e, h_s, h_w;
        if(hh[0]>hy*RATE){ // the North point is inner point
            h_n = hy;
            row_indices.push_back(k);
            col_indices.push_back(k+1);
            values.push_back( -1.0/(hy*hy));
        }
        else{
            h_n = hh[0];
            b[k]+= g(x,y+h_n)/(hy*hh[0]);
        }

        if(hh[2]>hy*RATE){ // the Sorth point is inner point
            h_s = hy;
            row_indices.push_back(k);
            col_indices.push_back(k-1);
            values.push_back( -1.0/(hy*hy));
        }
        else{
            h_s = hh[2];
            b[k]+= g(x,y-h_s)/(hy*hh[2]);
        }

        if(hh[1]>hx*RATE){ // the East point is inner point
            h_e = hx;
            row_indices.push_back(k);
            col_indices.push_back(line_ind[(i+1)*N_y+j]);
            values.push_back( -1.0/(hx*hx));
        }
        else{
            h_e = hh[1];
            b[k]+= g(x+h_e,y)/(hx*hh[1]);
        }

        if(hh[3]>hx*RATE){ // the West point is inner point
            h_w = hx;
            row_indices.push_back(k);
            col_indices.push_back(line_ind[(i-1)*N_y+j]);
            values.push_back( -1.0/(hx*hx));
        }
        else{
            h_w = hh[3];
            b[k]+= g(x-h_w,y)/(hx*hh[3]);
        }

        row_indices.push_back(k);
        col_indices.push_back(k);
        values.push_back( (1.0/h_n + 1.0/h_s)/hy + (1.0/h_e + 1.0/h_w)/hx );
    }

    SparseMatrixCSR M = SparseMatrixCSR(N_grids, N_grids, row_indices, col_indices, values);

    // std::vector<std::vector<double>> dense = M.ToDenseMatrix();
    // for(int i =0; i<N_grids; i++){
    //     for(int j =0; j<N_grids; j++){
    //         if (abs(dense[i][j]- dense[j][i]) > 0.01) 
    //             std::cout << i << ", " <<j << " "<< abs(dense[i][j]- dense[j][i]) <<"\n";
    //     }
    // }
    
    std::cout<<N_grids<<"\n";

    std::vector<double> x_0(N_grids, 0.0);
    std::vector<double> u = CGSolver(M, b);
    // std::vector<double> u = GMRES(M, b, x_0);
    
    result_output(grid_indexer_, u, hx, hy, x_min, y_min, "result_1");

    return;
}


// 0 for inner point; 1 otherwise

void fill_matrix(std::vector<int> &row_indices, std::vector<int> &col_indices, std::vector<double> &values,
    int k, int id_col, double v){

    row_indices.push_back(k);
    col_indices.push_back(id_col);
    values.push_back(v);
    return;
}

void Neumann_Possion_solver(double x_min, double x_max, double y_min, double y_max, int N_x, int N_y){

    double hx = (x_max-x_min)/(N_x-1), hy=(y_max-y_min)/(N_y-1);

    grid_indexer grid_indexer_ = grid_indexer_init(x_min, x_max, y_min, y_max, N_x, N_y);
    int N_grids = grid_indexer_.N_grid_points;
    std::vector<int> line_ind = grid_indexer_.line_ind;

    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<double> values;
    std::vector<double> b(N_grids, 0);

    for (int k=0; k<N_grids; k++){
        
        int i = grid_indexer_.x_grid[k], j = grid_indexer_.y_grid[k];
        double x = x_min + i*hx, y = y_min + j*hy;

        if ( k!= line_ind[i*N_y+j] ) {
            throw std::out_of_range("Wrong index!");
        }

        std::vector<double> hh = grid_indexer_.h_news[k];
        int state_n, state_e, state_s, state_w;
        state_n = state_judge(hh[0], hy);
        state_e = state_judge(hh[1], hx);
        state_s = state_judge(hh[2], hy);
        state_w = state_judge(hh[3], hx);
        
        bool success = false ;
        int s = state_n + state_e + state_s + state_w;
        if (s==0){

            fill_matrix(row_indices, col_indices, values, k, k, 2.0/(hy*hy) + 2.0/(hx*hx));
            fill_matrix(row_indices, col_indices, values, k, k+1, -1.0/(hy*hy));
            fill_matrix(row_indices, col_indices, values, k, k-1, -1.0/(hy*hy));
            fill_matrix(row_indices, col_indices, values, k, line_ind[(i+1)*N_y+j], -1.0/(hx*hx));
            fill_matrix(row_indices, col_indices, values, k, line_ind[(i-1)*N_y+j], -1.0/(hx*hx));

            b[k] = f(x,y);
            success = true;
        }
        if (s==1){
            if(state_n==1){
                fill_matrix(row_indices, col_indices, values, k, k, 2.0/(hy*hy+2*hy*hh[0]) + 2.0/(hx*hx));
                fill_matrix(row_indices, col_indices, values, k, k-1, -2.0/(hy*hy+2*hy*hh[0]));
                fill_matrix(row_indices, col_indices, values, k, line_ind[(i+1)*N_y+j], -1.0/(hx*hx));
                fill_matrix(row_indices, col_indices, values, k, line_ind[(i-1)*N_y+j], -1.0/(hx*hx));
            b[k] = f(x,y) + direc_derivative(x, y+hh[0])*2/(hy+2*hh[0]);
            success = true;
            }

            if(state_e==1){
                fill_matrix(row_indices, col_indices, values, k, k, (2.0/(hx*hx) + 1.0/(hx*hy))/std::sqrt(5) );
                fill_matrix(row_indices, col_indices, values, k, line_ind[(i-1)*N_y+j], -2.0/(hx*hx*std::sqrt(5)) );
                fill_matrix(row_indices, col_indices, values, k, k+1, -1.0/(hx*hy*std::sqrt(5)) );
                b[k] = direc_derivative(x + 0.8*hh[1], y-0.4*hh[1]) / hx;
                success = true;
            }

            if(state_s==1){
                fill_matrix(row_indices, col_indices, values, k, k, 2.0/(hy*hy+2*hy*hh[2]) + 2.0/(hx*hx));
                fill_matrix(row_indices, col_indices, values, k, k+1, -2.0/(hy*hy+2*hy*hh[2]));
                fill_matrix(row_indices, col_indices, values, k, line_ind[(i+1)*N_y+j], -1.0/(hx*hx));
                fill_matrix(row_indices, col_indices, values, k, line_ind[(i-1)*N_y+j], -1.0/(hx*hx));
                b[k] = f(x,y) + direc_derivative(x, y-hh[2])*2/(hy+2*hh[2]);
                success = true;
            }

            if(state_w==1){
                fill_matrix(row_indices, col_indices, values, k, k, 2.0/(hx*hx+2*hx*hh[3]) + 2.0/(hy*hy));
                fill_matrix(row_indices, col_indices, values, k, line_ind[(i+1)*N_y+j], -2.0/(hx*hx+2*hx*hh[3]));
                fill_matrix(row_indices, col_indices, values, k, k+1, -1.0/(hy*hy));
                fill_matrix(row_indices, col_indices, values, k, k-1, -1.0/(hy*hy));
                b[k] = f(x,y) + direc_derivative(x-hh[3], y)*2/(hx+2*hh[3]);
                success = true;
            }
            }

        if (s==2){
            if(state_n && state_e){
                fill_matrix(row_indices, col_indices, values, k, k, (1.0/(hx*hx) + 1.0/(hx*hy))/std::sqrt(2) );
                fill_matrix(row_indices, col_indices, values, k, line_ind[(i-1)*N_y+j], -1.0/(hx*hx*std::sqrt(2)) );
                fill_matrix(row_indices, col_indices, values, k, k-1, -1.0/(hx*hy*std::sqrt(2)) );
                b[k] = direc_derivative(x + hh[0]/std::sqrt(2), y + hh[0] /std::sqrt(2) ) / hx;
                success = true;

            }

            if(state_e && state_s){
                fill_matrix(row_indices, col_indices, values, k, k, (2.0/(hx*hx) + 1.0/(hx*hy))/std::sqrt(5) );
                fill_matrix(row_indices, col_indices, values, k, line_ind[(i-1)*N_y+j], -2.0/(hx*hx*std::sqrt(5)) );
                fill_matrix(row_indices, col_indices, values, k, k+1, -1.0/(hx*hy*std::sqrt(5)) );
                b[k] = direc_derivative(x + 0.8*hh[1], y-0.4*hh[1]) / hx;
                success = true;
            }

            if(state_s && state_w){
                fill_matrix(row_indices, col_indices, values, k, k, 2.0/(hx*hx) + 2.0/(hx*hy) );
                fill_matrix(row_indices, col_indices, values, k, line_ind[(i+1)*N_y+j], -2.0/(hx*hx));
                fill_matrix(row_indices, col_indices, values, k, k+1, -2.0/(hx*hx));
                b[k] = (direc_derivative(x, y - hh[2] ) + direc_derivative(x - hh[2], y) ) * 2/ hx;
                success = true;
            }

            if(state_n && state_w){
                fill_matrix(row_indices, col_indices, values, k, k, (1.0/(hx*hx) + 1.0/(hx*hy))/std::sqrt(2) );
                fill_matrix(row_indices, col_indices, values, k, line_ind[(i+1)*N_y+j], -1.0/(hx*hx*std::sqrt(2)) );
                fill_matrix(row_indices, col_indices, values, k, k-1, -1.0/(hx*hy*std::sqrt(2)) );
                b[k] = direc_derivative(x - hh[3]/std::sqrt(2), y + hh[3] /std::sqrt(2) ) / hx;
                success = true;
            }
        }

        if (s==3){
            if( state_s ==0 ){
                fill_matrix(row_indices, col_indices, values, k, k, 4.0/(hx*hx));
                fill_matrix(row_indices, col_indices, values, k, k-1, -4.0/(hx*hx) );
                b[k] = (direc_derivative(x + hh[0]/std::sqrt(2), y + hh[0] /std::sqrt(2)) +  
                direc_derivative(x - hh[0]/std::sqrt(2), y + hh[0] /std::sqrt(2)) )*2*sqrt(2) / hx;
                success = true;
            }

            if( state_w ==0 ){
                fill_matrix(row_indices, col_indices, values, k, k, 4.0/(hx*hx));
                fill_matrix(row_indices, col_indices, values, k, line_ind[(i-1)*N_y+j], -4.0/(hx*hx) );
                b[k] = ( sqrt(2) * direc_derivative(x + hh[0]/std::sqrt(2), y + hh[0] /std::sqrt(2)) +  
                direc_derivative(x , y - hh[2]) )*4 / hx;
                success = true;

            }

        }

        if(!success){
            std::cout<<"(x,y)=("<< x << "," <<y <<")\n";
            std::cout<< "NESW:" << state_n <<" "<< state_e <<" "<< state_s <<" "<< state_w;
            throw std::invalid_argument("Undefined Boundary Case"); 
        }
    }

    SparseMatrixCSR M = SparseMatrixCSR(N_grids, N_grids, row_indices, col_indices, values);

    std::vector<double> x_0(N_grids, 0.0);
    
    std::vector<double> u = GMRES(M, b, x_0, 50, 80);
        
    result_output(grid_indexer_, u, hx, hy, x_min, y_min, "result_2", true);

    return;
}


int main(){
    int N = 80;

    auto start_CG = std::chrono::high_resolution_clock::now();

    Dirichlet_Possion_solver(0.0, 2.0, -2.0, 2.0, 2*N+1, 4*N+1);
    // Neumann_Possion_solver(0.0, 2.0, -2.0, 2.0, 2*N+1, 4*N+1);
    auto end_CG = std::chrono::high_resolution_clock::now();

    auto CG_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_CG- start_CG);


    std::cout<< "Run time: "<< CG_time.count() << " ms"<< std::endl;
    
    system("pause");
    return 0;
}
