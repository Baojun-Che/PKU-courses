#include "test_functions.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <limits>

const double RATE = 1.001;
const double PI = 3.14159265;


//  Possion equation
double g(double x, double y){
    return sin(PI*x)*cos(PI*y);
}

double f(double x, double y){
    return 2*PI*PI*sin(PI*x)*cos(PI*y);
}

double direc_derivative(double x, double y){
    std::vector<double> grad = {PI * cos(PI*x)*cos(PI*y), -PI* sin(PI*x)*sin(PI*y)};
    std::vector<double> n;
    if (abs(x)<RATE-1)  {n = {-1,0}; return vector_dot(grad,n);}
    if (abs(y+2)<RATE-1)  {n = {0,-1}; return vector_dot(grad,n);}
    if (x<=1)  {n = {-1.0/std::sqrt(2), 1.0/std::sqrt(2)}; return vector_dot(grad,n);}
    if (x>=1 && ( y>1 || y<-1 ) ) {n = {1.0/std::sqrt(2), 1.0/std::sqrt(2)}; return vector_dot(grad,n);}
    if (x>=1 && y>-1)  {n = {2.0/std::sqrt(5), -1.0/std::sqrt(5)}; return vector_dot(grad,n);}
    std::cout<<"Warning: undefined direc_derivative at ("<<x<<","<<y<<")\n";
    return 0 ;
}




//  Heat equation
double rho(double t){
    return sin(t);
}

double u0(double x, double y){
    return 0.0;
}

double g_t(double x, double y, double t) {
    if (t <= 1e-6) return 0;
    double R2 = (x-1)*(x-1) + y*y;
    double R = sqrt(R2);
    
    if (R < 1e-10) {
        return std::numeric_limits<double>::infinity();
    }
    
    // 变量代换 u = 1/τ, τ = 1/u, dτ = -du/u²
    // 积分变为 ∫_{1/t}^{∞} exp(-R² u/4) sin(t - 1/u) / (4π) du
    double u_max = 1e6;  // 对应 τ_min = 1e-8
    double u_min = 1.0/t;
    
    // 对数采样 u
    int N = 200;
    double I = 0;
    double log_u_min = log(u_min);
    double log_u_max = log(u_max);
    double dlogu = (log_u_max - log_u_min) / N;
    
    for (int k = 0; k < N; k++) {
        double log_u = log_u_min + (k+0.5)*dlogu;
        double u = exp(log_u);
        double tau = 1.0/u;
        double jacobian = dlogu * u;  // du = u dlogu
        I += jacobian * exp(-R2*u/4) * sin(t - tau) / (4*PI);
    }
    return I;
}




// 定义网格

int state_judge(double dis, double h){  
    if (dis> h*RATE)
        return 0;
    else
        return 1;
}

std::vector<double> area_judge(double x, double y) {
    std::vector<double> hh(4, -1.0);
    if (x > 0 && x < 1 && y > -2 && y <= -1) { hh = {x + 1 - y, -y - x, y + 2, x}; }
    if (x > 0 && x < 1 && y > -1 && y <= 1) { hh = {x + 1 - y, (y + 3) / 2 - x, y + 2, x}; }
    if (x > 0 && x < 1 && y > 1 && y < x + 1) { hh = {x + 1 - y, 3 - y - x, y + 2, x - y + 1}; }
    if (x >= 1 && x < 2 && y > -2 && y < -x) { hh = {-y - x, -y - x, y + 2, x}; }
    if (x >= 1 && x < 2 && y > 2 * x - 3 && y <= 1) { hh = {3 - x - y, (y + 3) / 2 - x, 3 + y - 2 * x, x}; }
    if (x >= 1 && x < 2 && y > 1 && y < 3 - x) { hh = {3 - x - y, 3 - y - x, 3 + y - 2 * x, x - y + 1}; }
    return hh;
}

grid_indexer grid_indexer_init(double x_min, double x_max, double y_min, double y_max, int N_x, int N_y) {
    double hx = (x_max - x_min) / (N_x - 1), hy = (y_max - y_min) / (N_y - 1);

    int N_grid_points = 0;
    std::vector<int> x_grid;
    std::vector<int> y_grid;
    std::vector<int> line_ind;
    std::vector<std::vector<double>> h_news;

    for (int i = 0; i < N_x; i++) {
        int cnt = 0;
        for (int j = 0; j < N_y; j++) {
            double x = x_min + i * hx;
            double y = y_min + j * hy;
            std::vector<double> hh = area_judge(x, y);

            if (hh[0] > hy * (RATE - 1) && hh[1] > hx * (RATE - 1) && hh[2] > hy * (RATE - 1) && hh[3] > hx * (RATE - 1)) {
                line_ind.push_back(N_grid_points);
                N_grid_points++;
                x_grid.push_back(i);
                y_grid.push_back(j);
                h_news.push_back(hh);
                cnt++;
            } else {
                line_ind.push_back(-1);
            }
        }
    }
    
    grid_indexer grid_indexer_;
    grid_indexer_.N_grid_points = N_grid_points;
    grid_indexer_.x_grid = x_grid;
    grid_indexer_.y_grid = y_grid;
    grid_indexer_.h_news = h_news;
    grid_indexer_.line_ind = line_ind;
    return grid_indexer_;
}
