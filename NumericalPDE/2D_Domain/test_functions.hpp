#ifndef TEST_FUNCTIONS_HPP
#define TEST_FUNCTIONS_HPP

#include "../sparse_matrix/include/solvers.hpp"
#include <vector>
#include <string>


extern const double RATE;
extern const double PI;


//  Possion equation
double g(double x, double y);
double f(double x, double y);
double direc_derivative(double x, double y);

//  Heat equation
double rho(double t);
double u0(double x, double y);
double g_t(double x, double y, double t);

// 定义网格
int state_judge(double dis, double h);
struct grid_indexer {
    int N_grid_points;
    std::vector<int> x_grid;
    std::vector<int> y_grid;
    std::vector<int> line_ind;
    std::vector<std::vector<double>> h_news; 
};
std::vector<double> area_judge(double x, double y);
grid_indexer grid_indexer_init(double x_min, double x_max, double y_min, double y_max, int N_x, int N_y);


#endif

