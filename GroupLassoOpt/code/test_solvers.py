import utils
import numpy as np
import os
import time
from gl_cvx_mosek import gl_cvx_mosek
from gl_cvx_gurobi import gl_cvx_gurobi
from gl_mosek import gl_mosek
from gl_gurobi import gl_gurobi 
from gl_SGD_primal import gl_SGD_primal
from gl_GD_primal import gl_GD_primal
from gl_ProxGD_primal import gl_ProxGD_primal
from gl_FProxGD_primal import gl_FProxGD_primal
from gl_ALM_dual import gl_ALM_dual
from gl_ADMM_dual import gl_ADMM_dual
from gl_ADMM_primal import gl_ADMM_primal

def test_all_solvers(seed = 97006855, n = 512, m = 256, l = 2, mu = 0.01, sparse = 0.1):

    A, b, u, x0 = utils.test_data_init(seed = seed, n = n, m = m, l = l, mu = mu, sparse = sparse)
    f_values_array = []
    labels = []


    os.makedirs("code/datas", exist_ok=True)
    with open("code/datas/result_all_solvers.txt", "w") as file:

        header = f"{'Solver':<15} | {'Fval':>10} {'Errfun':>10} {'Errfun_Exact':>13} {'Time(s)':>8} {'Iter':>6} {'Sparsity':>9}\n"
        file.write(header)
        file.write("-" * len(header) + "\n")
        none = '-'

        ######## CVX-mosek ########
        method = "cvx_mosek"
        start = time.time()
        x_mosek, _, _ = gl_cvx_mosek(x0, A, b, mu, {})
        end = time.time()
        f_opt = 0.5 * np.sum( (A @ x_mosek - b)**2 ) + mu * np.sum(np.linalg.norm(x_mosek, axis=1))
        t = end - start
        err_exact = utils.relative_error(u, x_mosek)
        sparsity = utils.compute_nonzero_ratio(x_mosek)
        line = ( f"{method:<15} | " f"{f_opt:>10.8f} " f"{none:>10} " f"{err_exact:>13.2e} " f"{t:>8.4f} " f"{none:>6} "f"{sparsity:>9.4f}\n")
        file.write(line)
        print(f"cvx-mosek 求得目标函数最小值: {f_opt}")


        ######## cvx_gurobi, mosek_direct, gurobi_direct ########
        method = "cvx_gurobi"
        start = time.time()
        x_opt, _, _ = gl_cvx_gurobi(x0, A, b, mu, {})
        end = time.time()
        f_opt = 0.5 * np.sum( (A @ x_mosek - b)**2 ) + mu * np.sum(np.linalg.norm(x_mosek, axis=1))
        t = end - start
        err = utils.relative_error(x_mosek, x_opt)
        err_exact = utils.relative_error(u, x_opt)
        sparsity = utils.compute_nonzero_ratio(x_opt)
        line = ( f"{method:<15} | " f"{f_opt:>10.8f} " f"{err:>10.2e} " f"{err_exact:>13.2e} " f"{t:>8.4f} " f"{none:>6} "f"{sparsity:>9.4f}\n")
        file.write(line)

        method = "mosek_direct"
        start = time.time()
        x_opt, _, _ = gl_mosek(x0, A, b, mu, {})
        end = time.time()
        f_opt = 0.5 * np.sum( (A @ x_mosek - b)**2 ) + mu * np.sum(np.linalg.norm(x_mosek, axis=1))
        t = end - start
        err = utils.relative_error(x_mosek, x_opt)
        err_exact = utils.relative_error(u, x_opt)
        sparsity = utils.compute_nonzero_ratio(x_opt)
        line = ( f"{method:<15} | " f"{f_opt:>10.8f} " f"{err:>10.2e} " f"{err_exact:>13.2e} " f"{t:>8.4f} " f"{none:>6} "f"{sparsity:>9.4f}\n")
        file.write(line)
        
        method = "gurobi_direct"
        start = time.time()
        x_opt, _, _ = gl_gurobi(x0, A, b, mu, {})
        end = time.time()
        f_opt = 0.5 * np.sum( (A @ x_mosek - b)**2 ) + mu * np.sum(np.linalg.norm(x_mosek, axis=1))
        t = end - start
        err = utils.relative_error(x_mosek, x_opt)
        err_exact = utils.relative_error(u, x_opt)
        sparsity = utils.compute_nonzero_ratio(x_opt)
        line = ( f"{method:<15} | " f"{f_opt:>10.8f} " f"{err:>10.2e} " f"{err_exact:>13.2e} " f"{t:>8.4f} " f"{none:>6} "f"{sparsity:>9.4f}\n")
        file.write(line)

        ######## other solvers ########

        method = "SGD_primal"
        start = time.time()
        x_opt, iter_count, f_values = gl_SGD_primal(x0, A, b, mu, {})
        end = time.time()
        f_opt = 0.5 * np.sum( (A @ x_opt - b)**2 ) + mu * np.sum(np.linalg.norm(x_opt, axis=1))
        err = utils.relative_error(x_mosek, x_opt)
        err_exact = utils.relative_error(u, x_opt)
        t = end - start
        sparsity = utils.compute_nonzero_ratio(x_opt)
        line = ( f"{method:<15} | " f"{f_opt:>10.8f} " f"{err:>10.2e} " f"{err_exact:>13.2e} " f"{t:>8.4f} " f"{iter_count:>6d} "f"{sparsity:>9.4f}\n")
        file.write(line)
        f_values_array.append(f_values)
        labels.append(method)

        method = "GD_primal"
        start = time.time()
        x_opt, iter_count, f_values = gl_GD_primal(x0, A, b, mu, {})
        end = time.time()
        f_opt = 0.5 * np.sum( (A @ x_opt - b)**2 ) + mu * np.sum(np.linalg.norm(x_opt, axis=1))
        err = utils.relative_error(x_mosek, x_opt)
        err_exact = utils.relative_error(u, x_opt)
        t = end - start
        sparsity = utils.compute_nonzero_ratio(x_opt)
        line = ( f"{method:<15} | " f"{f_opt:>10.8f} " f"{err:>10.2e} " f"{err_exact:>13.2e} " f"{t:>8.4f} " f"{iter_count:>6d} "f"{sparsity:>9.4f}\n")
        file.write(line)
        f_values_array.append(f_values)
        labels.append(method)

        method = "ProxGD_primal"
        start = time.time()
        x_opt, iter_count, f_values = gl_ProxGD_primal(x0, A, b, mu, {})
        end = time.time()
        f_opt = 0.5 * np.sum( (A @ x_opt - b)**2 ) + mu * np.sum(np.linalg.norm(x_opt, axis=1))
        err = utils.relative_error(x_mosek, x_opt)
        err_exact = utils.relative_error(u, x_opt)
        t = end - start
        sparsity = utils.compute_nonzero_ratio(x_opt)
        line = ( f"{method:<15} | " f"{f_opt:>10.8f} " f"{err:>10.2e} " f"{err_exact:>13.2e} " f"{t:>8.4f} " f"{iter_count:>6d} "f"{sparsity:>9.4f}\n")
        file.write(line)
        f_values_array.append(f_values)
        labels.append(method)

        method = "FProxGD_primal"
        start = time.time()
        x_opt, iter_count, f_values = gl_FProxGD_primal(x0, A, b, mu, {})
        end = time.time()
        f_opt = 0.5 * np.sum( (A @ x_opt - b)**2 ) + mu * np.sum(np.linalg.norm(x_opt, axis=1))
        err = utils.relative_error(x_mosek, x_opt)
        err_exact = utils.relative_error(u, x_opt)
        t = end - start
        sparsity = utils.compute_nonzero_ratio(x_opt)
        line = ( f"{method:<15} | " f"{f_opt:>10.8f} " f"{err:>10.2e} " f"{err_exact:>13.2e} " f"{t:>8.4f} " f"{iter_count:>6d} "f"{sparsity:>9.4f}\n")
        file.write(line)
        f_values_array.append(f_values)
        labels.append(method)
        
        method = "ALM_dual"
        start = time.time()
        x_opt, iter_count, f_values = gl_ALM_dual(x0, A, b, mu, {})
        end = time.time()
        f_opt = 0.5 * np.sum( (A @ x_opt - b)**2 ) + mu * np.sum(np.linalg.norm(x_opt, axis=1))
        err = utils.relative_error(x_mosek, x_opt)
        err_exact = utils.relative_error(u, x_opt)
        t = end - start
        sparsity = utils.compute_nonzero_ratio(x_opt)
        line = ( f"{method:<15} | " f"{f_opt:>10.8f} " f"{err:>10.2e} " f"{err_exact:>13.2e} " f"{t:>8.4f} " f"{iter_count:>6d} "f"{sparsity:>9.4f}\n")
        file.write(line)
        f_values_array.append(f_values)
        labels.append(method)

        method = "ADMM_dual"
        start = time.time()
        x_opt, iter_count, f_values = gl_ADMM_dual(x0, A, b, mu, {})
        end = time.time()
        f_opt = 0.5 * np.sum( (A @ x_opt - b)**2 ) + mu * np.sum(np.linalg.norm(x_opt, axis=1))
        err = utils.relative_error(x_mosek, x_opt)
        err_exact = utils.relative_error(u, x_opt)
        t = end - start
        sparsity = utils.compute_nonzero_ratio(x_opt)
        line = ( f"{method:<15} | " f"{f_opt:>10.8f} " f"{err:>10.2e} " f"{err_exact:>13.2e} " f"{t:>8.4f} " f"{iter_count:>6d} "f"{sparsity:>9.4f}\n")
        file.write(line)
        f_values_array.append(f_values)
        labels.append(method)

        method = "ADMM_primal"
        start = time.time()
        x_opt, iter_count, f_values = gl_ADMM_primal(x0, A, b, mu, {})
        end = time.time()
        f_opt = 0.5 * np.sum( (A @ x_opt - b)**2 ) + mu * np.sum(np.linalg.norm(x_opt, axis=1))
        err = utils.relative_error(x_mosek, x_opt)
        err_exact = utils.relative_error(u, x_opt)
        t = end - start
        sparsity = utils.compute_nonzero_ratio(x_opt)
        line = ( f"{method:<15} | " f"{f_opt:>10.8f} " f"{err:>10.2e} " f"{err_exact:>13.2e} " f"{t:>8.4f} " f"{iter_count:>6d} "f"{sparsity:>9.4f}\n")
        file.write(line)
        f_values_array.append(f_values)
        labels.append(method)

    obj_mosek = 0.5 * np.sum( (A @ x_mosek - b)**2 ) + mu * np.sum(np.linalg.norm(x_mosek, axis=1))
    utils.plot_relative_error(f_values_array, labels, "plot_all_solvers", obj_mosek)


if __name__ == "__main__":
    test_all_solvers()
    