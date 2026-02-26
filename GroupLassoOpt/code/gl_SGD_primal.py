import numpy as np
import math

def cos_annealing(iter, max_iter, dt_min, dt_max):
    
    iter_cos_decay = max_iter
    if iter >= iter_cos_decay:
        return dt_min
    else:
        return dt_min + (1 + math.cos(math.pi * (iter/iter_cos_decay) ) ) * (dt_max-dt_min) /2

def subgrad_regular(x):
    n, l = x.shape
    grad = np.zeros_like(x)
    for i in range(n):
        row_i = x[i, :]
        norm_row_i = np.linalg.norm(row_i)
        if norm_row_i > 1e-6:
            grad[i, :] = row_i / norm_row_i
        else:
            grad[i, :] = row_i / (1 + norm_row_i)
    return grad

def SGD_opts_init():
    opts = {
        'N_out_iter' : 10,
        'max_iter_total' : 10000,
        'max_iter_inner' : 500,
        'alpha' : 0.5,
        'dt_max' : 0.001,
        'tol' : 1e-3,
    }
    return opts

def gl_SGD_primal(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts:dict = {}):

    x = x0.copy()
    m, n = A.shape
    _, l = b.shape
    
    if not opts:
        opts = SGD_opts_init()
    
    N_out_iter = opts['N_out_iter']
    max_iter_total = opts['max_iter_total']
    max_iter_inner = opts['max_iter_inner']
    alpha = opts['alpha']
    dt_max = opts['dt_max']
    tol = opts['tol']
    
    f_values = []
    
    x_opt = np.zeros_like(x)
    best_obj = 0.5 * np.sum(b**2)
    
    flag = False

    for out_iter in range(N_out_iter):

        mu_current = cos_annealing(out_iter, N_out_iter - 1, mu, max(2.0, mu))

        obj_current_old = 0.0

        if out_iter == N_out_iter - 1:
            flag = True
            tol = 1e-7
            max_iter_inner = 5000

        for k in range(max_iter_inner):

            r = A @ x - b
            f_smooth = 0.5 * np.sum(r**2)
            f_regular = np.sum(np.linalg.norm(x, axis=1))

            obj = f_smooth + mu * f_regular
            f_values.append(obj)
            if obj < best_obj:
                x_opt = x.copy()
                best_obj = obj
                
            obj_current_new = f_smooth + mu_current * f_regular
            if np.abs(obj_current_new - obj_current_old) < tol:
                break

            obj_current_old = obj_current_new

            grad_smooth = A.T @ r
            subgrad_non_smooth = mu_current * subgrad_regular(x)
            subgrad_total = grad_smooth + subgrad_non_smooth 
            
            dt = min( dt_max, alpha / (k+1) )

            x = x - dt * subgrad_total
            if len(f_values)-1 >= max_iter_total:
                flag = True
                break
        
        if flag :
            break

        tol = max(tol*0.8, 1e-6)
        dt_max *= 0.9      
    
    r = A @ x - b
    f_smooth = 0.5 * np.sum(r**2)
    f_regular = np.sum(np.linalg.norm(x, axis=1))

    obj = f_smooth + mu * f_regular
    f_values.append(obj)
    if obj < best_obj:
        x_opt = x.copy()
        best_obj = obj

    return x_opt, len(f_values)-1, f_values

