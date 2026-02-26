import numpy as np

def prox_group_lasso(z, dt, mu):

    norms = np.linalg.norm(z, axis=1, keepdims=True)  # Shape: (n, 1)
    k = dt * mu
    scaling = np.maximum(0.0, 1.0 - k / (norms + 1e-8))      
    x_next = scaling * z
    
    return x_next

def gl_ProxGD_primal(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts:dict = {}):
    
    norm_A = np.linalg.norm(A, ord=2)
    Lip =  norm_A**2 
    
    x = x0.copy()
    m, n = A.shape
    _, l = b.shape
    
    max_iter_total = 5000
    max_iter_inner = 500
    dt = 1.0/Lip
    tol = 1e-3
    
    f_values = []
    
    x_opt = np.zeros_like(x)
    best_obj = 0.5 * np.sum(b**2)
    
    iter_count = 0
    mu_current = 2e4 * mu
    flag = False

    while(1):

        mu_current = 0.1 * mu_current
        tol = 0.1* tol

        obj_current_old = 0.0

        if mu_current <=mu:
            mu_current = mu
            tol = 1e-7
            flag = True
            max_iter_inner = 1000

        # print(f"Iterations: {iter_count}, current mu={mu_current}")

        for k in range(max_iter_inner):

            r = A @ x - b
            f_smooth = 0.5 * np.sum(r**2)
            f_regular = np.sum(np.linalg.norm(x, axis=1))

            obj = f_smooth + mu * f_regular
            f_values.append(obj)
            iter_count += 1
            if obj < best_obj:
                x_opt = x.copy()
                best_obj = obj
                
            obj_current_new = f_smooth + mu_current * f_regular
            if np.abs(obj_current_new - obj_current_old) < tol:
                break

            obj_current_old = obj_current_new

            grad_smooth = A.T @ r
            z = x - dt * grad_smooth
            x = prox_group_lasso(z, dt, mu_current)

            if len(f_values)-1 >= max_iter_total:
                flag = True
                break
        
        if flag :
            break

    r = A @ x - b
    obj =  0.5 * np.sum(r**2) + mu * np.sum(np.linalg.norm(x, axis=1))
    f_values.append(obj)
    if obj < best_obj:
        x_opt = x.copy()
        best_obj = obj
        
    return x_opt, len(f_values)-1, f_values


