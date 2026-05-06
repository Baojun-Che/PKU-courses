using LinearAlgebra
using Random
using PyPlot

function Langevin(d_Phi, X0, N_iter = 10000)
    ens = copy(X0)
    N_x, N_ens = size(ens)
    for iter in 1:N_iter
        cov_ens = cov(ens, dims=2)
        cov_sqrt = cholesky(cov_ens).L
        grads = [d_Phi(ens[:, i]) for i in 1:N_ens]
        norms = [ norm(grads[i]) for i in 1:N_ens]
        stepsize = 1e-2 * min(1 / maximum(norms), 1/ norm(cov_ens), 1)
        for i = 1:N_ens
            ens[:, i] += -stepsize * cov_ens * grads[i] + sqrt(2 * stepsize) * cov_sqrt * randn(N_x) 
        end
    end
    return ens
end