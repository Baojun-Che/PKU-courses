using LinearAlgebra
using Random
using PyPlot

function ImportantSampling(logrho, N_ens=10000, N_x = 2, sigma_prior=4.0)
    θ_init = randn(N_x, N_ens) * sigma_prior
    weights = zeros(N_ens)
    for i in 1:N_ens
        weights[i] = exp(logrho(θ_init[:,i]) - norm(θ_init[:,i])^2 / (2 * sigma_prior^2) )
    end
    weights = weights ./ sum(weights)
    return θ_init, weights
end

function test_IS(logrho, plot_name = nothing)
    N_ens_array = [100, 1000, 10000, 100000]
    his_mean = []
    his_cov = []
    for N_ens in N_ens_array
        θ_init, weights = ImportantSampling(logrho, N_ens)
        mean = sum(θ_init[:, i] * weights[i] for i in 1:N_ens)
        cov = sum((θ_init[:, i]-mean) * (θ_init[:, i]-mean)' * weights[i] for i in 1:N_ens)
        push!(his_mean, mean)
        push!(his_cov, cov)
    end
    return his_mean, his_cov, N_ens_array
end
