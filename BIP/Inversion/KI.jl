using LinearAlgebra
using Random
using PyPlot
using ForwardDiff
include("QuadratureRule.jl")

function KI(forward_G, Sigma_0, Sigma_y, y, r0, name="ExtendedKI")

    Sigma_0_sqrt = cholesky(Sigma_0).L
    N_x = length(r0)
    N_y = length(y)
    C_xy = zeros(N_x, N_y)
    C_yy = zeros(N_y, N_y)
    y_hat = zeros(N_y)
    if name == "ExtendedKI" 
        jacobian = ForwardDiff.jacobian(forward_G, r0)
        y_hat = forward_G(r0)
        C_xy = Sigma_0 * jacobian'
        C_yy = jacobian * C_xy + Sigma_y
    elseif name == "UKI"
        N_ens, c_weights, mean_weights = generate_quadrature_rule(N_x, "cubature_transform_o5")
        ens = construct_ensemble(r0, Sigma_0_sqrt, c_weights = c_weights)'
        G_ens = [forward_G(ens[:,i]) for i in 1:N_ens]
        mean_weights = mean_weights ./ sum(mean_weights)
        y_hat = forward_G(r0)
        C_xy = sum(mean_weights[i] * (ens[:,i]-r0) * (G_ens[i]-y_hat)' for i in 1:N_ens) 
        C_yy = sum(mean_weights[i] * (G_ens[i]-y_hat) * (G_ens[i]-y_hat)' for i in 1:N_ens) + Sigma_y
    elseif name == "EnKI"
        N_ens = 100
        ens = construct_ensemble(r0, Sigma_0_sqrt, N_ens = N_ens)'
        mean_weights = ones(N_ens) ./ N_ens
        G_ens = [forward_G(ens[:,i]) for i in 1:N_ens]
        y_hat = forward_G(r0)
        C_xy = sum(mean_weights[i] * (ens[:,i]-r0) * (G_ens[i]-y_hat)' for i in 1:N_ens) 
        C_yy = sum(mean_weights[i] * (G_ens[i]-y_hat) * (G_ens[i]-y_hat)' for i in 1:N_ens) + Sigma_y
    end
    K = C_xy * inv(C_yy)
    m = r0 + K * (y - y_hat)
    C = Symmetric(Sigma_0 - K * C_xy')
    println("y_hat= $(y_hat)")
    println("Cxy= $(C_xy)")
    println("Cyy= $(C_yy)")
    return m, C
end
