using LinearAlgebra
using ForwardDiff
using Random
using PyPlot

include("../Inversion/AffineInvariantMCMC.jl")
include("../Inversion/GMNVI.jl")
include("../Inversion/ImportantSampling.jl")
include("../Inversion/KI.jl")
include("../Inversion/NormalizingFlow.jl")        
include("../Inversion/Langevin.jl")

include("utils.jl")

io = open("output.txt", "w")
redirect_stdout(io)



y = [0, 1]
r0 = [0, 0]
Sigma_0 = [100 0; 0 100]
Sigma_y = [0.01 0; 0 1]

function G_map(θ, c)
    return [θ[2]-c*θ[1]^2; θ[1]]
end

function func_Phi_map(θ, c)
    err = G_map(θ, c) - y
    ans = 0.5 * err' * inv(Sigma_y) * err + 0.5 * (θ - r0)' * inv(Sigma_0) * (θ - r0)
    return ans
end


for c in [0.01, 1.0]
     
    func_Phi(θ) = func_Phi_map(θ, c)
    func_G(θ) = G_map(θ, c)

    fig, ax = PyPlot.subplots(nrows=2, ncols=4, sharex=false, sharey=false, figsize=(20,10))

    println("c= $(c)")
    println("====================")

    println("0. Reference")
    mean_ref, cov_ref, X, Y, Z_ref = cal_mean_cov(func_Phi)
    color_lim = (minimum(Z_ref), maximum(Z_ref))
    ax[1,1].pcolormesh(X, Y, Z_ref, cmap="viridis", clim=color_lim)
    ax[1,1].scatter(mean_ref[1], mean_ref[2], marker="x", color="red", alpha=1.0) 
    println("mean_ref= $(mean_ref)")
    println("cov_ref= $(cov_ref)")
    println("-----------------")

    println("1. Important Sampling")
    logrho(x) = -func_Phi(x)
    his_mean, his_cov, N_ens_array = test_IS(logrho)
    is_mean = his_mean[end]
    is_cov = his_cov[end]
    error_plot(his_mean, his_cov, mean_ref, cov_ref, N_ens_array, "$c")
    println("mean= $(is_mean)")
    println("cov= $(is_cov)")   
    println("-----------------")


    println("2. Extended KI")
    eki_mean, eki_cov = KI(func_G, Sigma_0, Sigma_y, y, r0, "ExtendedKI")
    plot_Gaussian(ax[1,2], eki_mean, eki_cov, color_lim)
    println("mean= $(eki_mean)")
    println("cov= $(eki_cov)")   
    println("-----------------")

    println("3. UKI")
    uki_mean, uki_cov = KI(func_G, Sigma_0, Sigma_y, y, r0, "UKI")
    plot_Gaussian(ax[1,3], uki_mean, uki_cov, color_lim)
    println("mean= $(uki_mean)")
    println("cov= $(uki_cov)")   
    println("-----------------")

    println("4. EnKI")
    enki_mean, enki_cov = KI(func_G, Sigma_0, Sigma_y, y, r0, "EnKI")   
    plot_Gaussian(ax[1,4], enki_mean, enki_cov, color_lim)
    println("mean= $(enki_mean)")
    println("cov= $(enki_cov)")   
    println("-----------------")

    println("4. Gaussian Natural Variational Inference")
    gmnv_mean, gmnv_cov = GaussianVI(r0, Sigma_0, func_Phi)
    plot_Gaussian(ax[2,1], gmnv_mean, gmnv_cov, color_lim)
    println("mean= $(gmnv_mean)")
    println("cov= $(gmnv_cov)")   
    println("-----------------")

    println("5. Affine Invariant MCMC")
    func_prob(x) = exp(-func_Phi(x))
    mcmc_samples = Run_StretchMove(randn(2, 500), func_prob, N_iter = 5000)
    mcmc_mean, mcmc_cov = ens_mean_cov(mcmc_samples)
    ax[2,2].scatter(mcmc_samples[1,:]', mcmc_samples[2,:]', marker=".", color="blue", alpha=0.5)
    ax[2,2].scatter(mcmc_mean[1], mcmc_mean[2], marker="x", color="red", alpha=1.0) 
    println("mean= $(mcmc_mean)")
    println("cov= $(mcmc_cov)")     
    println("-----------------")

    println("6. Normalizing Flow")
    J = 10000
    init_samples = randn(2, J)
    flow, losses = NF(init_samples, func_Phi)
    nf_samples = sample_posterior(flow, J)
    nf_mean, nf_cov = ens_mean_cov(nf_samples)
    ax[2,3].scatter(nf_samples[1,:]', nf_samples[2,:]', marker=".", color="blue", alpha=0.1) 
    ax[2,3].scatter(nf_mean[1], nf_mean[2], marker="x", color="red", alpha=1.0) 
    println("mean= $(nf_mean)")
    println("cov= $(nf_cov)")   
    println("-----------------")

    println("7. Perconditioned Langevin")
    func_dPhi(x) = ForwardDiff.gradient(func_Phi, x)
    pl_samples = Langevin(func_dPhi, 0.5 * randn(2, 100), 10000)
    pl_mean, pl_cov = ens_mean_cov(pl_samples)
    ax[2,4].scatter(pl_samples[1,:]', pl_samples[2,:]', marker=".", color="blue", alpha=0.5) 
    ax[2,4].scatter(pl_mean[1], pl_mean[2], marker="x", color="red", alpha=1.0) 
    println("mean= $(pl_mean)")
    println("cov= $(pl_cov)")   
    println("-----------------")

    for i = 1:2, j=1:4
        ax[i,j].set_xlim(-2, 4)
        ax[i,j].set_ylim(-2, 4)
    end

    fig.tight_layout()
    fig.savefig("Plot_c=$(c).pdf")
end

close(io)


