using LinearAlgebra
using ForwardDiff

mutable struct MLP
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
end

function MLP(input_dim::Int, output_dim::Int, hidden_dim::Int=32)
    W1 = randn(input_dim, hidden_dim) * 0.1
    b1 = zeros(hidden_dim)
    W2 = randn(hidden_dim, output_dim) * 0.1
    b2 = zeros(output_dim)
    return MLP(W1, b1, W2, b2)
end

function (mlp::MLP)(x::AbstractVector)
    h = tanh.(mlp.W1' * x .+ mlp.b1)
    return mlp.W2' * h .+ mlp.b2
end

struct AffineCouplingLayer
    s_net::MLP
    t_net::MLP
    mask::Vector{Bool}
    dim::Int
end

function AffineCouplingLayer(dim::Int, d::Int)
    mask = [i <= d for i in 1:dim]
    s_net = MLP(d, dim - d)
    t_net = MLP(d, dim - d)
    return AffineCouplingLayer(s_net, t_net, mask, dim)
end

function forward(layer::AffineCouplingLayer, x::AbstractVector)
    x1 = x[layer.mask]
    x2 = x[.!layer.mask]
    
    s = layer.s_net(x1)
    t = layer.t_net(x1)
    
    y2 = x2 .* exp.(s) .+ t
    y = similar(x)
    y[layer.mask] = x1
    y[.!layer.mask] = y2
    
    log_det = sum(s)
    return y, log_det
end

function forward(layer::AffineCouplingLayer, x::AbstractMatrix)
    n_samples = size(x, 2)
    y = similar(x)
    log_det = zeros(n_samples)
    
    for i in 1:n_samples
        yi, ld = forward(layer, x[:, i])
        y[:, i] = yi
        log_det[i] = ld
    end
    
    return y, log_det
end

struct PermutationLayer
    perm::Vector{Int}
    inv_perm::Vector{Int}
end

function PermutationLayer(dim::Int)
    perm = randperm(dim)
    inv_perm = zeros(Int, dim)
    for i in 1:dim
        inv_perm[perm[i]] = i
    end
    return PermutationLayer(perm, inv_perm)
end

function forward(perm::PermutationLayer, x::AbstractVector)
    return x[perm.perm], 0.0
end

function forward(perm::PermutationLayer, x::AbstractMatrix)
    return x[perm.perm, :], zeros(size(x, 2))
end

mutable struct NormalizingFlow
    coupling_layers::Vector{AffineCouplingLayer}
    permutation_layers::Vector{PermutationLayer}
    c::Float64
    dim::Int
    n_layers::Int
end

function NormalizingFlow(dim::Int, n_layers::Int, c::Float64=1.0)
    coupling_layers = []
    permutation_layers = []
    
    for i in 1:n_layers
        d = div(dim, 2)
        push!(coupling_layers, AffineCouplingLayer(dim, d))
        push!(permutation_layers, PermutationLayer(dim))
    end
    
    return NormalizingFlow(coupling_layers, permutation_layers, c, dim, n_layers)
end

function forward(flow::NormalizingFlow, z::AbstractVector)
    x = z
    sum_log_det = 0.0
    
    for i in 1:flow.n_layers
        x, ld = forward(flow.coupling_layers[i], x)
        sum_log_det += ld
        
        if i < flow.n_layers
            x, _ = forward(flow.permutation_layers[i], x)
        end
    end
    
    return x, sum_log_det
end

function forward(flow::NormalizingFlow, z::AbstractMatrix)
    x = z
    sum_log_det = zeros(size(z, 2))
    
    for i in 1:flow.n_layers
        x, ld = forward(flow.coupling_layers[i], x)
        sum_log_det .+= ld
        
        if i < flow.n_layers
            x, _ = forward(flow.permutation_layers[i], x)
        end
    end
    
    return x, sum_log_det
end

function log_prob_prior(z::AbstractVector, c::Float64)
    return -0.5 * dot(z, z) / c - 0.5 * length(z) * log(2 * pi * c)
end

function get_params(flow::NormalizingFlow)
    params = Float64[]
    for layer in flow.coupling_layers
        append!(params, vec(layer.s_net.W1))
        append!(params, layer.s_net.b1)
        append!(params, vec(layer.s_net.W2))
        append!(params, layer.s_net.b2)
        append!(params, vec(layer.t_net.W1))
        append!(params, layer.t_net.b1)
        append!(params, vec(layer.t_net.W2))
        append!(params, layer.t_net.b2)
    end
    return params
end

function set_params!(flow::NormalizingFlow, params::Vector{Float64})
    idx = 1
    d = div(flow.dim, 2)
    hidden_dim = 32
    
    for layer in flow.coupling_layers
        layer.s_net.W1 = reshape(params[idx:idx+d*hidden_dim-1], d, hidden_dim)
        idx += d * hidden_dim
        layer.s_net.b1 = params[idx:idx+hidden_dim-1]
        idx += hidden_dim
        layer.s_net.W2 = reshape(params[idx:idx+hidden_dim*(flow.dim-d)-1], hidden_dim, flow.dim-d)
        idx += hidden_dim * (flow.dim - d)
        layer.s_net.b2 = params[idx:idx+(flow.dim-d)-1]
        idx += (flow.dim - d)
        
        layer.t_net.W1 = reshape(params[idx:idx+d*hidden_dim-1], d, hidden_dim)
        idx += d * hidden_dim
        layer.t_net.b1 = params[idx:idx+hidden_dim-1]
        idx += hidden_dim
        layer.t_net.W2 = reshape(params[idx:idx+hidden_dim*(flow.dim-d)-1], hidden_dim, flow.dim-d)
        idx += hidden_dim * (flow.dim - d)
        layer.t_net.b2 = params[idx:idx+(flow.dim-d)-1]
        idx += (flow.dim - d)
    end
end

function apply_flow_with_params(params::AbstractVector, z::AbstractVector, flow::NormalizingFlow)
    x = z
    sum_log_det = 0.0
    d = div(flow.dim, 2)
    hidden_dim = 32
    idx = 1
    
    for layer_idx in 1:flow.n_layers
        s_W1 = reshape(params[idx:idx+d*hidden_dim-1], d, hidden_dim)
        idx += d * hidden_dim
        s_b1 = params[idx:idx+hidden_dim-1]
        idx += hidden_dim
        s_W2 = reshape(params[idx:idx+hidden_dim*(flow.dim-d)-1], hidden_dim, flow.dim-d)
        idx += hidden_dim * (flow.dim - d)
        s_b2 = params[idx:idx+(flow.dim-d)-1]
        idx += (flow.dim - d)
        
        t_W1 = reshape(params[idx:idx+d*hidden_dim-1], d, hidden_dim)
        idx += d * hidden_dim
        t_b1 = params[idx:idx+hidden_dim-1]
        idx += hidden_dim
        t_W2 = reshape(params[idx:idx+hidden_dim*(flow.dim-d)-1], hidden_dim, flow.dim-d)
        idx += hidden_dim * (flow.dim - d)
        t_b2 = params[idx:idx+(flow.dim-d)-1]
        idx += (flow.dim - d)
        
        x1 = x[1:d]
        x2 = x[d+1:end]
        
        h_s = tanh.(s_W1' * x1 .+ s_b1)
        s = s_W2' * h_s .+ s_b2
        
        h_t = tanh.(t_W1' * x1 .+ t_b1)
        t = t_W2' * h_t .+ t_b2
        
        x2_new = x2 .* exp.(s) .+ t
        x = vcat(x1, x2_new)
        
        sum_log_det += sum(s)
        
        if layer_idx < flow.n_layers
            perm = flow.permutation_layers[layer_idx].perm
            x = x[perm]
        end
    end
    
    return x, sum_log_det
end

function loss_with_params(params::AbstractVector, flow::NormalizingFlow, z_batch::AbstractMatrix, func_Phi::Function)
    n_samples = size(z_batch, 2)
    total_loss = 0.0
    
    for i in 1:n_samples
        z = z_batch[:, i]
        x, log_det = apply_flow_with_params(params, z, flow)
        log_likelihood = -0.5 * dot(z, z) / flow.c - 0.5 * flow.dim * log(2 * pi * flow.c)
        log_target = -func_Phi(x)
        total_loss += -(log_target - log_likelihood - log_det)
    end
    
    return total_loss / n_samples
end

function train!(flow::NormalizingFlow, func_Phi::Function; n_iter::Int=1000, batch_size::Int=32, lr::Float64=0.001)
    losses = Float64[]
    params = get_params(flow)
    
    for iter in 1:n_iter
        z_batch = randn(flow.dim, batch_size) * sqrt(flow.c)
        loss_val = loss_with_params(params, flow, z_batch, func_Phi)
        push!(losses, loss_val)
        
        grad = ForwardDiff.gradient(p -> loss_with_params(p, flow, z_batch, func_Phi), params)
        params .-= lr * grad
        set_params!(flow, params)
        
        if iter % 100 == 0
            @info "Iteration $iter/$n_iter, Loss: $loss_val"
        end
    end
    
    return losses
end

function sample_posterior(flow::NormalizingFlow, n_samples::Int)
    z_samples = randn(flow.dim, n_samples) * sqrt(flow.c)
    x_samples, _ = forward(flow, z_samples)
    return x_samples
end

function NF(init_samples, gradient)
    dim = size(init_samples, 1)
    c = 1.0
    n_layers = 4
    flow = NormalizingFlow(dim, n_layers, c)
    
    func_Phi(x) = gradient(x)
    losses = train!(flow, func_Phi, n_iter=20000, batch_size=200, lr=0.001)
    
    return flow, losses
end
