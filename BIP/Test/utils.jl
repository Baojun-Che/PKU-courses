function cal_mean_cov(func_Phi, x_lim = [-2, 4], y_lim = [-2, 4], N = 200)
    # 生成网格点
    x = range(x_lim[1], x_lim[2]; length = N)
    y = range(y_lim[1], y_lim[2]; length = N)
    dx = (x_lim[2] - x_lim[1]) / (N - 1)
    dy = (y_lim[2] - y_lim[1]) / (N - 1)

    # 计算 Phi 和未归一化的概率密度 rho ∝ exp(-Phi)
    Phi_vals = [func_Phi([xi, yi]) for yi in y, xi in x]  # 注意：行是 y，列是 x
    p_unnorm = exp.(-Phi_vals)

    # 归一化常数 Z（积分近似）
    Z = sum(p_unnorm) * dx * dy
    p_norm = p_unnorm / Z  # 归一化后的概率密度，形状 (Ny, Nx)

    # 计算均值 ⟨x⟩, ⟨y⟩
    # 对 y 求和（沿行，dims=1）得到每个 x 的边缘分布
    marginal_x = vec(sum(p_norm, dims = 1)) * dy * dx   # 长度 N，对应 x
    marginal_y = vec(sum(p_norm, dims = 2)) * dx * dy   # 长度 N，对应 y

    mean_x = dot(marginal_x, x)   # ∑ p(x) * x
    mean_y = dot(marginal_y, y)   # ∑ p(y) * y

    # 构建网格矩阵用于协方差计算（与 p_norm 同形状）
    xx = [xi for yi in y, xi in x]  # 每行重复 x
    yy = [yi for yi in y, xi in x]  # 每列重复 y

    # 协方差计算（期望 E[(X - μx)(Y - μy)]）
    cov_xx = sum((xx .- mean_x).^2 .* p_norm) * dx * dy
    cov_yy = sum((yy .- mean_y).^2 .* p_norm) * dx * dy
    cov_xy = sum((xx .- mean_x) .* (yy .- mean_y) .* p_norm) * dx * dy

    cov_matrix = [cov_xx cov_xy;
                  cov_xy cov_yy]

    # 返回：
    # - 均值向量
    # - 协方差矩阵
    # - 网格向量 x, y（用于 heatmap 的坐标轴）
    # - 归一化密度矩阵 p_norm（注意：heatmap 通常需转置或调整方向）
    return [mean_x, mean_y], cov_matrix, x, y, p_norm
end

function plot_Gaussian(ax, x_mean, xx_cov, color_lim = nothing)
    
    inv_xx_cov = inv(xx_cov)
    p(x) = (x-x_mean)' * inv_xx_cov * (x-x_mean)    
    _, _,  X, Y, Z = cal_mean_cov(p)
    ax.pcolormesh(X, Y, Z, cmap="viridis", clim=color_lim)
    ax.scatter(x_mean[1], x_mean[2], marker="x", color="red", alpha=1.0) 

end

function ens_mean_cov(samples)
    mean_samples = dropdims(mean(samples, dims=2), dims=2)
    cov_samples = cov(samples, dims=2)
    return mean_samples, cov_samples
end
 
function error_plot(his_mean, his_cov, mean_ref, cov_ref, N_ens_array, name)
    N = length(his_mean)
    errors = zeros(N, 2)
    for i in 1:N
        errors[i, 1] = norm(his_mean[i] - mean_ref)
        errors[i, 2] = norm(his_cov[i] - cov_ref)
    end
    fig, ax = PyPlot.subplots(figsize=(5,5))
    ax.loglog(N_ens_array, errors[:, 1], label="mean error")
    ax.loglog(N_ens_array, errors[:, 2], label="cov error")
    ax.legend()
    ax.set_xlabel("number of samples")
    ax.set_ylabel("errors")
    fig.tight_layout()
    fig.savefig(name*"_error_plot.pdf")
end
