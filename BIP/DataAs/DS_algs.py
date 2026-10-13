import numpy as np
import copy
from SimplpePendulumSimu import SPendSimu

def ParticleFilter(pendulum, y_obs, N_particles=50, gamma=0.1):
    particles = np.random.normal(0, 1, size=(N_particles,2))
    weights = np.ones(N_particles)/N_particles
    mean_hist = [np.mean(particles, axis=0)]
    cov_hist = [np.cov(particles.T)]

    for yj in y_obs:
        # 预测
        for i in range(N_particles):
            particles[i] = pendulum.Euler_forward(particles[i])
        # 计算权重（使用对数空间避免数值下溢）
        obs_pred = particles[:,0]
        log_weights = -0.5 * ((yj - obs_pred) / gamma)**2
        log_weights -= np.max(log_weights)  # 数值稳定性：减去最大值
        weights = np.exp(log_weights)
        weights_sum = np.sum(weights)
        # 处理权重和为0的情况
        if weights_sum == 0 or np.isnan(weights_sum):
            weights = np.ones(N_particles) / N_particles
        else:
            weights /= weights_sum
        # 重采样
        idx = np.random.choice(N_particles, N_particles, p=weights)
        particles = particles[idx]
        # 重置权重
        weights = np.ones(N_particles)/N_particles
        # 估计
        est = np.mean(particles, axis=0)
        mean_hist.append(est)
        cov_hist.append(np.cov(particles.T))
    return mean_hist, cov_hist

def effective_sample_size(weights):
    return 1.0 / np.sum(weights**2)

def ParticleFilterModify(pendulum, y_obs, N_particles=50, gamma=0.1, process_noise=0.01):
    particles = np.random.normal(0, 1, size=(N_particles,2))
    weights = np.ones(N_particles)/N_particles
    mean_hist = [np.mean(particles, axis=0)]
    cov_hist = [np.cov(particles.T)]

    for yj in y_obs:
        # 预测（添加过程噪声）
        for i in range(N_particles):
            particles[i] = pendulum.Euler_forward(particles[i])
            particles[i] += np.random.normal(0, process_noise, size=2)
        
        # 计算权重（数值稳定）
        obs_pred = particles[:,0]
        log_weights = -0.5 * ((yj - obs_pred) / gamma)**2
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights_sum = np.sum(weights)
        
        if weights_sum == 0 or np.isnan(weights_sum):
            weights = np.ones(N_particles) / N_particles
        else:
            weights /= weights_sum
        
        # 自适应重采样
        ess = effective_sample_size(weights)
        if ess < N_particles * 0.5:
            idx = np.random.choice(N_particles, N_particles, p=weights)
            particles = particles[idx]
            # 重采样后添加扰动
            particles += np.random.normal(0, process_noise*0.5, size=particles.shape)
            weights = np.ones(N_particles)/N_particles
        
        # 估计（带正则化的协方差）
        est = np.mean(particles, axis=0)
        cov_mat = np.cov(particles.T) + np.eye(2) * 1e-6  # 正则化
        mean_hist.append(est)
        cov_hist.append(cov_mat)
    
    return mean_hist, cov_hist

def KalmanFilter(pendulum, y_obs, Q=np.eye(2)*1e-5, gamma=0.1, R=0.01, dt=0.2, delta=0.1, init_state=np.zeros(2)):
    """线性化卡尔曼滤波示例"""
    m = init_state.copy()
    P = np.eye(2)
    m_list = [m.copy()]
    cov_list = [P.copy()]

    for j, yj in enumerate(y_obs):
        # 预测
        v_pred = pendulum.Euler_forward(m)
        F = np.array([[1, dt],[ -dt*np.cos(m[0]), 1 - dt*delta]])  # 线性化
        P_pred = F @ P @ F.T + Q
        # 更新
        H = np.array([[1,0]])
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        m = v_pred + (K.flatten()) * (yj - H @ v_pred)
        P = (np.eye(2) - K @ H) @ P_pred
        m_list.append(m.copy())
        cov_list.append(P.copy())
    return m_list, cov_list


def Var3D(pendulum, y_obs, Cb_diag=1.0, init_state=np.zeros(2), gamma=0.1, H = np.array([[1,0]])):
    """简单 3D-Var 实现"""
    Cb = np.diag([Cb_diag, Cb_diag])
    m = init_state.copy()
    m_hist = [m.copy()]

    for j, yj in enumerate(y_obs):
        # 预测
        v_pred = pendulum.Euler_forward(m)
        # 计算增益
        S = H @ Cb @ H.T + gamma**2
        K = Cb @ H.T @ np.linalg.inv(S)
        # 更新
        m = v_pred + (K.flatten()) * (yj - H @ v_pred)
        m_hist.append(m.copy())
    return m_hist