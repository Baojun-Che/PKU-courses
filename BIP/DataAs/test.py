import numpy as np
import matplotlib.pyplot as plt
import os
from SimplpePendulumSimu import SPendSimu
from DS_algs import Var3D, KalmanFilter, ParticleFilter, ParticleFilterModify

np.random.seed(123)
# 确保Figs目录存在
os.makedirs('Figs', exist_ok=True)

# =========================
# 1. 模拟单摆生成观测
# =========================
dt = 0.2
delta = 0.1
gamma = 0.1  # 观测噪声标准差
n_steps = 200

pendulum = SPendSimu(dt=dt, init_u=np.pi/4, init_w=0.0, delta=delta)
pendulum.Euler_nstep(n_steps)
state = pendulum.state[1:]
state = np.array(state)

t_steps = np.arange(1, n_steps+1) * dt
# 观测 y = u + noise
y_obs_u = pendulum.observation(gamma=gamma, Obs_type='Obs1')
# 观测 y = w + noise
y_obs_w = pendulum.observation(gamma=gamma, Obs_type='Obs2')

# 绘制观测
plt.figure(figsize=(10,4))
plt.plot(t_steps, y_obs_u, marker='o', markersize=5, label='Obs: position(u)')
plt.plot(t_steps, y_obs_w, marker='x', markersize=5, label='Obs: velocity(w)')
plt.xlabel('Time')
plt.ylabel('Observation')
plt.legend()
plt.tight_layout()
plt.savefig('Figs/obs.pdf')

# =========================
# 2. 3D-Var 实验  - 
# =========================


# 测试 3D-Var
for obs_type in ['Position', 'Velocity']:

    if obs_type == 'Position':
        y_obs = y_obs_u
        H = np.array([[1,0]])
    else:
        y_obs = y_obs_w
        H = np.array([[0,1]])

    best_Cb = None
    best_err = np.inf
    m_best = None
    err_hist = []
    for Cb_test in np.linspace(0.001, 0.05, 50):
        m_hist = np.array(Var3D(pendulum, y_obs, Cb_diag = Cb_test, gamma=gamma, H=H))[1:]
        err = np.sum((m_hist - state)**2)
        err_hist.append(err)
        if err < best_err:
            best_Cb = Cb_test
            best_err = err
            m_best = m_hist

    print(f"{obs_type} obs, best_c: {best_Cb}, best_err: {best_err}")
    plt.figure(figsize=(10,4))
    plt.plot(np.linspace(0.001, 0.05, 50), err_hist, label='Error')
    plt.xlabel('c')
    plt.ylabel('error')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Figs/3D-Var-err-{obs_type}.pdf')

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10,4))
    ax[0].plot(state[:,0], label='True u')
    ax[0].plot(m_best[:,0], label='3D-Var estimate u')
    ax[0].legend()
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Position(u)')

    ax[1].plot(state[:,1], label='True w')
    ax[1].plot(m_best[:,1], label='3D-Var estimate w')  
    ax[1].legend()
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Velocity(w)')
    plt.tight_layout()
    plt.savefig(f'Figs/3D-Var-plt-{obs_type}.pdf')


# =========================
# 3. 简单卡尔曼滤波 (线性化)
# =========================

m_hist, cov_hist = KalmanFilter(pendulum, y_obs_u)
u_mean = np.array([m_hist[i+1][0] for i in range(n_steps)])
w_mean = np.array([m_hist[i+1][1] for i in range(n_steps)])
u_var = np.array([cov_hist[i+1][0][0] for i in range(n_steps)])
w_var = np.array([cov_hist[i+1][1][1] for i in range(n_steps)])
cov_norm = np.array([np.linalg.norm(cov_hist[i+1]) for i in range(n_steps)])

fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(14,4))
ax[0].plot(t_steps, state[:,0], label='True u', color='red')
ax[0].plot(t_steps, u_mean, label='KF estimate u', color='blue')
ax[0].fill_between(t_steps, u_mean - 3*np.sqrt(u_var), u_mean + 3*np.sqrt(u_var), alpha=0.3, color='blue')
ax[0].legend()
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Position(u)')

ax[1].plot(t_steps,state[:,1], label='True w', color='red')
ax[1].plot(t_steps, w_mean, label='KF estimate w', color='blue')
ax[1].fill_between(t_steps, w_mean - 3*np.sqrt(w_var), w_mean + 3*np.sqrt(w_var), alpha=0.3, color='blue')
ax[1].legend()
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Velocity(w)')

ax[2].plot(t_steps, cov_norm)
ax[2].set_xlabel('Time')
ax[2].set_ylabel('Covariance Norm')
plt.tight_layout()
plt.savefig(f'Figs/KF-plt.pdf')

# =========================
# 4. 粒子滤波框架
# =========================

N_ens_array = [5, 20, 50, 100]
fig, ax = plt.subplots(ncols=3, nrows=4, figsize=(14,12))
for (j, N_ens) in enumerate(N_ens_array):
    m_hist, cov_hist = ParticleFilter(pendulum, y_obs_u, N_particles=N_ens)
    u_mean = np.array([m_hist[i+1][0] for i in range(n_steps)])
    w_mean = np.array([m_hist[i+1][1] for i in range(n_steps)])
    u_var = np.array([cov_hist[i+1][0][0] for i in range(n_steps)])
    w_var = np.array([cov_hist[i+1][1][1] for i in range(n_steps)])
    cov_norm = np.array([np.linalg.norm(cov_hist[i+1]) for i in range(n_steps)])

    ax[j,0].plot(t_steps, state[:,0], label='True u', color='red')
    ax[j,0].plot(t_steps, u_mean, label='PF estimate u', color='blue')
    ax[j,0].fill_between(t_steps, u_mean - 3*np.sqrt(u_var), u_mean + 3*np.sqrt(u_var), alpha=0.3, color='blue')
    ax[j,0].set_xlabel('Time')
    ax[j,0].set_ylabel(f'J={N_ens}')

    ax[j,1].plot(t_steps,state[:,1], label='True w', color='red')
    ax[j,1].plot(t_steps, w_mean, label='PF estimate w', color='blue')
    ax[j,1].fill_between(t_steps, w_mean - 3*np.sqrt(w_var), w_mean + 3*np.sqrt(w_var), alpha=0.3, color='blue')
    ax[j,1].set_xlabel('Time')

    ax[j,2].plot(t_steps, cov_norm, label=f'Covariance Norm')
    ax[j,2].set_xlabel('Time')

    if j == 0:
        ax[j,1].legend()
        ax[j,2].legend()
        ax[j,0].legend()
plt.tight_layout()
plt.savefig(f'Figs/Particle-plt.pdf')



# # =========================
# # 4. 粒子滤波框架
# # =========================

N_ens_array = [5, 20, 50, 100]
fig, ax = plt.subplots(ncols=3, nrows=4, figsize=(14,12))
for (j, N_ens) in enumerate(N_ens_array):
    m_hist, cov_hist = ParticleFilterModify(pendulum, y_obs_u, N_particles=N_ens)
    u_mean = np.array([m_hist[i+1][0] for i in range(n_steps)])
    w_mean = np.array([m_hist[i+1][1] for i in range(n_steps)])
    u_var = np.array([cov_hist[i+1][0][0] for i in range(n_steps)])
    w_var = np.array([cov_hist[i+1][1][1] for i in range(n_steps)])
    cov_norm = np.array([np.linalg.norm(cov_hist[i+1]) for i in range(n_steps)])

    ax[j,0].plot(t_steps, state[:,0], label='True u', color='red')
    ax[j,0].plot(t_steps, u_mean, label='KF estimate u', color='blue')
    ax[j,0].fill_between(t_steps, u_mean - 3*np.sqrt(u_var), u_mean + 3*np.sqrt(u_var), alpha=0.3, color='blue')
    ax[j,0].set_xlabel('Time')
    ax[j,0].set_ylabel(f'J={N_ens}')

    ax[j,1].plot(t_steps,state[:,1], label='True w', color='red')
    ax[j,1].plot(t_steps, w_mean, label='KF estimate w', color='blue')
    ax[j,1].fill_between(t_steps, w_mean - 3*np.sqrt(w_var), w_mean + 3*np.sqrt(w_var), alpha=0.3, color='blue')
    ax[j,1].set_xlabel('Time')

    ax[j,2].plot(t_steps, cov_norm, label=f'Covariance Norm')
    ax[j,2].set_xlabel('Time')

    if j == 0:
        ax[j,1].legend()
        ax[j,2].legend()
        ax[j,0].legend()

plt.tight_layout()
plt.savefig(f'Figs/ParticleModify-plt.pdf')
