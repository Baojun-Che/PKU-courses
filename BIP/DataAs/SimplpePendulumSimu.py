import numpy as np
import copy

class SPendSimu:
    """
    Evaluation:
        dot u = w
        dot w = -delta w - sin u
    Observation:
        (Obs1) y = u + noise; noise ~ N(0, gamma^2)
        (Obs2) y = w + noise; noise ~ N(0, gamma^2)
    """
    def __init__(self, dt = 0.2, init_u = np.pi/4, init_w = 0.0, delta=0.1):
        self.dt = dt
        self.state = []
        self.state.append(np.array([init_u, init_w]))
        self.delta = delta

    def Euler_forward(self, state):
        u, w = state
        u_next = u + w * self.dt
        w_next = w + (-self.delta * w - np.sin(u)) * self.dt
        return np.array([u_next, w_next])
    
    def Euler_nstep(self, n_steps):
        for _ in range(n_steps):
             self.state.append(self.Euler_forward(self.state[-1]))
        return self.state[-1]
    
    def observation(self, gamma=0.1, Obs_type = 'Obs1', seed = 123, drop_init = True):

        np.random.seed(seed)
        L = len(self.state)
        noise = np.random.normal(0, gamma, L) 
        obs = np.zeros(L)
        for i in range(L):
            u, w = self.state[i]
            if Obs_type == 'Obs1':
                obs[i] = u + noise[i]
            elif Obs_type == 'Obs2':
                obs[i] = w + noise[i]

        if drop_init:
            obs = obs[1:]
        return obs
    