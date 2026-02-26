import numpy as np
# from numba import jit
import time

class PottsModel2D:
    def __init__(self, N, q, J=1.0, h=0.0, lattice = None):
        self.N = N
        self.q = q
        self.J = J
        self.h = h
        self.k_beta = 1.0 
        self.beta = 1.0
        self.field = False
        if np.abs(h) > 1e-4:
            self.field = True
        if lattice is not None:
            if lattice.shape != (N, N):
                raise ValueError(f"传入的lattice尺寸 {lattice.shape} 与指定的N={N}不匹配")
            if np.any(lattice < 1) or np.any(lattice > q):
                raise ValueError(f"lattice中的值必须在1到{q}之间")
            self.lattice = lattice.copy()
        else:
            self.lattice = np.random.randint(1, q+1, size=(N, N))

        
    def set_temperature(self, T):
        self.beta = 1.0 /(self.k_beta * T)
    
    def wolff_flip(self, get_delta_H = False):
        N, q, beta, J = self.N, self.q, self.beta, self.J
        
        i, j = np.random.randint(0, N, 2)
        old_spin = self.lattice[i, j]
        
        current_set = set()
        boundary = set([(i, j)])
        current_set.add((i, j))
        
        while boundary:
            current_point = boundary.pop()
            i, j = current_point
            
            neighbors = [((i+1)%N, j), ((i-1)%N, j), (i, (j+1)%N), (i, (j-1)%N)]
            
            for nb in neighbors:
                ni, nj = nb
                if nb not in current_set and self.lattice[ni, nj] == old_spin:
                    if np.random.random() < 1.0 - np.exp(-beta * J):
                        current_set.add(nb)
                        boundary.add(nb)
        
        new_spin = np.random.randint(1, q+1)
        while new_spin == old_spin:
            new_spin = np.random.randint(1, q+1)
        
        energy_before = 0.0
        energy_after = 0.0

        if get_delta_H:
            for site in current_set:
                i, j = site
                neighbors = [((i+1)%N, j), ((i-1)%N, j), (i, (j+1)%N), (i, (j-1)%N)]
                
                for nb in neighbors:
                    if nb not in current_set:
                        ni, nj = nb
                        neighbor_spin = self.lattice[ni, nj]
                        
                        # 翻转前：所有current_set内自旋都是old_spin
                        if neighbor_spin == old_spin:
                            energy_before -= J
                        
                        # 翻转后：所有current_set内自旋变为new_spin
                        if neighbor_spin == new_spin:
                            energy_after -= J
            
                # 外场贡献
                energy_before -= self.h * old_spin
                energy_after -= self.h * new_spin
        
        # 应用翻转
        for site in current_set:
            i, j = site
            self.lattice[i, j] = new_spin
        
        delta_H = energy_after - energy_before
        return delta_H
    
    def metropolis_flip(self):
        N = self.N
        total_delta_H = 0.0
        for _ in range(N*N):
            i, j = np.random.randint(0, N, 2)
            old_spin = self.lattice[i, j]
            new_spin = np.random.randint(1, self.q+1)
            
            delta_H = 0.0
            
            neighbors = [((i+1)%N, j), ((i-1)%N, j), (i, (j+1)%N), (i, (j-1)%N)]
            for nb in neighbors:
                ni, nj = nb
                if self.lattice[ni, nj] == old_spin:
                    delta_H += self.J
                if self.lattice[ni, nj] == new_spin:
                    delta_H -= self.J
            
            delta_H += self.h * (old_spin - new_spin)
            
            if delta_H <= 0 or np.random.random() < np.exp(-self.beta * delta_H):
                self.lattice[i, j] = new_spin
                total_delta_H += delta_H

        return total_delta_H
    

    def func_H(self):
        """Hamiltonian量"""
        N = self.N
        energy = 0.0
        
        for i in range(N):
            for j in range(N):
                neighbors = [((i+1)%N, j), (i, (j+1)%N)]
                for nb in neighbors:
                    ni, nj = nb
                    if self.lattice[i, j] == self.lattice[ni, nj]:
                        energy -= self.J
                energy -= self.h * self.lattice[i, j]
        
        return energy
    
    def spin_sum(self):
        return np.sum(self.lattice)
    
    def correlation_compute(self, corr_k):
        corr_gamma_ = np.zeros(len(corr_k))
        N = self.N
        for id in range(0,len(corr_k)):
            k = corr_k[id]
            di, dj = k//2, k-k//2
            for i in range(N):
                for j in range(N):
                    temp = self.lattice[(i+k)%N, j] + self.lattice[i, (j+k)%N]+ self.lattice[(i+di)%N,(j+dj)%N] + self.lattice[(i-dj)%N,(j+di)%N]
                    corr_gamma_[id] += self.lattice[i,j] * temp
        return corr_gamma_



def temperature_scheduler(iter, n_iter, T_min, T_max, n_decay):
    """stable cos decay"""
    if iter <= n_iter - n_decay:
        return  T_max
    else:
        return T_min + 0.5 * (T_max - T_min) * (1 + np.cos(np.pi * (iter - (n_iter - n_decay)) / n_decay))


def mcmc_without_external_field(N, q, T,
    n_tempering=500, n_measure=1000, n_step=1, RATE = 3,
    lattice = None, mes_energy = True, mes_manetization = False, corr_k = [], get_energy = False):

    results = {
        'temperature': T,
        'internal_energy': 0,
        'specific_heat': 0,
        'magnetization': 0,
        'corr_k': corr_k,
        'corr_gamma': np.zeros(len(corr_k)),
        'Hamilton': []
    }
    
    print(f"{N}*{N} Potts模型MCMC: q={q}, T={T:.3f}, 测量共{n_measure}次")
    
    model = PottsModel2D(N, q, lattice= lattice)
    
    # 热化过程
    n_decay = round(n_tempering * 0.2) 
    T_min, T_max = T, max(1.5*T, 2.0)
    for step in range(n_tempering):

        T_tempering = temperature_scheduler(step, n_tempering, T_min, T_max, n_decay)
        model.set_temperature(T_tempering)
        model.wolff_flip()
    
    
    # 迭代测量过程
    model.set_temperature(T)

    energies = np.zeros(n_measure)
    manetizations = 0
    if len(corr_k)>0:
        lattice_sum = np.zeros((N, N))
        multiple_dis_k = np.zeros(len(corr_k))

    for step in range((RATE-1)*n_measure):
        for _ in range(n_step):
            model.wolff_flip()
        
    H = model.func_H()
    
    # 后(1/RATE)时间, 测量物理量并记录
    for step in range(n_measure):            

        for _ in range(n_step):
            H += model.wolff_flip(get_delta_H = mes_energy)
            
        if mes_energy:
            energy = H
            energies[step] = energy
        if mes_manetization:
            manetizations += model.spin_sum()
        if len(corr_k)>0:
            multiple_dis_k += model.correlation_compute(corr_k)
            lattice_sum += model.lattice

        if (step+1) % (n_measure/5) == 0:
            if mes_energy:
                print(f"迭代至第{((RATE-1)*n_measure+step+1)*n_step}/{RATE*n_measure*n_step}步, Hamilton量={energy}")
            else:
                print(f"迭代至第{((RATE-1)*n_measure+step+1)*n_step}/{RATE*n_measure*n_step}步")

    # 计算统计量
    internal_energy = np.mean(energies) / (N**2)
    specific_heat = np.cov(energies) * model.k_beta * model.beta**2 / (N**2)
    manetizations /= n_measure * N**2
    if len(corr_k)>0:
        multiple_dis_k /= 4 *n_measure * N**2
        lattice_mean = lattice_sum / n_measure
        for id in range(0,len(corr_k)):
            k = corr_k[id]
            di, dj = k//2, k-k//2
            
            for i in range(N):
                for j in range(N):
                    temp = lattice_mean[(i+k)%N, j] + lattice_mean[i, (j+k)%N]+ lattice_mean[(i+di)%N,(j+dj)%N] + lattice_mean[(i-dj)%N,(j+di)%N]
                    multiple_dis_k[id] -= lattice_mean[i,j] * temp / (4 * N**2)
    
    results['internal_energy'] = internal_energy
    results['specific_heat'] = specific_heat
    results['magnetization'] = manetizations
    if len(corr_k)>0:
        results['corr_gamma'] = multiple_dis_k
    
    if get_energy:
        return results, energies, model.lattice
    else:
        return results




def mcmc_with_external_field(N, q, T, h,  n_tempering=50, n_measure=200,
        n_step=1, RATE = 3, lattice = None, mes_manetization = True, get_lattice = False):
    
    print(f"MCMC模拟: q={q}, T={T:.3f}, 测量共{n_measure}次")
    
    model = PottsModel2D(N, q, h = h, lattice = lattice)
    
    # 热化过程
    n_decay = round(n_tempering * 0.2) 
    T_min, T_max = T, max(2*T, 5)
    for step in range(n_tempering):

        T_tempering = temperature_scheduler(step, n_tempering, T_min, T_max, n_decay)
        model.set_temperature(T_tempering)

        model.wolff_flip(get_delta_H = False)
        model.metropolis_flip()
    
    
    # 迭代测量过程
    model.set_temperature(T)
    energies = np.zeros(n_measure)
    manetizations = 0

    for step in range((RATE-1)*n_measure):
        
        for _ in range(n_step):
            model.wolff_flip(get_delta_H = False)
            model.metropolis_flip()

    for step in range(n_measure):
        
        for _ in range(n_step):
            model.wolff_flip(get_delta_H = False)
            model.metropolis_flip()
            
        if mes_manetization:
            manetizations += model.spin_sum()
       
        if (step+1) % (n_measure/5) == 0:
            print(f"迭代至第{((RATE-1)*n_measure+step+1)*n_step}/{RATE*n_measure*n_step}步, Hamilton量={model.func_H()}")
            


    # 计算统计量
    manetizations /= n_measure * N**2
   
    if get_lattice:
        return manetizations, model.lattice
    else:
        return manetizations




def mcmc_without_external_field_2(N, q, T,
    n_tempering=500, n_measure=1000, n_step=1, RATE = 3,
    lattice = None, mes_energy = True, mes_manetization = False, corr_k = [], get_energy = False):

    results = {
        'temperature': T,
        'internal_energy': 0,
        'specific_heat': 0,
        'magnetization': 0,
        'corr_k': corr_k,
        'corr_gamma': np.zeros(len(corr_k)),
        'Hamilton': []
    }
    
    print(f"{N}*{N} Potts模型MCMC: q={q}, T={T:.3f}, 测量共{n_measure}次")
    
    model = PottsModel2D(N, q, lattice= lattice)
    
    # 热化过程
    n_decay = round(n_tempering * 0.2) 
    T_min, T_max = T, max(1.5*T, 2.0)
    for step in range(n_tempering):

        T_tempering = temperature_scheduler(step, n_tempering, T_min, T_max, n_decay)
        model.set_temperature(T_tempering)
        model.metropolis_flip()
    
    
    # 迭代测量过程
    model.set_temperature(T)

    energies = np.zeros(n_measure)
    manetizations = 0
    if len(corr_k)>0:
        lattice_sum = np.zeros((N, N))
        multiple_dis_k = np.zeros(len(corr_k))

    for step in range((RATE-1)*n_measure):
        for _ in range(n_step):
            model.metropolis_flip()
        
    H = model.func_H()
    
    # 后(1/RATE)时间, 测量物理量并记录
    for step in range(n_measure):            

        for _ in range(n_step):
            H += model.metropolis_flip()
            
        if mes_energy:
            energy = H
            energies[step] = energy
        if mes_manetization:
            manetizations += model.spin_sum()
        if len(corr_k)>0:
            multiple_dis_k += model.correlation_compute(corr_k)
            lattice_sum += model.lattice

        if (step+1) % (n_measure/5) == 0:
            if mes_energy:
                print(f"迭代至第{((RATE-1)*n_measure+step+1)*n_step}/{RATE*n_measure*n_step}步, Hamilton量={energy}")
            else:
                print(f"迭代至第{((RATE-1)*n_measure+step+1)*n_step}/{RATE*n_measure*n_step}步")

    # 计算统计量
    internal_energy = np.mean(energies) / (N**2)
    specific_heat = np.cov(energies) * model.k_beta * model.beta**2 / (N**2)
    manetizations /= n_measure * N**2
    if len(corr_k)>0:
        multiple_dis_k /= 4 *n_measure * N**2
        lattice_mean = lattice_sum / n_measure
        for id in range(0,len(corr_k)):
            k = corr_k[id]
            di, dj = k//2, k-k//2
            
            for i in range(N):
                for j in range(N):
                    temp = lattice_mean[(i+k)%N, j] + lattice_mean[i, (j+k)%N]+ lattice_mean[(i+di)%N,(j+dj)%N] + lattice_mean[(i-dj)%N,(j+di)%N]
                    multiple_dis_k[id] -= lattice_mean[i,j] * temp / (4 * N**2)
    
    results['internal_energy'] = internal_energy
    results['specific_heat'] = specific_heat
    results['magnetization'] = manetizations
    if len(corr_k)>0:
        results['corr_gamma'] = multiple_dis_k
    
    if get_energy:
        return results, energies, model.lattice
    else:
        return results
