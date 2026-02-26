from Potts_Model_2D import mcmc_without_external_field
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    
    N = 100
    q = 3
    T = 0.95
    _, energies, lattice = mcmc_without_external_field(N, q, T, n_tempering=1000, n_measure=400, RATE=1, n_step=50, get_energy=True)   
    file_name = f'lattice/N={N},q={q},T={T}.npy'
    np.save(file_name, lattice)
    print(energies)

    N = 50
    q = 10
    T = 0.67
    _, energies, lattice = mcmc_without_external_field(N, q, T, n_tempering=1000, n_measure=800, RATE=1, n_step=50, get_energy=True)   
    file_name = f'lattice/N={N},q={q},T={T}.npy'
    np.save(file_name, lattice)
    print(energies)

    N = 50
    q = 10
    T = 0.705
    _, energies, lattice = mcmc_without_external_field(N, q, T, n_tempering=1000, n_measure=2000, RATE=1, n_step=50, get_energy=True)   
    file_name = f'lattice/N={N},q={q},T={T}.npy'
    np.save(file_name, lattice)
    print(energies)