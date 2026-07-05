import time
import numpy as np
import scipy as sci
import itertools as itt
import matplotlib.pyplot as plt
import multiprocessing as mp


# ---------------------filling-----------------------
N_x = 4                # system size along b1 direction
N_y = 6                # system size along b2 direction
N = 8                  # number of electrons
N_uc = N_x * N_y       # number of unit cells
N_A = 2                # number of electrons of part A
N_B = N - N_A          # number of electrons of part B


def get_bit(state, pos):
    pointer = 1 << pos
    return (state & pointer) >> pos


def flip_bit(state, pos1, pos2):
    pointer = 2 ** pos1 + 2 ** pos2
    return state ^ pointer


def configuration(state, filling_number):
    config = []
    str_num = str(bin(state))[::-1]
    num = 0
    position = str_num.find("1")
    config.append(position)
    while num < filling_number - 1:
        position = str_num.find("1", position + 1)
        config.append(position)
        num += 1
    return config


def generate_basis(momentum_x, momentum_y, size_x, size_y, filling_number):
    base = []
    config_list = []
    for ii in range(2 ** (size_x * size_y)):
        if bin(ii).count("1") == filling_number:
            momentum_config = configuration(ii, filling_number)
            if np.sum(np.array(momentum_config) % size_x) % size_x == momentum_x and np.sum(np.array(momentum_config) // size_x) % size_y == momentum_y:
                base.append(ii)
                config_list.append(momentum_config)
    return base, config_list


def generate_reduced_basis(size_x, size_y, number_of_A):
    base = []
    config_list = []
    for ky in range(size_y):
        for kx in range(size_x):
            basis = []
            config = []
            for ii in range(2 ** (size_x * size_y)):
                if bin(ii).count("1") == number_of_A:
                    momentum_config = configuration(ii, number_of_A)
                    if np.sum(np.array(momentum_config) % size_x) % size_x == kx and np.sum(np.array(momentum_config) // size_x) % size_y == ky:
                        basis.append(ii)
                        config.append(momentum_config)
            base.append(basis)
            config_list.append(config)
    return base, config_list


def particle_entanglement_spectrum(size_x, size_y, momentum_x, momentum_y, base_row, configs_row, base_column, configs_column, base_sector, ground_states, ground_sector):
    basis_row = base_row[momentum_y * size_x + momentum_x]
    config_row = configs_row[momentum_y * size_x + momentum_x]
    basis_column = []
    config_column = []
    reduced_ground_sector = []  # non-repetitive momentum sectors of degenerate ground states
    for sector in ground_sector:
        if sector not in reduced_ground_sector:
            reduced_ground_sector.append(sector)
    for sector in reduced_ground_sector:
        momentum_x_col = sector[0] - momentum_x
        momentum_y_col = sector[1] - momentum_y
        if momentum_x_col < 0:
            momentum_x_col += size_x
        if momentum_y_col < 0:
            momentum_y_col += size_y
        basis_column.append(base_column[momentum_y_col * size_x + momentum_x_col])
        config_column.append(configs_column[momentum_y_col * size_x + momentum_x_col])
    length_row = len(basis_row)
    reduced_density_matrix = np.zeros((length_row, length_row), dtype=complex)
    for state in range(len(ground_states)):
        sector = reduced_ground_sector.index(ground_sector[state])
        particle_matrix = np.zeros((length_row, len(basis_column[sector])), dtype=complex)
        basis_sector = np.array(base_sector[sector])
        for ii in range(length_row):
            for jj in range(len(basis_column[sector])):
                position = np.where(basis_sector == basis_row[ii] + basis_column[sector][jj])
                if len(position[0]) > 0:
                    sign = 0
                    for aa in range(len(config_row[ii])):
                        sign += np.searchsorted(config_column[sector][jj], config_row[ii][aa])
                    particle_matrix[ii, jj] += ground_states[state][position[0][0]] * (-1) ** sign
        density_matrix = np.dot(particle_matrix, np.conjugate(particle_matrix.T))
        reduced_density_matrix[:, :] += density_matrix
    reduced_density_matrix = reduced_density_matrix / len(ground_states)
    singular_values = np.real(np.linalg.eigvalsh(reduced_density_matrix))
    entanglement_spectrum = []
    momentum_sector = []
    for mm in range(len(singular_values)):
        if singular_values[mm] > 1e-10:
            momentum_sector.append(momentum_y * size_x + momentum_x)
            entanglement_spectrum.append(-np.log(singular_values[mm]))
    return momentum_sector, entanglement_spectrum


if __name__ == '__main__':
    start_t = time.time()
    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    num_cores_used = 12
    print("当前正在使用" + str(num_cores_used) + "个核心")
    pool = mp.Pool(num_cores_used)
    # ---------------------basis generation-----------------------
    # need to load the degenerate ground states from file and corresponding momentum sectors by hand
    ground_states_list = []  # degenerate ground states
    degenerate_ground_states_one = np.loadtxt("eleventh_band_ground_state_Nx4_Ny6_N8_sector0.txt", dtype=complex)
    degenerate_ground_states_two = np.loadtxt("eleventh_band_ground_state_Nx4_Ny6_N8_sector8.txt", dtype=complex)
    degenerate_ground_states_three = np.loadtxt("eleventh_band_ground_state_Nx4_Ny6_N8_sector16.txt", dtype=complex)
    ground_states_list.append(degenerate_ground_states_one)
    ground_states_list.append(degenerate_ground_states_two)
    ground_states_list.append(degenerate_ground_states_three)
    ground_states_sector = [[0, 0], [0, 2], [0, 4]]  # momentum sectors of degenerate ground states
    reduced_ground_states_sector = []  # non-repetitive momentum sectors of degenerate ground states
    basis_ground_sector = []  # basis of momentum sector of degenerate ground states
    for i in range(len(ground_states_sector)):
        if ground_states_sector[i] not in reduced_ground_states_sector:
            reduced_ground_states_sector.append(ground_states_sector[i])
            basis_ground_sector.append(generate_basis(ground_states_sector[i][0], ground_states_sector[i][1], N_x, N_y, N)[0])
    basis_A, config_A = generate_reduced_basis(N_x, N_y, N_A)
    basis_B, config_B = generate_reduced_basis(N_x, N_y, N_B)
    # ---------------------multiprocessing------------------------
    param_list = []
    for j in range(N_y):
        for i in range(N_x):
            param_list.append((N_x, N_y, i, j, basis_A, config_A, basis_B, config_B, basis_ground_sector, ground_states_list, ground_states_sector))
    results = [pool.apply_async(particle_entanglement_spectrum, args=param) for param in param_list]
    pool.close()
    pool.join()
    results = [p.get() for p in results]
    print(results)
    end_t = time.time()
    elapsed_sec = (end_t - start_t)
    print("多进程计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")
    total_momentum = []
    entanglement_spectrum_list = []
    for i in range(len(results)):
        total_momentum += results[i][0]
        entanglement_spectrum_list += results[i][1]
    entanglement_gap = 4
    counting = 0
    for i in range(len(entanglement_spectrum_list)):
        if entanglement_spectrum_list[i] < entanglement_gap:
            counting += 1
    print("total counting:", counting)
    # np.savetxt("eleventh_band_Nx4_Ny6_N8_entanglement_spectrum.txt", entanglement_spectrum_list, fmt='%.8f')
    # np.savetxt("eleventh_band_ES_Nx4_Ny6_N8_total_momentum.txt", total_momentum, fmt='%.8f')

    # plt.subplot(1, 2, 1)
    plt.scatter(total_momentum, entanglement_spectrum_list, color='r', marker='_')
    plt.xlabel(r"$K_x + N_x K_y$")
    plt.ylabel(r"$\xi$")
    x_major_locator = plt.MultipleLocator(5)
    x_minor_locator = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(x_minor_locator)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()


