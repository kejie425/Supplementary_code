import numpy as np
import scipy as sci
import itertools as itt
import matplotlib.pyplot as plt
import time
import pythtb as tb
import multiprocessing as mp


# ------------------import wannier90 module-----------------
frozen = tb.w90(r"./DFT_data/frozen_afm/frozen", r"wannier90")
my_model = frozen.model(min_hopping_norm=0.0001)
# my_model.display()
my_model_reduced = my_model.reduce_dim(2, 0)
# my_model_reduced.display()
my_model_reduced.ignore_position_operator_offdiagonal()
fermi_energy = -4.42

k_path = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.333, 0.333, 0.0], [0.0, 0.0, 0.0]]
k_label = (r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$')  # labels of the nodes


number_of_orbitals = my_model.get_num_orbitals()
# print("Number of orbitals:", number_of_orbitals)
orbitals = my_model.get_orb()
# print("Orbitals:", orbitals)
a1 = np.array([11.6963, 0, 0])
a2 = np.array([-5.84815, 10.12929, 0])
fig1 = plt.figure(figsize=(8, 8))
ax = fig1.add_subplot(111)
xx = []
yy = []
for i in range(number_of_orbitals):
    xx.append(orbitals[i][0] * a1[0] + orbitals[i][1] * a2[0])
    yy.append(orbitals[i][0] * a1[1] + orbitals[i][1] * a2[1])
ax.scatter(xx, yy, s=100, c='r')
ax.axis('equal')


# ------------------band structure------------------
def band_structure(path, label):
    (k_vec, k_dist, k_node) = my_model.k_path(path, 202)  # call function k_path to construct the actual path
    # print("k_vec.shape:", k_vec.shape)
    # print("k_dist.shape:", k_dist.shape)
    # print("k_node.shape:", k_node.shape)
    energy, vectors = my_model.solve_all(k_vec, eig_vectors=True)
    # print("energy.shape:", energy.shape)
    # print("vectors.shape:", vectors.shape)
    # print("target:", energy[9, 0] - fermi_energy)
    # energy_test, vectors_test = my_model.solve_one([0, 0, 0], eig_vectors=True)
    # print("target:", energy_test[9] - fermi_energy)
    # print("vectors_norm:", np.dot(vectors_test[9, :], np.conj(vectors_test[10, :])))
    for ii in range(energy.shape[0]):
        for jj in range(len(k_dist)):
            energy[ii][jj] = energy[ii][jj] - fermi_energy
    fig, axx = plt.subplots()
    for ii in range(energy.shape[0]):
        axx.plot(k_dist, energy[ii], "k-")
    for n in range(len(k_node)):
        axx.axvline(x=k_node[n], linewidth=0.5, color='k')
    axx.set_xlabel("Path in k-space")
    axx.set_ylabel("Band energy (eV)")
    axx.set_xlim(k_dist[0], k_dist[-1])
    axx.set_xticks(k_node)
    axx.set_xticklabels(label)
    fig.tight_layout()
    plt.show()


# ------------------filling--------------------
N_x = 4  # system size along x-axis
N_y = 6  # system size along y-axis
N = 8    # number of electrons
bm = 0.09872   # length of reciprocal lattice vector
b1 = bm * np.array([1/2, -np.sqrt(3)/2])
b2 = bm * np.array([1/2, np.sqrt(3)/2])


def get_bit(state, pos):
    """
    This is a function to get the bit located in particular position of a state.
    :param state: a binary number describing a basis.
    :param pos: the position of desired bit.
    :return: the bit located in particular position.
    """
    pointer = 1 << pos
    return (state & pointer) >> pos


def flip_bit(state, pos1, pos2):
    pointer = 2 ** pos1 + 2 ** pos2
    return state ^ pointer


def basis_momentum_list(size_x, size_y):
    momentum_pair = []
    for jj in range(size_y):
        for ii in range(size_x):
            momentum_pair.append([ii, jj])
    return momentum_pair


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


def basis_transformation(size_x, size_y, flux, target_band):
    coefficient = []
    for jj in range(size_y):
        for ii in range(size_x):
            energy, vectors = my_model.solve_one([ii/size_x + flux, jj/size_y, 0], eig_vectors=True)
            coefficient.append(vectors[target_band, :])
    return coefficient


def fermi_exchange_sign(state_initial, interation_set):
    sign = 0
    if interation_set[2] > interation_set[0]:
        sign1 = bin((2 ** interation_set[2] - 2 ** (interation_set[0] + 1)) & state_initial).count("1")
        test_state = flip_bit(state_initial, interation_set[2], interation_set[0])
        if interation_set[3] > interation_set[1]:
            sign2 = bin((2 ** interation_set[3] - 2 ** (interation_set[1] + 1)) & test_state).count("1")
            sign = sign + sign1 + sign2
        else:
            sign2 = bin((2 ** interation_set[1] - 2 ** (interation_set[3] + 1)) & test_state).count("1")
            sign = sign + sign1 + sign2
    else:
        sign1 = bin((2 ** interation_set[0] - 2 ** (interation_set[2] + 1)) & state_initial).count("1")
        test_state = flip_bit(state_initial, interation_set[0], interation_set[2])
        if interation_set[3] > interation_set[1]:
            sign2 = bin((2 ** interation_set[3] - 2 ** (interation_set[1] + 1)) & test_state).count("1")
            sign = sign + sign1 + sign2
        else:
            sign2 = bin((2 ** interation_set[1] - 2 ** (interation_set[3] + 1)) & test_state).count("1")
            sign = sign + sign1 + sign2
    return sign


def phase_factor(size_x, size_y, k1, k2, k3, k4, orbital1, orbital2):
    factor = np.exp(-1j * 2 * np.pi * (np.dot(np.array([(k1 % size_x - k2 % size_x) / size_x, (k1 // size_x - k2 // size_x) / size_y, 0]), orbitals[orbital1]) + np.dot(np.array([(k3 % size_x - k4 % size_x) / size_x, (k3 // size_x - k4 // size_x) / size_y, 0]), orbitals[orbital2])))
    return factor


def self_energy(size_x, size_y, coefficients):
    index_list = []
    value_list = []
    for ii in itt.combinations(range(size_x * size_y), 2):
        index_list.append(list(ii))
        value1 = 0
        value2 = 0
        for mm in range(3):
            for jj in range(int(number_of_orbitals / 3)):
                for kk in range(jj + 1, int(number_of_orbitals / 3)):
                    value1 += (coefficients[ii[0]][mm * int(number_of_orbitals / 3) + jj] * np.conj(coefficients[ii[0]][mm * int(number_of_orbitals / 3) + jj]) * coefficients[ii[1]][mm * int(number_of_orbitals / 3) + kk] * np.conj(coefficients[ii[1]][mm * int(number_of_orbitals / 3) + kk])
                                + coefficients[ii[1]][mm * int(number_of_orbitals / 3) + jj] * np.conj(coefficients[ii[1]][mm * int(number_of_orbitals / 3) + jj]) * coefficients[ii[0]][mm * int(number_of_orbitals / 3) + kk] * np.conj(coefficients[ii[0]][mm * int(number_of_orbitals / 3) + kk]))
                    value2 += ((-1) * phase_factor(size_x, size_y, ii[0], ii[1], ii[1], ii[0], mm * int(number_of_orbitals / 3) + jj, mm * int(number_of_orbitals / 3) + kk) *
                                np.conj(coefficients[ii[0]][mm * int(number_of_orbitals / 3) + jj]) * coefficients[ii[1]][mm * int(number_of_orbitals / 3) + jj] * np.conj(coefficients[ii[1]][mm * int(number_of_orbitals / 3) + kk]) * coefficients[ii[0]][mm * int(number_of_orbitals / 3) + kk])
        value = np.real(value1 + (value2 + np.conj(value2))) / (size_x * size_y) / 2  # divided by 2 due to Hermitian
        value_list.append(value)
    return index_list, value_list


def inter_energy(size_x, size_y, coefficients):
    index_list = []
    value_list = []
    for ii in range(size_x * size_y):
        for jj in range(ii + 1, size_x * size_y):
            for mm in range(size_x * size_y):
                if mm != ii and mm != jj:
                    for nn in range(mm + 1, size_x * size_y):
                        if abs((ii % size_x + jj % size_x) - (mm % size_x + nn % size_x)) % size_x == 0 and abs((ii // size_x + jj // size_x) - (mm // size_x + nn // size_x)) % size_y == 0:
                            form_factor = 0
                            for alpha in range(3):
                                for beta in range(int(number_of_orbitals / 3)):
                                    for gamma in range(beta + 1, int(number_of_orbitals / 3)):
                                        form_factor += (phase_factor(size_x, size_y, nn, jj, mm, ii, alpha * int(number_of_orbitals / 3) + beta, alpha * int(number_of_orbitals / 3) + gamma) *
                                                        np.conj(coefficients[nn][alpha * int(number_of_orbitals / 3) + beta]) * coefficients[jj][alpha * int(number_of_orbitals / 3) + beta] * np.conj(coefficients[mm][alpha * int(number_of_orbitals / 3) + gamma]) * coefficients[ii][alpha * int(number_of_orbitals / 3) + gamma])
                                        form_factor += (phase_factor(size_x, size_y, mm, ii, nn, jj, alpha * int(number_of_orbitals / 3) + beta, alpha * int(number_of_orbitals / 3) + gamma) *
                                                        np.conj(coefficients[mm][alpha * int(number_of_orbitals / 3) + beta]) * coefficients[ii][alpha * int(number_of_orbitals / 3) + beta] * np.conj(coefficients[nn][alpha * int(number_of_orbitals / 3) + gamma]) * coefficients[jj][alpha * int(number_of_orbitals / 3) + gamma])
                                        form_factor += ((-1) * phase_factor(size_x, size_y, mm, jj, nn, ii, alpha * int(number_of_orbitals / 3) + beta, alpha * int(number_of_orbitals / 3) + gamma) *
                                                        np.conj(coefficients[mm][alpha * int(number_of_orbitals / 3) + beta]) * coefficients[jj][alpha * int(number_of_orbitals / 3) + beta] * np.conj(coefficients[nn][alpha * int(number_of_orbitals / 3) + gamma]) * coefficients[ii][alpha * int(number_of_orbitals / 3) + gamma])
                                        form_factor += ((-1) * phase_factor(size_x, size_y, nn, ii, mm, jj, alpha * int(number_of_orbitals / 3) + beta, alpha * int(number_of_orbitals / 3) + gamma) *
                                                        np.conj(coefficients[nn][alpha * int(number_of_orbitals / 3) + beta]) * coefficients[ii][alpha * int(number_of_orbitals / 3) + beta] * np.conj(coefficients[mm][alpha * int(number_of_orbitals / 3) + gamma]) * coefficients[jj][alpha * int(number_of_orbitals / 3) + gamma])
                            index_list.append([ii, jj, mm, nn])
                            value_list.append(form_factor / (size_x * size_y))
    return index_list, value_list


def exact_diagonal(size_x, size_y, filling_number, k_x, k_y, target_band, name):
    momentum_sector = []
    energy_sector = []
    transformation_coefficient = basis_transformation(size_x, size_y, 0, target_band)
    basis, configuration_list = generate_basis(k_x, k_y, size_x, size_y, filling_number)
    self_index, self_value = self_energy(size_x, size_y, transformation_coefficient)
    inter_index, inter_value = inter_energy(size_x, size_y, transformation_coefficient)
    dimension = len(basis)
    matrix_x = []
    matrix_y = []
    matrix_element = []
    for ii in range(dimension):
        config1 = configuration_list[ii]
        for mm in range(filling_number):
            for nn in range(mm + 1, filling_number):
                position = self_index.index([config1[mm], config1[nn]])
                value = self_value[position]
                if abs(value) > 1e-10:
                    matrix_x.append(ii)
                    matrix_y.append(ii)
                    matrix_element.append(value)
        for jj in range(ii + 1, dimension):
            inter = basis[ii] & basis[jj]
            if bin(inter).count("1") == filling_number - 2:
                interset = []
                str1 = str(bin(basis[ii] ^ inter))[::-1]
                position1 = str1.find("1")
                interset.append(position1)
                interset.append(str1.find("1", position1 + 1))
                str2 = str(bin(basis[jj] ^ inter))[::-1]
                position2 = str2.find("1")
                interset.append(position2)
                interset.append(str2.find("1", position2 + 1))
                sign = fermi_exchange_sign(basis[ii], interset)
                position = inter_index.index(interset)
                value = inter_value[position] * (-1) ** sign
                if abs(value) > 1e-10:
                    matrix_x.append(jj)
                    matrix_y.append(ii)
                    matrix_element.append(value)
    ham = sci.sparse.coo_matrix((matrix_element, (matrix_x, matrix_y)), shape=(dimension, dimension), dtype=complex)
    hamiltonian = ham + (np.conjugate(ham)).T
    eigen_energy, eigen_vectors = sci.sparse.linalg.eigsh(hamiltonian, k=10, which="SA", return_eigenvectors=True)
    # if k_y * size_x + k_x == 15 or k_y * size_x + k_x == 16 or k_y * size_x + k_x == 17 or k_y * size_x + k_x == 18 or k_y * size_x + k_x == 19:
    # if k_y * size_x + k_x == 0 or k_y * size_x + k_x == 8 or k_y * size_x + k_x == 16:
    #     np.savetxt("eleventh_band_ground_state_Nx4_Ny6_N8_sector%d.txt" % (k_y * size_x + k_x), eigen_vectors[:, 0], fmt='%.10f')
    for energy in eigen_energy:
        energy_sector.append(float(energy))
        momentum_sector.append(k_y * size_x + k_x)
    print("%s is done!" % name)
    return momentum_sector, energy_sector


if __name__ == '__main__':
    band_structure(k_path, k_label)
    start_t = time.time()
    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    num_used_cores = 12
    print("当前正在使用" + str(num_used_cores) + "个核心")
    pool = mp.Pool(num_used_cores)
    param_list = []
    for j in range(N_y):
        for i in range(N_x):
            param_list.append(["task%d" % (j * N_x + i + 1), i, j])
    print(param_list)
    results = [pool.apply_async(exact_diagonal, args=(N_x, N_y, N, param[1], param[2], 8, param[0])) for param in param_list]
    pool.close()
    pool.join()
    results = [p.get() for p in results]
    print(results)
    end_t = time.time()
    elapsed_sec = (end_t - start_t)
    print("多进程计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")
    total_momentum = []
    energy_list = []
    for i in range(len(results)):
        total_momentum += results[i][0]
        energy_list += results[i][1]
    minimum = min(energy_list)
    print(minimum)
    for i in range(len(energy_list)):
        energy_list[i] = energy_list[i] - minimum

    # np.savetxt("eleventh_band_total_momentum_Nx4_Ny6_N8_dislocation.txt", total_momentum, fmt='%.8f')
    # np.savetxt("eleventh_band_energy_list_Nx4_Ny6_N8_dislocation.txt", energy_list, fmt='%.8f')

    # plt.subplot(1, 2, 1)
    plt.scatter(total_momentum, energy_list)
    plt.xlabel(r"$K_x + N_x K_y$")
    plt.ylabel(r"$E - E_{GS}$")
    x_major_locator = plt.MultipleLocator(5)
    x_minor_locator = plt.MultipleLocator(1)
    # y_major_locator = plt.MultipleLocator(0.02)
    # y_minor_locator = plt.MultipleLocator(0.004)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(x_minor_locator)
    ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_minor_locator(y_minor_locator)
    # ax.yaxis.set_major_locator(y_major_locator)
    # plt.ylim(0, 0.14)
    plt.show()
