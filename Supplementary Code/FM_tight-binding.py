import numpy as np
import matplotlib.pyplot as plt


# -------------------parameters-------------------
a = 1  # lattice constant
b = 4 * np.pi / (np.sqrt(3) * a)                              # reciprocal lattice constant
V_dd_sigma_NN = -0.02329326                                # nearest neighbor d-d hopping by sigma bond
V_dd_pi_NN = 0.07843804                                   # nearest neighbor d-d hopping by pi bond
V_dd_delta_NN = -0.06380194                               # nearest neighbor d-d hopping by delta bond
V_dd_sigma_NNN = 0.00267459                              # next nearest neighbor d-d hopping by sigma bond
V_dd_pi_NNN = 0.00511626                                 # next nearest neighbor d-d hopping by pi bond
V_dd_delta_NNN = -0.0126933                             # next nearest neighbor d-d hopping by delta bond
U_dz2 = 1.70774821                                        # onsite energy of dz2 orbital
U_dx2y2 = 1.61924221                                                # onsite energy of dx2y2 orbital
U_dxy = 1.61924221                                                 # onsite energy of dxy orbital
lambda1 = 0.18                                   # onsite SOC difference between Y22 and Y2-2
lambda2 = -0.01                                   # nearest hopping SOC strength between the same spherical orbital
lambda3 = -0.01                                   # nearest hopping SOC strength between Y20 and Y22, Y2-2
e1 = a * np.array([1, 0])  # first hopping vector
e2 = a * np.array([1/2, np.sqrt(3)/2])  # second hopping vector
e3 = a * np.array([1/2, -np.sqrt(3)/2])  # third hopping vector


# ---------------hopping parameters---------------
def direction_cosine(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return x / r, y / r, z / r


def dz2_dz2(l, m, n, sigma_bond, pi_bond, delta_bond):
    hop = (n ** 2 - 0.5 * (l ** 2 + m ** 2)) ** 2 * sigma_bond + 3 * n ** 2 * (l ** 2 + m ** 2) * pi_bond + 0.75 * (l ** 2 + m ** 2) ** 2 * delta_bond
    return hop


def dx2y2_dz2(l, m, n, sigma_bond, pi_bond, delta_bond):
    hop = np.sqrt(3) / 2 * (l ** 2 - m ** 2) * (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * sigma_bond + np.sqrt(3) * n ** 2 * (m ** 2 - l ** 2) * pi_bond + np.sqrt(3) / 4 * (1 + n ** 2) * (l ** 2 - m ** 2) * delta_bond
    return hop


def dxy_dz2(l, m, n, sigma_bond, pi_bond, delta_bond):
    hop = np.sqrt(3) * l * m * (n ** 2 - 0.5 * (l ** 2 + m ** 2)) * sigma_bond - 2 * np.sqrt(3) * l * m * n ** 2 * pi_bond + np.sqrt(3) / 2 * l * m * (1 + n ** 2) * delta_bond
    return hop


def dxy_dx2y2(l, m, n, sigma_bond, pi_bond, delta_bond):
    hop = 1.5 * l * m * (l ** 2 - m ** 2) * sigma_bond + 2 * l * m * (m ** 2 - l ** 2) * pi_bond + 0.5 * l * m * (l ** 2 - m ** 2) * delta_bond
    return hop


def dxy_dxy(l, m, n, sigma_bond, pi_bond, delta_bond):
    hop = 3 * l ** 2 * m ** 2 * sigma_bond + (l ** 2 + m ** 2 - 4 * l ** 2 * m ** 2) * pi_bond + (n ** 2 + l ** 2 * m ** 2) * delta_bond
    return hop


def dx2y2_dx2y2(l, m, n, sigma_bond, pi_bond, delta_bond):
    hop = 0.75 * (l ** 2 - m ** 2) ** 2 * sigma_bond + (l ** 2 + m ** 2 - (l ** 2 - m ** 2) ** 2) * pi_bond + (n ** 2 + 0.25 * (l ** 2 - m ** 2) ** 2) * delta_bond
    return hop


# -----------------hamiltonian matrix------------------
def tight_binding_element(orbital, direction_array, momentum_array, sigma, pi, delta, index):
    direction_cosine_array = direction_cosine(direction_array[0], direction_array[1], 0)
    if index:
        element = orbital(direction_cosine_array[0], direction_cosine_array[1], direction_cosine_array[2], sigma, pi, delta) * np.cos(np.dot(momentum_array, direction_array))
    else:
        element = orbital(direction_cosine_array[0], direction_cosine_array[1], direction_cosine_array[2], sigma, pi, delta) * np.exp(1j * np.dot(momentum_array, direction_array))
    return element


def nearest_soc_same_orbital(direction_array, momentum_array, index):
    if index:
        element = lambda2 * 2 * np.cos(np.dot(momentum_array, direction_array))
    else:
        element = -lambda2 * 2 * np.cos(np.dot(momentum_array, direction_array))
    return element


def nearest_soc_diff_orbital(direction_array, momentum_array, num, index):
    if index:
        phi = 2 * np.pi / 3
    else:
        phi = -2 * np.pi / 3
    element = lambda3 * np.exp(1j * (np.dot(momentum_array, direction_array) + num * phi))
    return element


def hamiltonian(kx, ky, sigma_NN, pi_NN, delta_NN, sigma_NNN, pi_NNN, delta_NNN):
    k = np.array([kx, ky])
    H = np.zeros((3, 3), dtype=complex)
    H[0, 0] += (tight_binding_element(dz2_dz2, e1, k, sigma_NN, pi_NN, delta_NN, True) + tight_binding_element(dz2_dz2, e2, k, sigma_NN, pi_NN, delta_NN, True) + tight_binding_element(dz2_dz2, e3, k, sigma_NN, pi_NN, delta_NN, True)
                + tight_binding_element(dz2_dz2, e1 + e2, k, sigma_NNN, pi_NNN, delta_NNN, True) + tight_binding_element(dz2_dz2, e2 - e3, k, sigma_NNN, pi_NNN, delta_NNN, True) + tight_binding_element(dz2_dz2, e1 + e3, k, sigma_NNN, pi_NNN, delta_NNN, True))
    H[0, 1] += (tight_binding_element(dx2y2_dz2, e1, k, sigma_NN, pi_NN, delta_NN, False) + tight_binding_element(dx2y2_dz2, e2, k, sigma_NN, pi_NN, delta_NN, False) + tight_binding_element(dx2y2_dz2, e3, k, sigma_NN, pi_NN, delta_NN, False)
                + tight_binding_element(dx2y2_dz2, -e1, k, sigma_NN, pi_NN, delta_NN, False) + tight_binding_element(dx2y2_dz2, -e2, k, sigma_NN, pi_NN, delta_NN, False) + tight_binding_element(dx2y2_dz2, -e3, k, sigma_NN, pi_NN, delta_NN, False)
                + tight_binding_element(dx2y2_dz2, e1 + e2, k, sigma_NNN, pi_NNN, delta_NNN, False) + tight_binding_element(dx2y2_dz2, e2 - e3, k, sigma_NNN, pi_NNN, delta_NNN, False) + tight_binding_element(dx2y2_dz2, -e1 - e3, k, sigma_NNN, pi_NNN, delta_NNN, False)
                + tight_binding_element(dx2y2_dz2, -e1 - e2, k, sigma_NNN, pi_NNN, delta_NNN, False) + tight_binding_element(dx2y2_dz2, -e2 + e3, k, sigma_NNN, pi_NNN, delta_NNN, False) + tight_binding_element(dx2y2_dz2, e1 + e3, k, sigma_NNN, pi_NNN, delta_NNN, False))
    H[0, 2] += (tight_binding_element(dxy_dz2, e1, k, sigma_NN, pi_NN, delta_NN, False) + tight_binding_element(dxy_dz2, e2, k, sigma_NN, pi_NN, delta_NN, False) + tight_binding_element(dxy_dz2, e3, k, sigma_NN, pi_NN, delta_NN, False)
                + tight_binding_element(dxy_dz2, -e1, k, sigma_NN, pi_NN, delta_NN, False) + tight_binding_element(dxy_dz2, -e2, k, sigma_NN, pi_NN, delta_NN, False) + tight_binding_element(dxy_dz2, -e3, k, sigma_NN, pi_NN, delta_NN, False)
                + tight_binding_element(dxy_dz2, e1 + e2, k, sigma_NNN, pi_NNN, delta_NNN, False) + tight_binding_element(dxy_dz2, e2 - e3, k, sigma_NNN, pi_NNN, delta_NNN, False) + tight_binding_element(dxy_dz2, -e1 - e3, k, sigma_NNN, pi_NNN, delta_NNN, False)
                + tight_binding_element(dxy_dz2, -e1 - e2, k, sigma_NNN, pi_NNN, delta_NNN, False) + tight_binding_element(dxy_dz2, -e2 + e3, k, sigma_NNN, pi_NNN, delta_NNN, False) + tight_binding_element(dxy_dz2, e1 + e3, k, sigma_NNN, pi_NNN, delta_NNN, False))
    H[1, 1] += (tight_binding_element(dx2y2_dx2y2, e1, k, sigma_NN, pi_NN, delta_NN, True) + tight_binding_element(dx2y2_dx2y2, e2, k, sigma_NN, pi_NN, delta_NN, True) + tight_binding_element(dx2y2_dx2y2, e3, k, sigma_NN, pi_NN, delta_NN, True)
                + tight_binding_element(dx2y2_dx2y2, e1 + e2, k, sigma_NNN, pi_NNN, delta_NNN, True) + tight_binding_element(dx2y2_dx2y2, e2 - e3, k, sigma_NNN, pi_NNN, delta_NNN, True) + tight_binding_element(dx2y2_dx2y2, e1 + e3, k, sigma_NNN, pi_NNN, delta_NNN, True))
    H[1, 2] += (tight_binding_element(dxy_dx2y2, e1, k, sigma_NN, pi_NN, delta_NN, False) + tight_binding_element(dxy_dx2y2, e2, k, sigma_NN, pi_NN, delta_NN, False) + tight_binding_element(dxy_dx2y2, e3, k, sigma_NN, pi_NN, delta_NN, False)
                + tight_binding_element(dxy_dx2y2, -e1, k, sigma_NN, pi_NN, delta_NN, False) + tight_binding_element(dxy_dx2y2, -e2, k, sigma_NN, pi_NN, delta_NN, False) + tight_binding_element(dxy_dx2y2, -e3, k, sigma_NN, pi_NN, delta_NN, False)
                + tight_binding_element(dxy_dx2y2, e1 + e2, k, sigma_NNN, pi_NNN, delta_NNN, False) + tight_binding_element(dxy_dx2y2, e2 - e3, k, sigma_NNN, pi_NNN, delta_NNN, False) + tight_binding_element(dxy_dx2y2, -e1 - e3, k, sigma_NNN, pi_NNN, delta_NNN, False)
                + tight_binding_element(dxy_dx2y2, -e1 - e2, k, sigma_NNN, pi_NNN, delta_NNN, False) + tight_binding_element(dxy_dx2y2, -e2 + e3, k, sigma_NNN, pi_NNN, delta_NNN, False) + tight_binding_element(dxy_dx2y2, e1 + e3, k, sigma_NNN, pi_NNN, delta_NNN, False))
    H[2, 2] += (tight_binding_element(dxy_dxy, e1, k, sigma_NN, pi_NN, delta_NN, True) + tight_binding_element(dxy_dxy, e2, k, sigma_NN, pi_NN, delta_NN, True) + tight_binding_element(dxy_dxy, e3, k, sigma_NN, pi_NN, delta_NN, True)
                + tight_binding_element(dxy_dxy, e1 + e2, k, sigma_NNN, pi_NNN, delta_NNN, True) + tight_binding_element(dxy_dxy, e2 - e3, k, sigma_NNN, pi_NNN, delta_NNN, True) + tight_binding_element(dxy_dxy, e1 + e3, k, sigma_NNN, pi_NNN, delta_NNN, True))
    H = H + np.conjugate(H).T
    H[0, 0] += U_dz2
    H[1, 1] += U_dx2y2
    H[2, 2] += U_dxy
    #  This is a unitary transformation to change basis from (dz2, dx2y2, dxy) to spherical harmonics (Y20, Y22, Y2-2)
    unitary_transformation = np.array([[1, 0, 0], [0, 1/np.sqrt(2), 1/np.sqrt(2)], [0, 1j/np.sqrt(2), -1j/np.sqrt(2)]])
    H = np.dot(np.conjugate(unitary_transformation).T, np.dot(H, unitary_transformation))
    H[1, 1] += lambda1
    H[2, 2] += -lambda1
    H[1, 1] += nearest_soc_same_orbital(e1, k, True) + nearest_soc_same_orbital(e2, k, True) + nearest_soc_same_orbital(e3, k, True)
    H[2, 2] += nearest_soc_same_orbital(e1, k, False) + nearest_soc_same_orbital(e2, k, False) + nearest_soc_same_orbital(e3, k, False)
    s_dx2y2 = nearest_soc_diff_orbital(e1, k, 0, True) + nearest_soc_diff_orbital(e2, k, 1, True) + nearest_soc_diff_orbital(-e3, k, 2, True) + nearest_soc_diff_orbital(-e1, k, 3, True) + nearest_soc_diff_orbital(-e2, k, 4, True) + nearest_soc_diff_orbital(e3, k, 5, True)
    H[0, 1] += s_dx2y2
    H[1, 0] += np.conj(s_dx2y2)
    s_dxy = nearest_soc_diff_orbital(e1, k, 0, False) + nearest_soc_diff_orbital(e2, k, 1, False) + nearest_soc_diff_orbital(-e3, k, 2, False) + nearest_soc_diff_orbital(-e1, k, 3, False) + nearest_soc_diff_orbital(-e2, k, 4, False) + nearest_soc_diff_orbital(e3, k, 5, False)
    H[0, 2] += s_dxy
    H[2, 0] += np.conj(s_dxy)
    # H = np.dot(unitary_transformation, np.dot(H, np.conjugate(unitary_transformation).T))
    return H


#  ------------------band structure------------------
K = b * np.array([np.sqrt(3)/3, 0])
K_prime = b / np.sqrt(3) * np.array([1/2, np.sqrt(3)/2])
gamma = b * np.array([0, 0])
M = b * np.array([np.sqrt(3)/4, 1/4])
highpoint = np.concatenate((gamma, K, M, K_prime, gamma), axis=0)
highpoint = highpoint.reshape(5, 2)


def band_structure_2D():
    steps = [150, 75, 75, 150]
    energy_band = np.zeros((sum(steps), 3))
    for num in range(4):
        path = np.linspace(highpoint[num, :], highpoint[num + 1, :], steps[num])
        for momentum in range(steps[num]):
            k_x = path[momentum, 0]
            k_y = path[momentum, 1]
            energy, vector = np.linalg.eigh(hamiltonian(k_x, k_y, V_dd_sigma_NN, V_dd_pi_NN, V_dd_delta_NN, V_dd_sigma_NNN, V_dd_pi_NNN, V_dd_delta_NNN))
            energy_band[sum([steps[jj] for jj in range(num)]) + momentum, :] = energy
    x_axis = [i for i in range(sum(steps))]
    x_ticks1 = [r"$\Gamma$", "K", "M", "K'", r"$\Gamma$"]
    x_ticks2 = [0, steps[0], steps[0] + steps[1] - 1, steps[0] + steps[1] + steps[2] - 1, steps[0] + steps[1] + steps[2] + steps[3] - 1]
    np.savetxt("fm_momentum_soc10.txt", x_axis)
    np.savetxt("fm_energyband_soc10.txt", energy_band)
    plt.plot(x_axis, energy_band, c="black")
    plt.xticks(x_ticks2, x_ticks1)
    plt.xlim(x_axis[0], x_axis[-1])
    plt.ylim(1.2, 2.0)
    plt.ylabel(r"$E\ (meV)$")
    plt.title(r"FM states")
    # print(dz2_orbital)
    # plt.plot(x_axis, dz2_orbital, label=r"$\Delta_{z^2}$")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    band_structure_2D()