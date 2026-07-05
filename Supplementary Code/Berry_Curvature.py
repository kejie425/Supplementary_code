import time
import numpy as np
import matplotlib.pyplot as plt
from pythtb import *  # import TB model class


# ------------------import wannier90 module-----------------
frozen = w90(r"./frozen_afm/frozen", r"wannier90")
my_model = frozen.model(min_hopping_norm=0.0001)
# my_model.display()
my_model_reduced = my_model.reduce_dim(2, 0)
# my_model_reduced.display()
my_model_reduced.ignore_position_operator_offdiagonal()
fermi_energy = -4.42

k_path = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.333, 0.333, 0.0], [0.0, 0.0, 0.0]]
k_label = (r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$')  # labels of the nodes


def band_structure(path, label):
    """
    solve model on a path and plot it.
    :param path: a list of k-points coordinates expressed by (m, n) with k = m * b1 + n * b2.
    :param label: a list of labels for the k-points.
    :return: band structure.
    """
    (k_vec, k_dist, k_node) = my_model.k_path(path, 202)  # call function k_path to construct the actual path
    print("k_vec.shape:", k_vec.shape)
    print("k_dist.shape:", k_dist.shape)
    print("k_node.shape:", k_node.shape)
    energy, vectors = my_model.solve_all(k_vec, eig_vectors=True)
    print("energy.shape:", energy.shape)
    print("vectors.shape:", vectors.shape)
    print("target:", energy[9, 0])
    for i in range(energy.shape[0]):
        for j in range(len(k_dist)):
            energy[i][j] = energy[i][j] - fermi_energy
    fig, ax = plt.subplots()
    for i in range(energy.shape[0]):
        ax.plot(k_dist, energy[i], "k-")
    for n in range(len(k_node)):
        ax.axvline(x=k_node[n], linewidth=0.5, color='k')
    ax.set_xlabel("Path in k-space")
    ax.set_ylabel("Band energy (eV)")
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_xticks(k_node)
    ax.set_xticklabels(label)
    fig.tight_layout()
    plt.show()


def eigen_vectors(steps):
    """
    This is a function to collect the eigen vectors of tight-binding model on a meshgrid.
    :param steps: the number of steps in the meshgrid.
    :return: a list of eigen vectors on the meshgrid.
    """
    grid1 = steps
    grid2 = steps
    ham_vec = []
    for ii in range(grid1 + 1):
        ham_vec.append([0] * (grid2 + 1))
    for ii in range(grid2 + 1):
        path = [[ii / grid1, 0, 0], [ii / grid1, 1, 0]]
        (k_vec, k_dist, k_node) = my_model.k_path(path, grid2 + 1)
        energy, vectors = my_model.solve_all(k_vec, eig_vectors=True)
        for jj in range(grid2 + 1):
            ham_vec[ii][jj] = vectors[:, jj, :].T
    return ham_vec


def berry_curvature(b1, b2, steps, target_band):
    """
    This is a function to calculate the Berry curvature and fluctuation of a tight-binding model on a meshgrid.
    :param b1: reciprocal lattice vector of the first Brillouin zone.
    :param b2: reciprocal lattice vector of the first Brillouin zone.
    :param steps: grid density for the calculation.
    :param target_band: list of target bands for the calculation, eg.[1, 2, 3].
    :return: the Chern number and fluctuation of the Berry curvature.
    """
    grid1 = steps
    grid2 = steps
    eigen_vectors_list = eigen_vectors(steps)
    flux = np.zeros((grid1, grid2))
    for ii in range(grid1):
        for jj in range(grid2):
            temp_loop = np.eye(len(target_band))
            temp_loop = np.dot(temp_loop, np.dot(eigen_vectors_list[ii][jj][:, target_band[0]:target_band[-1] + 1].T.conjugate(), eigen_vectors_list[ii + 1][jj][:, target_band[0]:target_band[-1] + 1]))
            temp_loop = np.dot(temp_loop, np.dot(eigen_vectors_list[ii + 1][jj][:, target_band[0]:target_band[-1] + 1].T.conjugate(), eigen_vectors_list[ii + 1][jj + 1][:, target_band[0]:target_band[-1] + 1]))
            temp_loop = np.dot(temp_loop, np.dot(eigen_vectors_list[ii + 1][jj + 1][:, target_band[0]:target_band[-1] + 1].T.conjugate(), eigen_vectors_list[ii][jj + 1][:, target_band[0]:target_band[-1] + 1]))
            temp_loop = np.dot(temp_loop, np.dot(eigen_vectors_list[ii][jj + 1][:, target_band[0]:target_band[-1] + 1].T.conjugate(), eigen_vectors_list[ii][jj][:, target_band[0]:target_band[-1] + 1]))
            flux[ii, jj] = np.sum(np.angle(np.linalg.eig(temp_loop)[0]))
    chern_num = np.sum(flux) / (2 * np.pi)
    Brillouin_area = abs(np.cross(b1, b2)[2])
    seperate_area = Brillouin_area / (grid1 * grid2)
    fluctuation_berry = (flux / seperate_area - 2 * np.pi * chern_num / Brillouin_area) ** 2
    fluctuation_berry = np.sqrt(np.sum(fluctuation_berry) * seperate_area * Brillouin_area) / (2 * np.pi)
    X, Y = np.meshgrid(np.arange(0, grid1, 1), np.arange(0, grid2, 1))
    kx = X * b1[0] / grid1 + Y * b2[0] / grid2
    ky = X * b1[1] / grid1 + Y * b2[1] / grid2
    fig = plt.figure(figsize=(12, 6), facecolor='w')
    ax = fig.add_subplot(121)
    aa = ax.pcolormesh(kx, ky, flux / 2 / np.pi, cmap='rainbow')
    # np.savetxt("kx_berry.txt", kx, fmt='%.6f')
    # np.savetxt("ky_berry.txt", ky, fmt='%.6f')
    # np.savetxt("Berry.txt", flux / 2 / np.pi, fmt='%.6f')
    # flux_min, flux_max = flux.min(), flux.max()
    # print("flux_min:", flux_min)
    # print("flux_max:", flux_max)
    bar = fig.colorbar(aa)
    # bar.mappable.set_clim(0, 50)
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_title(r"$Distribution\quad of\quad flux$")
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(kx, ky, flux, cmap='rainbow')
    ax2.set_xlabel('kx')
    ax2.set_ylabel('ky')
    ax2.set_zlabel('flux')
    ax2.set_title(r"$Distribution\quad of\quad flux$")
    plt.show()
    return chern_num, fluctuation_berry


if __name__ == '__main__':
    band_structure(k_path, k_label)
    a, b = berry_curvature(np.array([0.0855, 0.04936, 0]), np.array([0, 0.09872, 0]), 50, [11])
    print("chern number:", a)
    print("fluctuation_berry:", b)
