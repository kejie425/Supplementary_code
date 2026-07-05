import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# -----------------------parameters---------------------
J_prime = 5.29 / 6.582 / 1.5 * 10 ** 3                # exchange coupling J/hbar in unit of ps^(-1)
D_prime = 0.06 / 6.582 / 1.5 * 10 ** 3               # Dzyaloshinskii-Moriya coupling D/hbar in unit of ps^(-1)
K_prime = 0.11 / 6.582 / 1.5 * 10 ** 3              # uniaxial anisotropy K/hbar in unit of ps^(-1)
Hx = -1.758
Hy = 0
Hz = 0
theta_prime = 0.0013118 * 1.483
alpha = 0.003         # Gilbert damping


# ----------------------equations-----------------------
def LLG_equation(variables, time_range):
    m1x, m1y, m1z, m2x, m2y, m2z, m3x, m3y, m3z = variables
    dHdm1x = 3 * J_prime * (m2x + m3x) + 3 * D_prime * (m3z - m2z) - Hx
    dHdm1y = 3 * J_prime * (m2y + m3y) - Hy
    dHdm1z = 3 * J_prime * (m2z + m3z) + 3 * D_prime * (m2x - m3x) - 2 * K_prime * m1z - Hz
    dHdm2x = 3 * J_prime * (m1x + m3x) + 3 * D_prime * (m1z - m3z) - K_prime * (3/2 * m2x - np.sqrt(3)/2 * m2z) - Hx
    dHdm2y = 3 * J_prime * (m1y + m3y) - Hy
    dHdm2z = 3 * J_prime * (m1z + m3z) + 3 * D_prime * (m3x - m1x) - K_prime * (-np.sqrt(3)/2 * m2x + 1/2 * m2z) - Hz
    dHdm3x = 3 * J_prime * (m1x + m2x) + 3 * D_prime * (m2z - m1z) - K_prime * (3/2 * m3x + np.sqrt(3)/2 * m3z) - Hx
    dHdm3y = 3 * J_prime * (m1y + m2y) - Hy
    dHdm3z = 3 * J_prime * (m1z + m2z) + 3 * D_prime * (m1x - m2x) - K_prime * (np.sqrt(3)/2 * m3x + 1/2 * m3z) - Hz
    dm1xdt = 1/(1 + alpha ** 2) * ((m1y * dHdm1z - m1z * dHdm1y) + alpha * (m1x * dHdm1x + m1y * dHdm1y + m1z * dHdm1z) * m1x - alpha * dHdm1x - theta_prime * m1y * m1x - alpha * theta_prime * m1z)
    dm1ydt = 1/(1 + alpha ** 2) * ((m1z * dHdm1x - m1x * dHdm1z) + alpha * (m1x * dHdm1x + m1y * dHdm1y + m1z * dHdm1z) * m1y - alpha * dHdm1y - theta_prime * m1y * m1y + theta_prime)
    dm1zdt = 1/(1 + alpha ** 2) * ((m1x * dHdm1y - m1y * dHdm1x) + alpha * (m1x * dHdm1x + m1y * dHdm1y + m1z * dHdm1z) * m1z - alpha * dHdm1z - theta_prime * m1y * m1z + alpha * theta_prime * m1x)

    dm2xdt = 1/(1 + alpha ** 2) * ((m2y * dHdm2z - m2z * dHdm2y) + alpha * (m2x * dHdm2x + m2y * dHdm2y + m2z * dHdm2z) * m2x - alpha * dHdm2x - theta_prime * m2y * m2x - alpha * theta_prime * m2z)
    dm2ydt = 1/(1 + alpha ** 2) * ((m2z * dHdm2x - m2x * dHdm2z) + alpha * (m2x * dHdm2x + m2y * dHdm2y + m2z * dHdm2z) * m2y - alpha * dHdm2y - theta_prime * m2y * m2y + theta_prime)
    dm2zdt = 1/(1 + alpha ** 2) * ((m2x * dHdm2y - m2y * dHdm2x) + alpha * (m2x * dHdm2x + m2y * dHdm2y + m2z * dHdm2z) * m2z - alpha * dHdm2z - theta_prime * m2y * m2z + alpha * theta_prime * m2x)

    dm3xdt = 1/(1 + alpha ** 2) * ((m3y * dHdm3z - m3z * dHdm3y) + alpha * (m3x * dHdm3x + m3y * dHdm3y + m3z * dHdm3z) * m3x - alpha * dHdm3x - theta_prime * m3y * m3x - alpha * theta_prime * m3z)
    dm3ydt = 1/(1 + alpha ** 2) * ((m3z * dHdm3x - m3x * dHdm3z) + alpha * (m3x * dHdm3x + m3y * dHdm3y + m3z * dHdm3z) * m3y - alpha * dHdm3y - theta_prime * m3y * m3y + theta_prime)
    dm3zdt = 1/(1 + alpha ** 2) * ((m3x * dHdm3y - m3y * dHdm3x) + alpha * (m3x * dHdm3x + m3y * dHdm3y + m3z * dHdm3z) * m3z - alpha * dHdm3z - theta_prime * m3y * m3z + alpha * theta_prime * m3x)

    dmdt = [dm1xdt, dm1ydt, dm1zdt, dm2xdt, dm2ydt, dm2zdt, dm3xdt, dm3ydt, dm3zdt]
    return dmdt


def rotation(phi):
    mat = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    return mat


# ----------------------simulation-----------------------
initial_arrow1 = np.array([0, 1])
initial_arrow2 = np.array([-np.sqrt(3)/2, -1/2])
initial_arrow3 = np.array([np.sqrt(3)/2, -1/2])
rotated_arrow1 = np.dot(rotation(0), initial_arrow1)
rotated_arrow2 = np.dot(rotation(0), initial_arrow2)
rotated_arrow3 = np.dot(rotation(0), initial_arrow3)
# rotated_arrow1 = np.dot(rotation(-np.pi/4), initial_arrow1)
# rotated_arrow2 = np.dot(rotation(-np.pi/4), initial_arrow2)
# rotated_arrow3 = np.dot(rotation(-np.pi/4), initial_arrow3)
initial_guess = np.array([rotated_arrow1[0], rotated_arrow1[1], rotated_arrow2[0], rotated_arrow2[1], rotated_arrow3[0], rotated_arrow3[1]])
print("initial_guess:", initial_guess)
m_initial = [rotated_arrow1[0], 0, rotated_arrow1[1], rotated_arrow2[0], 0, rotated_arrow2[1], rotated_arrow3[0], 0, rotated_arrow3[1]]
t = np.linspace(0, 30, 3000001)

solution = sp.integrate.odeint(LLG_equation, m_initial, t)
# np.save("solution.npy", solution)

arrow1 = np.array([solution[:, 0][-1], solution[:, 1][-1], solution[:, 2][-1]])
arrow2 = np.array([solution[:, 3][-1], solution[:, 4][-1], solution[:, 5][-1]])
arrow3 = np.array([solution[:, 6][-1], solution[:, 7][-1], solution[:, 8][-1]])
angle12 = np.arccos(np.dot(arrow1, arrow2) / (np.linalg.norm(arrow1) * np.linalg.norm(arrow2))) / np.pi * 180
angle13 = np.arccos(np.dot(arrow1, arrow3) / (np.linalg.norm(arrow1) * np.linalg.norm(arrow3))) / np.pi * 180
angle23 = np.arccos(np.dot(arrow2, arrow3) / (np.linalg.norm(arrow2) * np.linalg.norm(arrow3))) / np.pi * 180

print(arrow1, arrow2, arrow3)
print(arrow1 + arrow2 + arrow3)
print(angle12, angle13, angle23)
print(np.arctan(arrow1[2] / arrow1[0]) / np.pi * 180)

fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(121)
ax.plot(t, solution[:, 0], 'r', linestyle='-', label=r"$m_{1x}$")
ax.plot(t, solution[:, 1], 'r', linestyle='--', label=r"$m_{1y}$")
ax.plot(t, solution[:, 2], 'r', linestyle=':', label=r"$m_{1z}$")
ax.plot(t, solution[:, 3], 'b', linestyle='-', label=r"$m_{2x}$")
ax.plot(t, solution[:, 4], 'b', linestyle='--', label=r"$m_{2y}$")
ax.plot(t, solution[:, 5], 'b', linestyle=':', label=r"$m_{2z}$")
ax.plot(t, solution[:, 6], 'g', linestyle='-', label=r"$m_{3x}$")
ax.plot(t, solution[:, 7], 'g', linestyle='--', label=r"$m_{3y}$")
ax.plot(t, solution[:, 8], 'g', linestyle=':', label=r"$m_{3z}$")
ax.legend(loc='best')
ax.set_xlabel('t / (0.1 ns)')
ax.grid()
ax = fig.add_subplot(122)
ax.arrow(0, 0, float(arrow1[0]), float(arrow1[2]), head_width=0.05, head_length=0.1, fc='r', ec='r', label=r"$m_{1}$")
ax.arrow(0, 0, float(arrow2[0]), float(arrow2[2]), head_width=0.05, head_length=0.1, fc='b', ec='b', label=r"$m_{2}$")
ax.arrow(0, 0, float(arrow3[0]), float(arrow3[2]), head_width=0.05, head_length=0.1, fc='g', ec='g', label=r"$m_{3}$")
plt.legend()
plt.show()

