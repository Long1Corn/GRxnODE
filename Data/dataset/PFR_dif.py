import os

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

segs = 500
D = 250


def system_ODE(C, t, k=None, F=None, C_in_data=None):
    step = int(t)

    try:
        C_in = C_in_data[step]
    except:
        C_in = C_in_data[-1]

    C = np.array(np.split(C, 3))

    r_1 = -k[0] * C[0, :] ** 2 + k[3] * C[1, :] ** 2
    r_2 = -k[1] * C[1, :] ** 2 + k[2] * C[2, :] ** 2

    r = np.stack((r_1, -r_1 + r_2, -r_2), axis=1).transpose()
    r = 0

    dC = get_d(C)
    d2C = get_d(dC)

    dC[:, 0] = - C_in + C[:, 0]
    dC[:, 1:] = C[:, 1:] - C[:, :-1]

    dydt = r - F * segs * dC + D * d2C
    # dydt[:, 0] = dydt[:, 0] + F * C_in

    dydt2 = dydt.flatten()

    if t > 1:
        a = 0

    return dydt2


def get_d(y, h=1):
    h = 1
    dy = np.zeros_like(y)
    dy[:, 0] = (-3 * y[:, 0] + 4 * y[:, 1] - y[:, 2]) / (2 * h)
    dy[:, 1:-1] = (y[:, 2:] - y[:, :-2]) / (2 * h)
    dy[:, -1] = (y[:, -3] - 4 * y[:, -2] + 3 * y[:, -1]) / (2 * h)

    return dy


C_0 = np.zeros(segs * 3)
C_0[0] = 1

size = 1
period = 80

C_in = np.zeros([size * period, 3])

# for i in range(size):
#     magn = np.random.rand(3)
#     for C in C_in[i * period:(i + 1) * period]:
#         C[:] = magn

# C_in[100*i:100*(i+1)][0] = np.random.rand(1)*np.ones((100,1))

k = [0.5, 0.3, 0.1, 0.006]
F = 0.05
t = np.linspace(0, size * period - 1, size * period)

para = (k, F, C_in)

sol = odeint(system_ODE, C_0, t, args=para, rtol=1e-12, atol=1e-10)

sol = sol.reshape((size * period, 3, segs))
sol = sol[:, :, -1]

s = 10

# plt.plot(t[::s], sol[::s, 0]/sum(sol[::s, 0]), 'bo', label='A(t)')
# plt.plot(t[::s], sol[::s, 1], 'go', label='B(t)')
# plt.plot(t[::s], sol[::s, 2], 'ro', label='C(t)')
plt.plot(t, sol[:, 0] / sum(sol[:, 0]) * 20, 'b')
plt.plot(t, sol[:, 1], 'g')
plt.plot(t, sol[:, 2], 'r')
plt.legend(loc='best')
plt.xlabel('t')
# plt.ylim(0,2)
plt.grid()
plt.show()

dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(dir, 'PFR_dif_RTD/')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
#
with open(os.path.join(save_dir, 'RTD.npy'), 'wb') as f:
    np.save(f, sol[:, 0])
#
# with open(os.path.join(save_dir, 'Reactor_ODE_1_data_output.npy'), 'wb') as f:
#     np.save(f, sol)
#
# with open(os.path.join(save_dir, 'Reactor_ODE_1_data_input.npy'), 'wb') as f:
#     np.save(f, C_in)
