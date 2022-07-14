import os

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def system_ODE(C, t, k=None, F=None, C_in_data=None):

    step = int(t)

    try:
       C_in = C_in_data[step]
    except:
        C_in = C_in_data[-1]

    dydt = [-k[0] * C[0] + F * (C_in[0] - C[0]),
            k[0] * C[0] - k[1] * C[1] + F * (C_in[1] - C[1]),
            k[1] * C[1] + F * (C_in[2] - C[2])]

    return dydt


C_0 = [0, 0, 0]

size = 50
period = 50

C_in = np.zeros([size*period, 3])

for i in range(size):
    magn = np.random.rand(1)
    for C in C_in[i*period:(i+1)*period]:
        C[0] = magn

    # C_in[100*i:100*(i+1)][0] = np.random.rand(1)*np.ones((100,1))

k = [0.1, 0.02]
F = 0.02
t = np.linspace(0, size * period - 1, size * period)

para = (k, F, C_in)

sol = odeint(system_ODE, C_0, t, args=para)

plt.plot(t, sol[:, 0], 'b', label='A(t)')
plt.plot(t, sol[:, 1], 'g', label='B(t)')
plt.plot(t, sol[:, 2], 'r', label='C(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(dir, 'dataset/test/')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(os.path.join(save_dir,'Reactor_ODE_1_data_t.npy'), 'wb') as f:
    np.save(f, t.astype('int'))

with open(os.path.join(save_dir,'Reactor_ODE_1_data_output.npy'), 'wb') as f:
    np.save(f, sol)

with open(os.path.join(save_dir,'Reactor_ODE_1_data_input.npy'), 'wb') as f:
    np.save(f, C_in)

