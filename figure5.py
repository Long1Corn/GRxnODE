import matplotlib.pyplot as plt
import numpy as np
import torch
# import NeuralODE.Engine.Trainer as Trainer
from NeuralODE.Engine.cfg import CFG
import numpy
import math


def RTD_f(model_path, G=True):
    cfg = CFG()
    cfg.RTD_G_CSTR = G

    cfg.model_in_var = 5
    cfg.model_hidden_var = 40
    cfg.model_out_var = 5

    trainer = Trainer.RTD(cfg, [], [])
    trainer.model.load_state_dict(torch.load(model_path))
    RTD_0 = trainer.model.get_RTD(torch.tensor([0.05]).to(cfg.device))
    RTD_1 = RTD_0.cpu().detach().numpy()

    # plt.figure()
    # plt.plot(numpy.linspace(cfg.RTD_max_tau / cfg.RTD_N,
    #                         cfg.RTD_max_tau,
    #                         cfg.RTD_N),
    #          numpy.flip(RTD_1) * cfg.RTD_N / cfg.RTD_max_tau)
    # plt.ylim(0, 1)
    # plt.xlim(0, cfg.RTD_max_tau)
    # plt.title("RTD")

    return RTD_1


def Et(t, n, tau=1):
    if n >= 100:
        Et = numpy.zeros_like(t)
        Et[t == 1] = 3
    else:
        Et = t ** (n - 1) / math.gamma(n) * ((n / tau) ** n) * numpy.exp(-t * n / tau)

    return Et


# from TheoryGuidedRNN.dataset import Reactor_real
# path1 = r"/home/pengfei/projects/neural_ode/NeuralODE/output/RTD_r_0.25.pth"
# path2 = r"/home/pengfei/projects/neural_ode/NeuralODE/output/G_r_0.25.pth"
#
# fig = plt.figure(figsize=(5, 5))
# ax1 = fig.add_subplot()
# t = numpy.linspace(4.0 / 100, 4.0, 100)
# RTD = RTD_f(path1, G=False)
# RTD_1 = numpy.flip(RTD) * 100 / 4
# ax1.plot(t, RTD_1, linewidth=3, label="RTD model", color="orange")
#
# RTD2 = RTD_f(path2, G=True)
# RTD_2 = numpy.flip(RTD2) * 100 / 4
# ax1.plot(t, RTD_2, linewidth=3, label="G-CSTR model", color="deepskyblue")
#
# RTD_3 = Et(t,1,tau=1)
# ax1.plot(t, RTD_3, linewidth=3, label="N-CSTR model", color="limegreen")
#
# Et_true = Reactor_real.sol[:, 5]
# print(sum(Et_true) * 0.04)
#
# ax1.plot(t, Et_true, linewidth=3, color="black", linestyle="--", label="True RTD")
#
# ax1.set_ylim(0, 1.0)
# ax1.set_xlim(0, 4)
# ax1.xaxis.set_ticks([0, 1, 2, 3, 4])
# ax1.yaxis.set_ticks(np.linspace(0, 1.0, 6))
# ax1.legend(loc="upper right")
#
# plt.show(dpi=600)

# ----------------------------------------------------------------------------------#
data_1 = [0.0472, 0.0521, 0.0519]  # G-CSTR
data_2 = [0.0332, 0.0414, 0.0535]  # RTD
data_3 = [0.0499, 0.0568, 0.0641]  # n-cstr
data_4 = [0.0886, 0.0919, 0.0711]  # bbox
data_5 = [0.0712, 0.1175, 0.2281]  # lstm
data_6 = [0.3793, 0.3860, 0.4286]  # DRSM

label = ["4", "2", "1"]

fig = plt.figure(figsize=(5, 10.5))
ax1 = fig.add_subplot(711)
bar1 = ax1.bar(label, data_1, color="black")
for bar in bar1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + .2, yval + .003, "{:.3f}".format(yval))
# ax1.set_ylim(0, 0.1)
ax1.yaxis.set_ticks(np.linspace(0, 0.1, 3))

ax2 = fig.add_subplot(712)
bar2 = ax2.bar(label, data_2, color="black")
for bar in bar2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + .25, yval + .003, "{:.3f}".format(yval))
# ax1.set_ylim(0, 0.1)
ax2.yaxis.set_ticks(np.linspace(0, 0.1, 3))

ax3 = fig.add_subplot(713)
bar3 = ax3.bar(label, data_3, color="black")
for bar in bar3:
    yval = bar.get_height()
    ax3.text(bar.get_x() + .25, yval + .003, "{:.3f}".format(yval))
# ax1.set_ylim(0, 0.1)
ax3.yaxis.set_ticks(np.linspace(0, 0.1, 3))

ax4 = fig.add_subplot(715)
bar4 = ax4.bar(label, data_4, color="black")
for bar in bar4:
    yval = bar.get_height()
    ax4.text(bar.get_x() + .25, yval + .003, "{:.3f}".format(yval))
# ax1.set_ylim(0, 0.1)
ax4.yaxis.set_ticks(np.linspace(0, 0.4, 3))

ax5 = fig.add_subplot(716)
bar5 = ax5.bar(label, data_5, color="black")
for bar in bar5:
    yval = bar.get_height()
    ax5.text(bar.get_x() + .25, yval + .003, "{:.3f}".format(yval))
# ax1.set_ylim(0, 0.1)
ax5.yaxis.set_ticks(np.linspace(0, 0.4, 3))

ax6 = fig.add_subplot(717)
bar6 = ax6.bar(label, data_6, color="black")
for bar in bar6:
    yval = bar.get_height()
    ax6.text(bar.get_x() + .25, yval + .003, "{:.3f}".format(yval))
# ax1.set_ylim(0, 0.1)
ax6.yaxis.set_ticks(np.linspace(0, 0.4, 3))

plt.show(dpi=1200)
