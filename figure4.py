import matplotlib.pyplot as plt
import numpy as np
import torch
import NeuralODE.Engine.Trainer as Trainer
from NeuralODE.Engine.cfg import CFG
import numpy
import math


def RTD_f(model_path, G=True):
    cfg = CFG()
    cfg.RTD_G_CSTR = False

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


# RTD_t = np.load(r"/home/pengfei/projects/neural_ode/TheoryGuidedRNN/dataset/PFR_dif_RTD/RTD.npy")

# path1 = r"/home/pengfei/projects/neural_ode/NeuralODE/output/A_P2_S5_D20.pth"
# path2 = r"/home/pengfei/projects/neural_ode/NeuralODE/output/G_2_0.25.pth"
# path3 = r"/home/pengfei/projects/neural_ode/NeuralODE/output/G_100_0.25.pth"

# fig = plt.figure(figsize=(5.5, 5))
# # ax1 = fig.add_subplot(321)
# t = numpy.linspace(4.0 / 100, 4.0, 100)
# RTD = RTD_f(path1)
# RTD_1 = numpy.flip(RTD) * 100 / 4
# plt.plot(t, RTD_1, linewidth=5, label="Learned RTD")
# plt.plot(np.linspace(0, 4, 80), RTD_t / sum(RTD_t) * 20, linewidth=5, color="black", linestyle="--", label="True RTD")
# plt.ylim(0, 2.5)
# plt.xlim(0, 4)
# plt.xticks(np.linspace(0, 4, 5), fontsize=16)
# plt.yticks(np.linspace(0, 2.5, 6), fontsize=16)
# plt.legend(loc="upper right", fontsize=16)
# plt.show()


#
# ax2 = fig.add_subplot(323)
# t = numpy.linspace(4.0 / 100, 4.0, 100)
# RTD = RTD_f(path2)
# RTD_1 = numpy.flip(RTD) * 100 / 4
# ax2.plot(t, RTD_1, linewidth=3)
# ax2.plot(t, Et(t, 2), linewidth=3, color="black", linestyle="--")
# ax2.set_ylim(0, 1)
# ax2.set_xlim(0, 4)
# ax2.xaxis.set_ticks(np.linspace(0, 4, 5))
# ax2.yaxis.set_ticks(np.linspace(0, 1, 3))
#
# ax3 = fig.add_subplot(325)
# t = numpy.linspace(4.0 / 100, 4.0, 100)
# RTD = RTD_f(path3)
# RTD_1 = numpy.flip(RTD) * 100 / 4
# ax3.plot(t, RTD_1, linewidth=3)
# ax3.plot(t, Et(t, 100), linewidth=3, color="black", linestyle="--")
# ax3.set_ylim(0, 1)
# ax3.set_xlim(0, 4)
# ax3.xaxis.set_ticks(np.linspace(0, 4, 5))
# ax3.yaxis.set_ticks(np.linspace(0, 1, 3))
#
# plt.show(dpi=600)


# path1 = r"/home/pengfei/projects/neural_ode/NeuralODE/output/RTD_1_0.25.pth"
# path2 = r"/home/pengfei/projects/neural_ode/NeuralODE/output/RTD_2_0.25.pth"
# path3 = r"/home/pengfei/projects/neural_ode/NeuralODE/output/RTD_100_0.25.pth"
#
# fig = plt.figure(figsize=(5, 5))
# ax4 = fig.add_subplot(322)
# t = numpy.linspace(4.0 / 100, 4.0, 100)
# RTD = RTD_f(path1, G=False)
# RTD_1 = numpy.flip(RTD) * 100 / 4
# ax4.plot(t, RTD_1, linewidth=3, label="Pred RTD")
# ax4.plot(t, Et(t, 1), linewidth=3, color="black", linestyle="--", label="True RTD")
# ax4.set_ylim(0, 1)
# ax4.set_xlim(0, 4)
# ax4.xaxis.set_ticks(np.linspace(0, 4, 5))
# ax4.yaxis.set_ticks([])
# ax4.legend(loc="upper right")
#
# ax5 = fig.add_subplot(324)
# t = numpy.linspace(4.0 / 100, 4.0, 100)
# RTD = RTD_f(path2, G=False)
# RTD_1 = numpy.flip(RTD) * 100 / 4
# ax5.plot(t, RTD_1, linewidth=3)
# ax5.plot(t, Et(t, 2), linewidth=3, color="black", linestyle="--")
# ax5.set_ylim(0, 1)
# ax5.set_xlim(0, 4)
# ax5.xaxis.set_ticks(np.linspace(0, 4, 5))
# ax5.yaxis.set_ticks([])
#
# ax6 = fig.add_subplot(326)
# t = numpy.linspace(4.0 / 100, 4.0, 100)
# RTD = RTD_f(path3, G=False)
# RTD_1 = numpy.flip(RTD) * 100 / 4
# ax6.plot(t, RTD_1, linewidth=3)
# ax6.plot(t, Et(t, 100), linewidth=3, color="black", linestyle="--")
# ax6.set_ylim(0, 1)
# ax6.set_xlim(0, 4)
# ax6.xaxis.set_ticks(np.linspace(0, 4, 5))
# ax6.yaxis.set_ticks([])
#
# plt.show(dpi=600)

# ----------------------------------------------------------------------------

# data_1 = [0.0183, 0.0138, 0.0582, 0.0549]
# data_2 = [0.0124, 0.0203, 0.0549, 0.0176]
# label = ["1 CSTR", "2 CSTR", "PFR", "Non-ideal"]
#
# fig = plt.figure(figsize=(5, 5))
# ax1 = fig.add_subplot(212)
# bar1 = ax1.bar(label, data_1, color="black")
# for bar in bar1:
#     yval = bar.get_height()
#     ax1.text(bar.get_x() + .2, yval + .003, "{:.3f}".format(yval))
# # ax1.set_ylim(0, 0.1)
# ax1.yaxis.set_ticks(np.linspace(0, 0.1, 6))
#
#
# ax2 = fig.add_subplot(211)
# bar2 = ax2.bar(label, data_2, color="black")
# for bar in bar2:
#     yval = bar.get_height()
#     ax2.text(bar.get_x() + .25, yval + .003, "{:.3f}".format(yval))
# # ax1.set_ylim(0, 0.1)
# ax2.yaxis.set_ticks(np.linspace(0, 0.1, 6))
# plt.show(dpi=600)

# ------------------------------------------------------------------------------
#
# path1 = r"/home/pengfei/projects/neural_ode/NeuralODE/output/RTD_u_0.25_test.pth"
# path2 = r"/home/pengfei/projects/neural_ode/NeuralODE/output/G_u_0.25.pth"
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
# Et_true = numpy.zeros_like(t)
# Et_true[10:] = Et(t[:100 - 10], 1, tau=0.6)

# ax1.plot(t, Et_true, linewidth=3, color="black", linestyle="--", label="True RTD")

# ax1.set_ylim(0, 1.5)
# ax1.set_xlim(0, 4)
# ax1.xaxis.set_ticks([0, 0.5, 1, 2, 3, 4])
# ax1.yaxis.set_ticks(np.linspace(0, 1.5, 7))
# ax1.legend(loc="upper right")
#
# plt.show(dpi=600)
