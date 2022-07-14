import matplotlib.pyplot as plt
import numpy as np

data_1 = [0.0077, 0.0063, 0.0094, 0.0090]
data_2 = [0.0082, 0.0106, 0.0121, 0.0098]
data_100 = [0.0350, 0.0431, 0.0334, 0.0376]
label = ["10", "4", "2", "1"]

fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_subplot(311)
bar1 = ax1.bar(label, data_1, color="black")
for bar in bar1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + .2, yval + .005, "{:.3f}".format(yval))
# ax1.set_ylim(0, 0.1)
ax1.yaxis.set_ticks(np.linspace(0, 0.1, 3))

ax2 = fig.add_subplot(312)
bar2 = ax2.bar(label, data_2, color="black")
for bar in bar2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + .2, yval + .005, "{:.3f}".format(yval))
ax2.yaxis.set_ticks(np.linspace(0, 0.1, 3))

ax3 = fig.add_subplot(313)
bar3 = ax3.bar(label, data_100, color="black")
for bar in bar3:
    yval = bar.get_height()
    ax3.text(bar.get_x() + .2, yval + .005, "{:.3f}".format(yval))
ax3.yaxis.set_ticks([0, 0.1 ,0.16])

plt.show(dpi=600)





data_B1 = [0.0167, 0.0157, 0.0187, 0.0244]
data_B2 = [0.0490, 0.0517, 0.0455, 0.0408]
data_Bi = [0.1213, 0.1034, 0.1190, 0.1131]
data_L1 = [0.0293, 0.0470, 0.0448, 0.0969]
data_L2 = [0.0464, 0.0292, 0.0421, 0.0807]
data_Li = [0.1280, 0.0940, 0.1420, 0.1600]

label = ["10", "4", "2", "1"]

fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_subplot(321)
bar1 = ax1.bar(label, data_B1, color="black")
for bar in bar1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + .05, yval + .005, "{:.3f}".format(yval))
# ax1.set_ylim(0, 0.1)
ax1.yaxis.set_ticks(np.linspace(0, 0.1, 3))

ax2 = fig.add_subplot(323)
bar2 = ax2.bar(label, data_B2, color="black")
for bar in bar2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + .05, yval + .005, "{:.3f}".format(yval))
ax2.yaxis.set_ticks(np.linspace(0, 0.1, 3))

ax3 = fig.add_subplot(325)
bar3 = ax3.bar(label, data_Bi, color="black")
for bar in bar3:
    yval = bar.get_height()
    ax3.text(bar.get_x() + .05, yval + .005, "{:.3f}".format(yval))
ax3.yaxis.set_ticks([0, 0.1, 0.16])

ax4 = fig.add_subplot(322)
bar4 = ax4.bar(label, data_L1, color="black")
for bar in bar4:
    yval = bar.get_height()
    ax4.text(bar.get_x() + .05, yval + .005, "{:.3f}".format(yval))
ax4.yaxis.set_ticks([])

ax5 = fig.add_subplot(324)
bar5 = ax5.bar(label, data_L2, color="black")
for bar in bar5:
    yval = bar.get_height()
    ax5.text(bar.get_x() + .05, yval + .005, "{:.3f}".format(yval))
ax5.yaxis.set_ticks([])

ax6 = fig.add_subplot(326)
bar6= ax6.bar(label, data_Li, color="black")
for bar in bar6:
    yval = bar.get_height()
    ax6.text(bar.get_x() + .05, yval + .005, "{:.3f}".format(yval))
ax6.yaxis.set_ticks([])

plt.show(dpi=600)
