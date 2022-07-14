import numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from Data.PFR_dif import get_data

trainset_1, testset_1, data_scale = get_data(s=1, data="CSTR_mix")

t = trainset_1["time_data"]
y = trainset_1["y_data"].squeeze()
y_in = trainset_1["p_data"][0][:, 1:]

# t = testset_1["time_data"]
# y = testset_1["y_data"].squeeze()
# y_in = testset_1["p_data"][0][:, 1:]

y_sample = y[::1, :]
t_sample = t[::1]
y_in = y_in[::1]

# y = numpy.concatenate((y[:, 0:1], y[:, 1:2] + y[:, 2:3], y[:, 3:4]), axis=1)

colors = ["r", "g", "b"]
labels = ["A", "B", "C"]

xlim = (0,1200)
ylim= (0,1.8)

###########################################################################3
# plt.subplots(figsize=(5.5, 5))
# for coln in range(y.shape[1]):
#     plt.plot(t, y[:, coln], linestyle="-", color=colors[coln], linewidth=4)
#     plt.scatter(t_sample, y_sample[:, coln], color=colors[coln], label="Training data " + labels[coln], s=200)
#     # ax1.plot(t, y[:, coln], color=colors[coln],linewidth=1,linestyle="-")
#     # plt.legend(loc='upper left')
#     plt.ylim(ylim)
#     plt.xticks([])
#     plt.yticks(numpy.linspace(0.8,1.6,5),fontsize=24)
#     # plt.ylabel("Reduced\n Output Concentration")
#     plt.xlim(xlim)
# plt.show(dpi=1200)
###########################################################################3

fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(5.5, 5))

for coln in range(y.shape[1]):
    ax1.plot(t, y[:, coln], linestyle="-", color=colors[coln], linewidth=1)
    ax1.scatter(t_sample, y_sample[:, coln], color=colors[coln], label="Training data " + labels[coln], s=10)
    # ax1.plot(t, y[:, coln], color=colors[coln],linewidth=1,linestyle="-")
    ax1.legend(loc='upper left')
    ax1.set_ylim(ylim)
    # ax1.set_yticks(numpy.linspace(0, 2, 5))
    ax1.set_ylabel("Reduced\n Output Concentration")
    ax1.set_xlim(xlim)

for coln in range(y.shape[1]):
    ax2.plot(numpy.arange(y_in.shape[0]), y_in[:, coln], linestyle="-", color=colors[coln], linewidth=2)
    ax2.set_ylim(0, 10)
    ax2.set_yticks(numpy.linspace(0.0, 10, 6))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.set_ylabel("Reduced\n Feed Concentration")
    ax2.set_xlim(xlim)

ax2.set_xlabel("Time (s)")




# plt.plot([660,660],[0.65,1.3],linestyle="--",color="black",linewidth=1)
# plt.plot([720,720],[0.65,1.3],linestyle="--",color="black",linewidth=1)

plt.legend()
plt.show(dpi=1200)
plt.close()

# plt.figure(figsize=(5.5, 5))
#
#
# t_1 = t[650:721]
# y_1 = y[650:721, :]
#
# for coln in range(y_1.shape[1]):
#     plt.plot(t_1, y_1[:, coln], linestyle="-", color=colors[coln],  linewidth=5)
#
# t_1_s = t[680:721:20]
# y_1_s = y[680:721:20, :]
#
# for coln in range(y_1.shape[1]):
#     plt.scatter(t_1_s, y_1_s[:, coln], linestyle="-", color=colors[coln],  s=200)
#
# plt.xticks([660,680,700,720],["0",r'$\tau$',r'2$\tau$',r'3$\tau$'],fontsize=20)
# plt.yticks([0.6,0.8,1.0,1.2],fontsize=20)
#
# plt.plot([660,660],[0,2],linestyle="--",color="black",linewidth=4)
# plt.plot([720,720],[0,2],linestyle="--",color="black",linewidth=4)
#
# plt.xlim(650,725)
# plt.ylim(0.6,1.2)
#
#
#
# plt.show(dpi=1200)
