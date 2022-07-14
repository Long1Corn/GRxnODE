import numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter



def data_vis(dataset):
    t = dataset["time_data"]
    y = dataset["y_data"].squeeze()
    y_in = dataset["p_data"][0][:, 1:]

    y_sample = y[::1, :]
    t_sample = t[::1]
    y_in = y_in[::1]

    colors = ["r", "g", "b"]
    labels = ["A", "B", "C"]

    xlim = (0, 1200)
    ylim = (0, 1.8)

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(5.5, 5))

    for coln in range(y.shape[1]):
        ax1.plot(t, y[:, coln], linestyle="-", color=colors[coln], linewidth=1)
        ax1.scatter(t_sample, y_sample[:, coln], color=colors[coln], label="Training data " + labels[coln], s=10)
        ax1.legend(loc='upper left')
        ax1.set_ylim(ylim)
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

    plt.legend()
    plt.show()
    plt.close()
