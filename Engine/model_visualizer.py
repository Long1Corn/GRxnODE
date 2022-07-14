import numpy
import numpy as np
import scipy.interpolate
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle


def plot_slices(x, y, z, data, xslice, yslice, zslice, x_d, y_d, z_d, ax=None, ):
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
    # Normalize data to [0, 1] range
    vmin, vmax = 0.0, 1.0
    data_n = (data - vmin) / (vmax - vmin)
    # Take slices interpolating to allow for arbitrary values

    data_z = scipy.interpolate.interp1d(z, data_n, axis=2)(zslice)
    data_y = scipy.interpolate.interp1d(y, data_n, axis=1)(yslice)
    data_x = scipy.interpolate.interp1d(x, data_n, axis=0)(xslice)
    # Pick color map
    cmap = plt.cm.YlOrRd
    # Plot X slice
    xs, ys, zs = data.shape

    ax.scatter3D(x_d, y_d, z_d, label="train set", color="royalblue", s=5, alpha=0.7)

    zplot = ax.plot_surface(x[:, np.newaxis], y[np.newaxis, :], np.atleast_2d(zslice),
                            facecolors=cmap(data_z), shade=False, alpha=0.7, linewidth=0)

    yplot = ax.plot_surface(x[:, np.newaxis], yslice, z[np.newaxis, :],
                            facecolors=cmap(data_y), shade=False, alpha=0.7, linewidth=0)

    xplot = ax.plot_surface(xslice, y[:, np.newaxis], z[np.newaxis, :], cmap="YlOrRd",
                            facecolors=cmap(data_x), shade=False, alpha=0.7, linewidth=0)

    ax.set_xlim3d(0.5, 1.5)  # Reproduce magnification
    ax.set_ylim3d(0.5, 1.5)  # ...
    ax.set_zlim3d(0.5, 1.5)

    ax.view_init(elev=10, azim=45)

    fig.colorbar(xplot, ax=ax, shrink=0.7)

    ax.legend()

    ax.xaxis._axinfo["grid"]['linestyle'] = ":"
    ax.yaxis._axinfo["grid"]['linestyle'] = ":"
    ax.zaxis._axinfo["grid"]['linestyle'] = ":"

    plt.show()

    return xplot, yplot, zplot


class Visualizer:
    def __init__(self, model, datasets, data_scale, cfg):
        self.model = model
        self.cfg = cfg
        self.data_scale = data_scale
        self.datasets = datasets

    def get_scatter(self):
        scatter_x = []
        scatter_y = []
        scatter_z = []

        for dataset in self.datasets:
            x = dataset["y_data"][:, -1, 0]
            y = dataset["y_data"][:, -1, 1]
            z = dataset["y_data"][:, -1, 2]

            scatter_x.append(x)
            scatter_y.append(y)
            scatter_z.append(z)

        return scatter_x, scatter_y, scatter_z

    def true_rate(self, conc):

        C = (conc * self.data_scale[0])
        k = [0.5, 0.3, 0.1, 0.006]

        rate = [-k[0] * C[0] ** 2 + k[3] * C[1] ** 2,
                k[0] * C[0] ** 2 + k[2] * C[2] ** 2 - (k[1] + k[3]) * C[1] ** 2,
                k[1] * C[1] ** 2 - k[2] * C[2] ** 2]

        rate = rate / self.data_scale[0]

        return rate

    def pred_rate(self, conc):
        self.model.to(self.cfg.device)
        rate = self.model.get_rate(torch.tensor(conc).float().to(self.cfg.device)).cpu().detach().numpy()

        return rate

    def run(self):
        self.vis_rate()
        self.vis_rtd()

    def vis_rate(self):
        self.model.eval()

        minn = 0.7
        maxn = 1.3

        X_min, X_max = minn, maxn
        Y_min, Y_max = minn, maxn
        Z_min, Z_max = minn, maxn

        X, Y, Z = np.mgrid[X_min:X_max:30j, Y_min:Y_max:30j, Z_min:Z_max:30j]
        positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

        true_map = np.zeros([30 * 30 * 30, 3])
        pred_map = np.zeros([30 * 30 * 30, 3])

        num = 0
        for pos in positions:
            true_map[num] = self.true_rate(pos)
            num = num + 1

        num = 0
        for pos in positions:
            pred_map[num] = self.pred_rate(pos)
            num = num + 1

        scatter_x, scatter_y, scatter_z = self.get_scatter()

        x = np.linspace(X_min, X_max, 30)
        y = np.linspace(Y_min, Y_max, 30)
        z = np.linspace(Z_min, Z_max, 30)
        dev = np.abs(pred_map / (true_map + 10 ** -5) - 1)[:, 0].reshape(30, 30, 30)
        plot_slices(x, y, z, dev, 1, 1, 1, scatter_x, scatter_y, scatter_z)

        X = X[:, :, 0]
        Y = Y[:, :, 0]

        Title = ["A", "B", "C"]

        for n in range(3):
            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(131, projection='3d')

            XY_true_map = true_map[15::30, n].reshape(30, 30)
            XY_pred_map = pred_map[15::30, n].reshape(30, 30)

            ax1.plot_surface(X, Y, XY_true_map, linewidth=0, cmap='winter', alpha=0.7, edgecolors='b')
            ax1.plot_surface(X, Y, XY_pred_map, linewidth=0, cmap='autumn', alpha=0.7, edgecolors='g')
            ax1.view_init(10, 45)
            ax1.title.set_text('rate curve')
            ax1.zaxis.set_ticks(np.linspace(-0.2, 0.2, 5))
            ax1.xaxis.set_ticks(np.linspace(0.7, 1.3, 4))
            ax1.yaxis.set_ticks(np.linspace(0.7, 1.3, 4))

            ax2 = fig.add_subplot(132)
            dev = abs(XY_pred_map / (XY_true_map + 10 ** -5) - 1)
            dev = np.clip(dev, 0, 1)
            plot2 = ax2.contourf(X, Y, dev, 20, cmap='YlOrRd', vmin=0, vmax=1.01)
            cbar = fig.colorbar(plot2, ax=ax2, ticks=[0, 0.25, 0.5, 0.75, 1])
            # cbar.ax.get_yaxis().set_ticks(np.linspace(0.0, 1.0, 6))
            ax2.scatter(scatter_x, scatter_y, label="train set", s=5, color="royalblue")
            ax2.set_xlim(minn, maxn)
            ax2.set_ylim(minn, maxn)
            ax2.title.set_text('relative error')
            ax2.legend()

            ax3 = fig.add_subplot(133)
            dev = abs(XY_pred_map - XY_true_map)
            plot3 = ax3.contourf(X, Y, dev, 20, cmap='YlOrRd')
            fig.colorbar(plot3, ax=ax3)
            ax3.scatter(scatter_x, scatter_y, label="train set", s=5, color="royalblue")
            ax3.title.set_text('absolute error')
            ax3.set_xlim(minn, maxn)
            ax3.set_ylim(minn, maxn)
            ax3.legend()

            fig.suptitle(Title[n])
            plt.show()
            plt.clf()

    def vis_rtd(self):

        RTD_0 = self.model.get_RTD(torch.tensor([0.05]).to(self.cfg.device))
        RTD_1 = RTD_0.cpu().detach().numpy()

        plt.figure()
        plt.plot(numpy.linspace(self.cfg.RTD_max_tau / self.cfg.RTD_N,
                                self.cfg.RTD_max_tau,
                                self.cfg.RTD_N),
                 numpy.flip(RTD_1) * self.cfg.RTD_N / self.cfg.RTD_max_tau)
        plt.ylim(0, 1)
        plt.xlim(0, self.cfg.RTD_max_tau)
        plt.title("RTD")
        plt.show()

        pass
