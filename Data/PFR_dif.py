from Engine.model_visualizer import Visualizer
from NeuralODE.Engine.cfg import CFG
import numpy as np
import os


def get_dataset(data_dir, f=5, s=1, scale=None):
    data_time = np.load(os.path.join(data_dir, 'Reactor_ODE_1_data_t.npy'))
    data_output = np.load(os.path.join(data_dir, 'Reactor_ODE_1_data_output.npy'))
    data_input = np.load(os.path.join(data_dir, 'Reactor_ODE_1_data_input.npy'))

    L = len(data_time)
    L = int(L * s)

    data_time = data_time[:L]
    data_output = data_output[:L]
    data_input = data_input[:L]

    data_std = np.ndarray.std(data_output, axis=0)
    data_mean = np.ndarray.mean(data_output, axis=0)

    if scale is None:
        scale = [data_mean, data_std]

    data_output = np.expand_dims(data_output, axis=1)
    data_output = (data_output[:, 0:1, :]) / scale[0]
    data_input = (data_input) / scale[0]


    y0_data = (np.zeros((100, 3))) / scale[0]

    time_data = data_time[::f]
    y_data = data_output[::f, 0:1, :]

    y_in_data = np.concatenate((data_time.reshape(-1, 1), data_input), axis=1)
    F_data = np.concatenate((np.arange(L).reshape(-1, 1), np.zeros((L, 1)) + 0.05), axis=1, dtype=np.float32)
    p_data = [y_in_data, F_data]

    dataset = {"time_data": time_data,
               "y_data": y_data,
               "p_data": p_data,
               "y0_data": y0_data}

    return dataset, scale


def get_data(f=5, s=1 , data=""):
    train_dir = r"/home/pengfei/projects/neural_ode/TheoryGuidedRNN/dataset/PFR_dif_train"
    test_dir = r"/home/pengfei/projects/neural_ode/TheoryGuidedRNN/dataset/PFR_dif_test"

    trainset, scale = get_dataset(train_dir, f=f, s=s)
    testset, _ = get_dataset(test_dir, f=5, s=1, scale=scale)

    return trainset, testset, scale


if __name__ == "__main__":
    trainset, testset, scale = get_data(data="CSTR_2")

    y = trainset["y_data"].squeeze()
    p = trainset["p_data"][0]
    import matplotlib.pyplot as plt

    plt.plot(y)
    plt.show()
    plt.clf()

    plt.plot(p[:, 1:])
    plt.show()