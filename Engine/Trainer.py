import math
from Model_Zoo import Model_Zoo
import os
import time
from torchdiffeq import odeint
import torch
from torch.optim.lr_scheduler import StepLR, LinearLR, SequentialLR, MultiStepLR
import numpy
import matplotlib.pyplot as plt


def timeSince(since):
    """Record time
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def get_batch(datasets, device):
    """ Get data from dataset = {"time_data": time_data,
                                 "y_data": y_data,
                                "p_data": p_data,
                                "y0_data": y0_data}
        process input: p_data = [numpy[t;p1], numpy[t,p2], ...]
                       (default var:  [[y_in], [F]])
        y0_data: initial condition of reactors, all reactors required if N>1
    """

    batch_data = []

    for dataset in datasets:

        time_data = dataset["time_data"]
        p_data = dataset["p_data"]
        y_data = dataset["y_data"]
        y0_data = dataset["y0_data"]

        L = len(time_data)
        size = L - 1
        start = torch.randint(0, L - size, (1,))
        time_train = time_data[start:start + size] - time_data[start]
        time_train = torch.tensor(time_train, dtype=torch.float).to(device)

        y_train = y_data[start:start + size, :]
        y_train = torch.from_numpy(y_train).float().to(device)

        p_train = []
        for one_p in p_data:
            one_p_copy = one_p.copy()
            p_train.append(torch.tensor(one_p_copy, dtype=torch.float).to(device))

        y0_train = torch.from_numpy(y0_data).float().to(device)

        batch_data.append([y0_train, time_train, y_train, p_train])

    return batch_data


def get_ode_tol(cfg, itr):
    gamma = 1.0

    if cfg.ode_tol_step != 0 and itr >= cfg.ode_tol_step:
        gamma = cfg.ode_tol_gamma

    return cfg.ode_rtol * gamma, cfg.ode_atol * gamma


class RTD:
    """RTD model
      Model the system behavior by the residence time distribution (RTD)
      The RTD information of the system is learned from the trainset by attention mechanism.
      The system output is calculated using the RTD and the degree of mixing of the system:
            (   perfect mixing  :   the concentration is uniform in the reactor,
                                    ideal mixing
                segregation     :   feed into the reactor as segregate droplets,
                                    ideal segregation,
                non-ideal       :   the system shows both mixing and segregation effect)

      params:
          cfg         :   config (refer to cfg.py for more details)
          trainset    :   dataset for training (refer to dataset format)
          testset     :   dataset for testing
          """

    def __init__(self, cfg, trainsets, testsets):
        self.cfg = cfg
        self.trainsets = get_batch(trainsets, device=self.cfg.device)
        self.testsets = get_batch(testsets, device=self.cfg.device)
        self.model = Model_Zoo.RTD(self.cfg.model_in_var,
                                   self.cfg.model_hidden_var,
                                   self.cfg.model_out_var,
                                   N=self.cfg.RTD_N,
                                   tau=self.cfg.RTD_max_tau).to(self.cfg.device)

    def test(self):
        self.model.eval()

        test_error = []

        for testset in self.testsets:
            batch_y0, batch_t, batch_y, batch_p = testset
            self.model.process_input = batch_p
            y_pred = odeint(self.model, batch_y0, batch_t, rtol=self.cfg.ode_rtol,
                            atol=self.cfg.ode_atol,
                            method=self.cfg.ode_method).to(self.cfg.device)

            Loss_f = torch.nn.L1Loss(reduction='mean')
            loss = Loss_f(y_pred[:, -1, :], batch_y.squeeze())

            y_true = batch_y.squeeze().cpu().detach().numpy()
            y_pred = y_pred[:, -1, :].squeeze().cpu().detach().numpy()

            error = numpy.abs(y_true - y_pred)
            test_error.append(error)

        test_error = numpy.average(numpy.array(test_error))

        print("Test error = {:.4f}".format(test_error))

        return test_error

    def train(self):

        dirr = os.getcwd()
        save_dir = os.path.join(dirr, 'output/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        device = self.cfg.device

        optimizer_rate = torch.optim.Adam([{"params": self.model.rate_f.parameters(),
                                            "lr": self.cfg.lr_rate,
                                            "eps": 0.0001,
                                            "weight_decay": self.cfg.weight_decay}], betas=(0.9, 0.99))

        optimizer_rtd = torch.optim.Adam([{"params": self.model.RTD_f.parameters(),
                                           "lr": self.cfg.lr_RTD,
                                           "eps": 0.0001},
                                          {"params": self.model.G_CSTR_f.parameters(),
                                           "lr": self.cfg.lr_RTD,
                                           "eps": 0.0001}], betas=(0.9, 0.99))

        Loss_f = torch.nn.L1Loss(reduction='mean')

        scheduler_rate = MultiStepLR(optimizer_rate, milestones=self.cfg.lr_step_size, gamma=self.cfg.lr_gamma)
        scheduler_rtd = LinearLR(optimizer_rtd, start_factor=0.001, total_iters=self.cfg.lr_warmup)

        start = time.time()
        current_loss = 0

        for itr in range(1, self.cfg.itr + 1):

            self.model.train()

            optimizer_rate.zero_grad()
            optimizer_rtd.zero_grad()

            total_loss = 0

            for trainset in self.trainsets:
                batch_y0, batch_t, batch_y, batch_p = trainset

                self.model.process_input = batch_p

                rtol, atol = get_ode_tol(self.cfg, itr)

                y_pred = odeint(self.model, batch_y0, batch_t, rtol=rtol,
                                atol=atol,
                                method=self.cfg.ode_method).to(device)

                loss = Loss_f(y_pred[:, -1, :], batch_y.squeeze())
                total_loss = total_loss + loss

            total_loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.cfg.train_grad_clip)

            optimizer_rate.step()
            optimizer_rtd.step()
            scheduler_rate.step()
            scheduler_rtd.step()

            current_loss = current_loss + total_loss.item()


            if self.cfg.print_freq != 0 and itr % self.cfg.print_freq == 0:
                print('itr:%d  time:(%s) loss:%.4f lr:%.4f %.4f' % (
                    itr,
                    timeSince(start),
                    current_loss / self.cfg.print_freq,
                    optimizer_rate.param_groups[0]['lr'],
                    optimizer_rtd.param_groups[0]['lr']))

                current_loss = 0

            if self.cfg.test_freq != 0 and itr % self.cfg.test_freq == 0:
                self.test()

            if self.cfg.save_freq != 0 and itr % self.cfg.save_freq == 0:
                torch.save(self.model.state_dict(), os.path.join(save_dir, "{:06d}.pth".format(itr)))

        torch.save(self.model.state_dict(), os.path.join(save_dir, self.cfg.save_filename))

    def predict(self, datasets):

        datasets = get_batch(datasets, device=self.cfg.device)
        self.model.eval()

        predictions = []

        for dataset in datasets:
            batch_y0, batch_t, batch_y, batch_p = dataset
            self.model.process_input = batch_p
            y_pred = odeint(self.model, batch_y0, batch_t, rtol=self.cfg.ode_rtol,
                            atol=self.cfg.ode_atol,
                            method=self.cfg.ode_method).to(self.cfg.device)

            y_pred = y_pred[:, -1, :].squeeze().cpu().detach().numpy()

            predictions.append(y_pred)

        print("prediction complete")

        return predictions
