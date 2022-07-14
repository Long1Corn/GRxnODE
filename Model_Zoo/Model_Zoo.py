from functools import lru_cache

import torch.nn as nn
import torch


class RTD(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 N=100, tau=4):

        super().__init__()

        self.N = N  # num of reactor in series
        self.process_input = []
        self.tau = tau
        self.t = torch.linspace(self.tau, self.tau / self.N, self.N).cuda()

        self.rate_f = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )  # use MLP to predict reaction rate dC/dt

        for m in self.rate_f.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)

        self.RTD_f = nn.Sequential(
            # nn.Linear(1, N + 2, bias=True),
            # nn.Conv1d(1, 1, kernel_size=(3,), bias=False, padding_mode='replicate'),
            nn.Linear(1, N, bias=True),
            nn.Conv1d(1, 1, kernel_size=(3,), bias=False, padding=1, padding_mode='reflect'),
            nn.Softmax(dim=2)
        )

        for m in self.RTD_f.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, val=0)
                nn.init.constant_(m.bias, val=0)

                n = torch.Tensor([1.5]).cuda()
                RTD_r = self.t ** (n - 1) / torch.lgamma(n).exp() * (n ** n) * (-self.t * n).exp()
                RTD = torch.log(RTD_r / self.N * self.tau)

                m.bias = torch.nn.Parameter(RTD)

            if isinstance(m, nn.Conv1d):
                nn.init.constant_(m.weight, val=1.0 / 3)
                for param in m.parameters():
                    param.requires_grad = False

        # with torch.no_grad():
        #     n = int(2 * N / tau)
        #     w1 = torch.ones(n)
        #     w2 = torch.ones(N - n) - 2
        #     w3 = torch.cat((w2, w1), dim=0)
        #     self.RTD_f[0].bias = torch.nn.Parameter(w3)

        self.G_CSTR_f = nn.Linear(1, 1)

        for m in self.G_CSTR_f.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, val=0)
                nn.init.constant_(m.bias, val=1.5)

        # with torch.no_grad():
        #     w1 = torch.linspace(-1, 1, N // tau)
        #     w2 = torch.linspace(1, -1, N - N // tau)
        #     w3 = torch.cat((w1, w2), dim=0)
        #
        #     self.RTD_f[0].bias = torch.nn.Parameter(w3)

    def forward(self, t, y):
        # y_in [var_num, 1] 'feed
        # F 'space velocity
        # y_0 [var_num, N] 'state of N reactor
        y_in, F = self._get_process_input(int(t), train=self.training)
        self.F = F

        RTD = self.get_RTD(torch.tensor([0.05]).cuda()).view(-1, 1)

        y_in_RTD = y_in * RTD

        dy = torch.zeros_like(y)

        rate = self.get_rate(y.clone())

        dy[0, :] = (y_in_RTD[0, :] + 0 - y[0, :]) * F * self.N / self.tau + rate[0, :]
        dy[1:, :] = (y_in_RTD[1:, :] + y[:-1, :] - y[1:, :]) * F * self.N / self.tau + rate[1:, :]

        return dy

    def get_rate(self, y):

        rate = self.rate_f(y)

        return rate

    def get_RTD(self, input):

        RTD = self.RTD_f(input.unsqueeze(0).unsqueeze(0)).squeeze().squeeze()

        return RTD

    @lru_cache(maxsize=2000)
    def _get_process_input(self, t, train=True):
        var = []
        dataset = self.process_input
        for data in dataset:

            L0 = 0
            L2 = len(data[:, 0])
            L1 = (L0 + L2) // 2

            cont = 1
            if data[-1, 0] <= t:
                var.append(data[L1, 1:])
                cont = 0

            while cont:
                if data[L1, 0] <= t < data[L1 + 1, 0]:
                    var.append(data[L1, 1:])
                    break
                elif data[L1, 0] > t:
                    L2 = L1
                    L1 = (L0 + L2) // 2
                elif data[L1 + 1, 0] <= t:
                    L0 = L1 + 1
                    L1 = (L0 + L2) // 2

        return var
