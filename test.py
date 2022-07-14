from Engine.model_visualizer import Visualizer
import Engine.Trainer as Trainer
from Engine.cfg import CFG
import numpy as np
import os
import torch
import pandas as pd

cfg = CFG()
cfg.itr = 100
cfg.lr_step_size = [700, ]

cfg.train_batch_size = 0

cfg.model_in_var = 3
cfg.model_hidden_var = 24
cfg.model_out_var = 3

cfg.RTD_N = 100
cfg.RTD_max_tau = 4

cfg.ode_method = "adaptive_heun"
# cfg.ode_method = "dopri5"
# cfg.ode_method = "euler"
cfg.ode_rtol = 1e-3
cfg.ode_atol = 1e-4
cfg.ode_tol_step = 0
cfg.ode_tol_gamma = 0.1

cfg.save_freq = 0
cfg.test_freq = 1
cfg.print_freq = 1

cfg.lr_rate = 0.005
cfg.lr_RTD = 0.01
cfg.weight_decay = 1e-4

cfg.device = "cuda"

# save_name = "A_C_S5_D20.pth"


from Data.CSTR_ideal import get_data

trainset_1, testset_1, data_scale = get_data(f=1, s=5, data="CSTR_1")
trainsets = [trainset_1, ]
testsets = [testset_1, ]

# cfg.save_filename = save_name
trainer = Trainer.RTD(cfg, trainsets, testsets)

# model_path = r"/home/pengfei/projects/neural_ode/NeuralODE/output/{}".format(save_name)
# trainer.model.load_state_dict(torch.load(model_path))

trainer.train()

visualizer = Visualizer(trainer.model, trainsets[0], data_scale)
visualizer.run()
