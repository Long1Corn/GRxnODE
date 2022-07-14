from Engine.data_visualizer import data_vis
from Engine.model_visualizer import Visualizer
import Engine.Trainer as Trainer
from Engine.cfg import CFG
import torch

from Data.CSTR_ideal import get_data

trainset_1, testset_1, data_scale = get_data(f=1, s=5)

data_vis(trainset_1)

cfg = CFG()
trainsets = [trainset_1, ]
testsets = [testset_1, ]

trainer = Trainer.RTD(cfg, trainsets, testsets)
trainer.train()

model_path = r"output/test.pth"
trainer.model.load_state_dict(torch.load(model_path))

visualizer = Visualizer(trainer.model, trainsets, data_scale, cfg)
visualizer.run()
