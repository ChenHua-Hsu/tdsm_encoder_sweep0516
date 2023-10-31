from tqdm import tqdm
from tqdm import trange
import time

import time, functools, torch, os, random, fnmatch, psutil, argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adamax
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
#from trans_tdsm import Gen, loss_fn, pc_sampler
#import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
import ignite
import torchvision.transforms as transforms
import torchvision.utils as vutils

from collections import OrderedDict
from ignite.metrics import FID, InceptionScore
import torch
from torch import nn, optim

from ignite.engine import Engine

import ignite.metrics.gan.fid as fid


def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

# create default optimizer for doctests

param_tensor = torch.zeros([1], requires_grad=True)
default_optimizer = torch.optim.SGD([param_tensor], lr=0.1)

# create default trainer for doctests
# as handlers could be attached to the trainer,
# each test must define his own trainer using `.. testsetup:`

def get_default_trainer():

    def train_step(engine, batch):
        return batch

    return Engine(train_step)

# create default model for doctests

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
#torch.backends.cudnn.deterministic = True

default_model = nn.Sequential(OrderedDict([
    ('base', nn.Linear(1, 1)),
    ('fc', nn.Linear(1, 1))
]))

#manual_seed(666)

#torch.save(default_model.state_dict(), 'model_state_dict.pt')

# Now, in subsequent runs, load the model's state dictionary
loaded_model_state_dict = torch.load('model_state_dict.pt')

metric = FID(num_features=1, feature_extractor=default_model)
metric.attach(default_evaluator, "fid")
all_e = [[5.0,3.0]]
all_e_gen = [[3.0,2.0]]
all_x = [[5.0,3.0]]
all_x_gen = [[3.0,1.0]]
all_y = [[5.0,3.0]]
all_y_gen = [[3.0,1.0]]
all_z = [[5.0,3.0]]
all_z_gen = [[3.0,1.0]]
all_e=torch.Tensor(all_e).reshape(-1,1)
all_e_gen=torch.Tensor(all_e_gen).reshape(-1,1)
all_x=torch.Tensor(all_x).reshape(-1,1)
all_x_gen=torch.Tensor(all_x_gen).reshape(-1,1)
all_y=torch.Tensor(all_y).reshape(-1,1)
all_y_gen=torch.Tensor(all_y_gen).reshape(-1,1)
all_z=torch.Tensor(all_z).reshape(-1,1)
all_z_gen=torch.Tensor(all_z_gen).reshape(-1,1)

state_e = default_evaluator.run([[all_e,all_e_gen]])
state_x = default_evaluator.run([[all_x,all_x_gen]])
state_y = default_evaluator.run([[all_y,all_y_gen]])
state_z = default_evaluator.run([[all_z,all_z_gen]])
#print(default_model.weight)
print(state_e.metrics["fid"])
print(state_x.metrics["fid"])
print(state_y.metrics["fid"])
print(state_z.metrics["fid"])

