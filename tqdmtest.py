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

# create default evaluator for doctests

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

# create default optimizer for doctests

#param_tensor = torch.zeros([1], requires_grad=True)
#default_optimizer = torch.optim.SGD([param_tensor], lr=0.1)

# create default trainer for doctests
# as handlers could be attached to the trainer,
# each test must define his own trainer using `.. testsetup:`

def get_default_trainer():

    def train_step(engine, batch):
        return batch

    return Engine(train_step)

# create default model for doctests

default_model = nn.Sequential(OrderedDict([
    ('base', nn.Linear(1, 1)),
    ('fc', nn.Linear(1, 1))
]))

#manual_seed(666)
#torch.manual_seed(666)
#torch.backends.cudnn.benchmark = False


import torch
import ignite.metrics.gan.fid as fid

# Assuming default_model is already defined
# ...

# Create an instance of the metric
metric = FID(num_features=1, feature_extractor=default_model)

# Recreate the default_model and metric
default_model = nn.Sequential(OrderedDict([
    ('base', nn.Linear(1, 1)),
    ('fc', nn.Linear(1, 1))
]))
torch.save(default_model.state_dict(), 'model_state_dict.pt')

# Now, in subsequent runs, load the model's state dictionary
#loaded_model_state_dict = torch.load('model_state_dict.pt')

metric = FID(num_features=1, feature_extractor=default_model)
metric.attach(default_evaluator, "fid")

for x in range(0, 5):
    x_try = [[5.0, 3.0]]
    y_try = [[3.0, 1.0]]
    x_try = torch.Tensor(x_try).reshape(-1, 1)
    y_try = torch.Tensor(y_try).reshape(-1, 1)
    
    state = default_evaluator.run([[x_try, x_try]])
    print(state.metrics["fid"])


po=torch.zeros(100,40)
vector=torch.ones(40,3)
lin=nn.Linear(40,3)
#lin.weight.data=torch.FloatTensor(vector.T)
#print(lin.weight.data)

test=torch.rand(1,100)
test=test.reshape(-1,1)
#print(test)


import numpy as np
from sklearn.preprocessing import QuantileTransformer
rng = np.random.RandomState(0)
#print(rng)
X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
#print(X)
qt = QuantileTransformer(n_quantiles=5, random_state=0)
qt.fit_transform(X)

#print(qt.fit_transform(X))

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
quantile_transformer = QuantileTransformer(random_state=0)
#print(X)

Z_test=[[5.0,2.0]]
z_try=[[1.0,3.0]]
#X_train_trans = quantile_transformer.fit_transform(Z_test)
#@print(X_train_trans)
#X_test_trans = quantile_transformer.transform(X_test)
#print(np.percentile(X_train[:, 0], [0, 25, 50, 75, 100]) )

# fig = plt.figure(figsize=(12, 4))
# plt.plot(X,color='blue',label='Experiment:Al')
# plt.plot(qt.fit_transform(X),color='green',label='Experiment:Cu')
# plt.plot(qt.inverse_transform(qt.fit_transform(X)),color='gray')
# plt.legend(loc='best')
# #plt.xlabel(r'$\theta$')
# #plt.ylabel(r'$\dfrac{d\sigma}{d\Omega}$')
# plt.savefig('test.png',dpi=300)
# plt.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

vec=MinMaxScaler()

#print(scaler.inverse_transform(scaler.fit_transform(Z_test)))
#print(vec.inverse_transform(vec.fit_transform(z_try)))

really=[[[1,2,3,4],[1,2,3,4]],[[1,2,3,4],[1,2,3,4]]]
#print(really[:][0])