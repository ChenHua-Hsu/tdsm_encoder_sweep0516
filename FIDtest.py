from collections import OrderedDict
import numpy as np
import torch
from torch import nn, optim

a=[[1,1],[2,1]]
b=[[2,1],[3,4]]


# print(np.matmul(a,b))
# def FID(data1,energy1,data2,energy2):
#     data1_x_mean=torch.mean(data1[:,1])
#     data1_y_mean=torch.mean(data1[:,2])
#     data1_z_mean=torch.mean(data1[:,3])
#     data2_x_mean=torch.mean(data2[:,1])
#     data2_y_mean=torch.mean(data2[:,2])
#     data2_z_mean=torch.mean(data2[:,3])
#     energy1_mean=torch.mean(energy1)
#     energy2_mean=torch.mean(energy2)

#     data1_x_covariance=torch.cov(data1[:,1])
#     data1_y_covariance=torch.cov(data1[:,2])
#     data1_z_covariance=torch.cov(data1[:,3])
#     data2_x_covariance=torch.cov(data2[:,1])
#     data2_y_covariance=torch.cov(data2[:,2])
#     data2_z_covariance=torch.cov(data2[:,3])
#     energy1_covariance=torch.cov(energy1)
#     energy2_covariance=torch.cov(energy2)

#     mean=((data1_x_mean-data2_x_mean)**2+(data1_y_mean-data2_y_mean)**2+(data1_z_mean-data2_z_mean)**2+(energy1_mean-energy2_mean)**2)
#     covariance=(data1_x_covariance**2+data1_y_covariance**2+data1_z_covariance**2+data2_x_covariance**2+data2_y_covariance**2+\
#                 data2_z_covariance**2+energy1_covariance**2+energy2_covariance**2-2*(torch.sqrt(data1_x_covariance*data2_x_covariance)+\
#                 torch.sqrt(data1_y_covariance*data2_y_covariance)+torch.sqrt(data1_z_covariance*data2_z_covariance)+torch.sqrt(energy1_covariance*energy2_covariance)))
#     return (mean+covariance)




# import torch
# _ = torch.manual_seed(123)
# from torchmetrics.image.fid import FrechetInceptionDistance
# fid = FrechetInceptionDistance(feature=64)
# # generate two slightly overlapping image intensity distributions
# imgs_dist1 = torch.randint(0, 20,(2,2),dtype=torch.uint8)
# imgs_dist2 = torch.randint(0,20,(2,2),dtype=torch.uint8)
# # fid.update(imgs_dist1, real=True)
# # fid.update(imgs_dist2, real=False)
# # fid.compute()

from fid_score import Score

original_energy = [[6.0, 3.0]]
original_x = [[5.0, 3.0]]
original_y = [[5.0, 3.0]]
original_z = [[5.0, 3.0]]
all_e_g = [[4.8, 3.2]]
all_x_g = [[4.8, 3.2]]
all_y_g = [[4.8, 3.2]]
all_z_g = [[4.8, 3.2]]

goi=Score(original_energy, original_x, original_y, original_z, all_e_g, all_x_g, all_y_g, all_z_g)

goi.FID_score()