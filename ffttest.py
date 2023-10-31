import numpy as np
import scipy
from scipy import integrate 
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import math
import torch

# x=np.linspace(0,100,10000,endpoint=True)

# y=np.sin(np.pi*x/2)
# print(y)
# z=fft(y)
# print(z)
# fig = plt.figure(figsize=(8, 4))
# #plt.plot(x,abs(z),color='red')
# #plt.plot(x,abs(y),color='blue')
# plt.plot(x,abs(fft(x)),color='green')
# plt.show()



# from sklearn.preprocessing import MinMaxScaler
# data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
# scaler_e = MinMaxScaler().fit(torch.tensor(data[:,0]).reshape(-1,1))

import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler

# qt = QuantileTransformer(n_quantiles=5, random_state=0)

training_dataset=[[[1.0,2.0,3.0,4.0],[2.0,3.0,4.0,5.0]],[[1.0,5.0,3.0,4.0],[2.0,3.0,4.0,4.0]]]

e=[0.0,1.0,2.0,0.0]
x=[-20.,-10.,0.,10.]
y=[-20.,-10.,0.,10.]
z=[1.0,2.0,3.0,4.0]
scaler_e = MinMaxScaler().fit(torch.tensor(e).reshape(1,-1))
scaler_x = MinMaxScaler().fit(torch.tensor(x).reshape(1,-1))
scaler_y = MinMaxScaler().fit(torch.tensor(y).reshape(1,-1))
scaler_z = MinMaxScaler().fit(torch.tensor(z).reshape(1,-1))

e=[1.0,2.0,3.0,1.0]
x=[-19.,-9.,1.,11.]
y=[-19.,-9.,1.,11.]
z=[2.0,3.0,4.0,5.0]

#print(e,x,y,z)

e=scaler_e.transform(torch.tensor(e))
x=scaler_x.transform(torch.tensor(x))
y=scaler_y.transform(torch.tensor(y))
z=scaler_z.transform(torch.tensor(z))

print(e,x,y,z)