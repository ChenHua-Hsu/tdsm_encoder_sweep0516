import torch
import numpy as np
import matplotlib.pyplot as plt

# create data set
a = torch.tensor([1.0], requires_grad=True)
y=torch.tensor([18.0])
x=torch.tensor([3.0])
q=[]


#training

for i in range(100):
    a.grad.zero_()
    loss = y - (a * x)
    loss.backward()
    with torch.no_grad():
        a -= a.grad * 0.01 * loss
        q+=a
k=np.linspace(0,99,99)
plt.plot(k,q)
plt.show()