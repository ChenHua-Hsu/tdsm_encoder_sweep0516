from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs=4
batch_size=10
learning_rate=0.001

train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
    transform=transforms.ToTensor(), download=True)
#print(train_dataset)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
    transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
    shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
    shuffle=True)


# class MnistDataset(Dataset):

    # def __init__(self,path,transform=None):
    #    data=np.loadtxt(path,dtype=np.float32,delimiter=',',skiprows=1)
    #    self.x=torch.from_numpy(data[:,1:])
    #    self.y=torch.from_numpy(data[:,[0]].flatten().astype(np.longlong))
    #    self.n_sample =data.shape[0]
    #    self.transform=transform

#     def __getitem__(self,index):
#        sample=self.x[index], self.y[index]

#        if self.transform:
#            sample=self.transform(sample)

#        return sample
    
#     def __len__(self):
#        return self.n_sample
    
# class ToImage:

#     def __call__(self,sample):
#         inputs,target=sample

#         return inputs.view(28,28).unsqueeze(0), target
    
# train_path='train.csv'
# test_path='test.csv'
# train_dataset=MnistDataset(train_path)
# test_dataset=MnistDataset(test_path)

#train_loader=DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
#test_loader=DataLoader(test_dataset, batch_size=batch_size,shuffle=True)


class CNNnet(nn.Module):

    def __init__(self):
        super(CNNnet,self).__init__()

        self.conv1=nn.Conv2d(in_channels=1,out_channels=3,kernel_size=5)
        self.conv2=nn.Conv2d(in_channels=3,out_channels=9,kernel_size=5)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(9*4*4,100)
        self.fc2=nn.Linear(100,50)
        self.fc3=nn.Linear(50,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=torch.flatten(x,1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return x
    
model=CNNnet().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

#Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # init optimizer
        optimizer.zero_grad()
        
        # forward -> backward -> update
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()

        optimizer.step()

        if (i + 1) % 1000 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

print('Finished Training')

#Testing
# 4) Testing loop
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        # max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {i}: {acc} %')
print("test")


