import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate two Gaussian blobs
mean1 = [1, 1]
cov1 = [[0.2, 0], [0, 0.2]]
X1 = np.random.multivariate_normal(mean1, cov1, 100)
#print(X1)

mean2 = [0, 0]
cov2 = [[0.1, 0], [0, 0.1]]
X2 = np.random.multivariate_normal(mean2, cov2, 100)

# Create the dataset
X = np.concatenate([X1, X2], axis=0)

y = np.concatenate([np.zeros(100), np.ones(100)], axis=0)

# Convert data to tensors
X_tensor = torch.from_numpy(X).float()
print(X_tensor)
y_tensor = torch.from_numpy(y).long()
print(y_tensor)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the loss function and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Train the neural network
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = net(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step() #自動調整參數
# Make predictions
with torch.no_grad():
    outputs = net(X_tensor)
    _, predicted = torch.max(outputs.data, 1)

# Plot the data and predictions
plt.scatter(X[:,0], X[:,1], c=predicted.numpy())
plt.show()
