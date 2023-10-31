import torch
import numpy as np
import torch.nn as nn
import pandas as pd

from sklearn import datasets

iris = datasets.load_iris()
# print(iris.DESCR)

# use pandas as dataframe and merge features and targetsf
feature = pd.DataFrame(iris.data, columns=iris.feature_names)
target = pd.DataFrame(iris.target, columns=['target'])
iris_data = pd.concat([feature, target], axis=1)

# keep only sepal length in cm, sepal width in cm and target
iris_data = iris_data[['sepal length (cm)', 'sepal width (cm)', 'target']]

# keep only Iris-Setosa and Iris-Versicolour classes
iris_data = iris_data[iris_data.target <= 1]
iris_data.head(5)

feature = iris_data[['sepal length (cm)', 'sepal width (cm)']]
target = iris_data[['target']]

n_samples, n_features = feature.shape

# split training data and testing data
from sklearn.model_selection import train_test_split

feature_train, feature_test, target_train, target_test = train_test_split(
    feature, target, test_size=0.3, random_state=4
)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
feature_train = sc.fit_transform(feature_train)
feature_test = sc.fit_transform(feature_test)
target_train = np.array(target_train)
target_test = np.array(target_test)

# change data to torch
feature_train = torch.from_numpy(feature_train.astype(np.float32))
feature_test = torch.from_numpy(feature_test.astype(np.float32))
target_train = torch.from_numpy(target_train.astype(np.float32))
target_test = torch.from_numpy(target_test.astype(np.float32))

class Logistic(nn.Module):
    def __init__(self, input_dimension):
        super(Logistic,self).__init__()

        self.linear=nn.Linear(input_dimension,1)

    def forward(self,x):
        y_prediction=torch.sigmoid(self.linear(x))

        return y_prediction
    
model=Logistic(n_features)

learning_rate=0.01

criteria=nn.BCELoss()

optermizer=torch.optim.SGD(model.parameters(),learning_rate)

for epoch in range(100):
    y_predict=model.forward(feature_train)
    loss=criteria(y_predict,target_train)

    loss.backward()

    optermizer.step()

    optermizer.zero_grad()

    if(epoch %10 ==9):
        print(f'epoch {epoch + 1}: loss = {loss:.8f}')

with torch.no_grad():
    y_predict=model(feature_test)
    y_predict_cls=y_predict.round()
    acc = y_predict_cls.eq(target_test).sum() / float(target_test.shape[0])
    print(f'accuracy = {acc: .4f}')



    
