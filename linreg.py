import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#prep data
X_numpy, Y_numpy = datasets.make_regression(n_samples= 100, n_features=1, noise=10, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape


input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

#loss and optimizer
lr = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

num_epochs = 100
for epoch in range(num_epochs):
    #forward and loss
    ypred = model(X)
    loss = criterion(ypred, Y)

    #back pass
    loss.backward()

    #update

    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        print(f'epoch: {epoch}, loss =  {loss.item():.3f}')

predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()