#1) design model: input size, output size, forward pass
#2)Loss and Optimizer
#3)Training loop: forward pass, backward pass, modify weights

import torch
import torch.nn as nn



# f = w * x +

#f = 2 * x

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype = torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features



class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

#model prediction
model = LinearRegression(input_size, output_size)


print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

#training

lr = 0.02
n_iters = 2000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)


for epoch in range(n_iters):
    #forward pass
    y_pred = model(X)

    #loss

    l = loss(Y, y_pred)

    
    #gradients
    l.backward() #dl/dw

    optimizer.step()

    #zero gradients
    optimizer.zero_grad()


    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')    

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')