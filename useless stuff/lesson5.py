import torch
from torch.autograd import Variable
from torch import nn
from torch import tensor


x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        ypred = self.linear(x)
        return ypred

model = Model()

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(500):
    ypred = model(x_data)

    loss = criterion(ypred, y_data)
    #print(epoch, loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

hour_var = Variable(torch.Tensor([[4.0]]))
print("final prediction", 4, model.forward(hour_var).data[0][0])
