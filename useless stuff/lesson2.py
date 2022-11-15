import torch
import math
import random
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0] 

w = 1.0

def forward(x):
    return x * w


def loss(x, y):
    ypred = forward(x)
    return (ypred - y) * (ypred - y)

def gradient(x, y):
    ypred = forward(x)
    return 2 * x * (x* w - y)

print("predict before training", 4, forward(4))

for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad
        print("\tgrad: ", x_val, y_val, grad)
        l = loss(x_val, y_val)
    print("progress:", epoch, "w=", w, "loss=", l )

print("predict after training:", "4 hours:", forward(4))

# w_list = []
# mse_list = []

# for w in np.arange(0.0, 4.1, 0.1):
#     print("w=", w)
#     l_sum = 0

#     for x_val, y_val in zip(x_data, y_data):
#         ypredval = forward(x_val)
#         l = loss(x_val, y_val)
#         l_sum += l
#         print("\t", x_val, y_val, ypredval, 1)

#     print("MSE=", l_sum / len(x_data))
#     w_list.append(w)
#     mse_list.append(l_sum / len(x_data))

# plt.plot(w_list, mse_list)
# plt.ylabel('Loss')
# plt.xlabel('w')
# plt.show()    