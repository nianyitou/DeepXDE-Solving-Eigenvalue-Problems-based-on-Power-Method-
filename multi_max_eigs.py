import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import torch as tr
from nn_multi_net import Multi_FNN

def normalized(y):
    return (tr.dot(y, y))**0.5

def pde(x, y):
    dimy = y.shape[1]
    Lu_k = y.clone().detach()
    loss = tr.zeros(y.shape[0], 1)
    for i in range(dimy):
        Lu_k[:, i:i+1] = dde.grad.hessian(y[:,i:i+1], x) + 1000.0*y[:,i:i+1]
        Lu_k = Lu_k.clone().detach()
        for j in range(i):
            cross = tr.dot(Lu_k[:, j], Lu_k[:, i]).clone().detach()
            Lu_k[:, i] = Lu_k[:, i] - cross * Lu_k[:, j]
            Lu_k[:, i] = Lu_k[:, i] / normalized(Lu_k[:, i])
            Lu_k[:, i] = Lu_k[:, i].clone().detach()
        Lu_k[:, i] = Lu_k[:, i] / normalized(Lu_k[:, i])
        Lu_k[:, i] = Lu_k[:, i].clone().detach()
        loss += tr.abs(Lu_k[:, i:i+1] - y[:,i:i+1])

    return loss

def transform(x, y):
    res = x * (1 - x)
    return res * y

def boundary(x, on_boundary):
    return on_boundary

def func(x):
    return np.sin(np.pi*x)

gamma = tr.tensor([0], dtype=tr.float64)

geom = dde.geometry.Interval(0, 1)
bc = []
data = dde.data.PDE(geom, pde, bc, 128, 0, solution=func, num_test=100, train_distribution="uniform")
layer_size = [1] + [10] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = Multi_FNN(2, layer_size, activation, initializer)
net.apply_output_transform(transform)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"], external_trainable_variables=None)
model.train(iterations=10000)

x = geom.uniform_points(1000, True)
u = model.predict(x)
u_exact = func(x)
for i in range(u.shape[1]):
    plt.plot(x, u[:,i], label=f"eig_{i}")
plt.legend()
plt.show()

def operator_func(i):
    def operator(x,y):
        Lu_k = dde.grad.hessian(y[:, i:i+1], x) + 1000.0 * y[:, i:i+1]
        return Lu_k
    return operator

for i in range(u.shape[1]):
    Lu = model.predict(x, operator=operator_func(i))
    gamma_pre = (np.dot(Lu.T, u[:, i:i+1]) / np.dot(u[:, i:i+1].T, u[:, i:i+1]))[0, 0]
    print(gamma_pre, 1000 - (i+1)**2*np.pi ** 2, abs(1000 - (i+1)**2*np.pi ** 2 - gamma_pre) / (1000 - (i+1)**2*np.pi ** 2))



