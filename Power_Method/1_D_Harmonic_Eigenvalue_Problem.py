import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import torch as tr

def normalized(y):
    return (tr.sum(y * y) / y.shape[0])**0.5

def normalized_np(y):
    return (np.sum(y * y) / y.shape[0])**0.5

def pde(x, y):
    u_kold = y.clone().detach()
    dy_xx = dde.grad.hessian(y, x)
    Lu_k=-dy_xx - gamma*y

    return Lu_k/normalized(Lu_k) - u_kold/normalized(u_kold)

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
data = dde.data.PDE(geom, pde, bc, 32, 0, solution=func, num_test=100, train_distribution="uniform")
layer_size = [1] + [10] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
net.apply_output_transform(transform)

model = dde.Model(data, net)
model.compile("adam", lr=0.00001, metrics=["l2 relative error"], external_trainable_variables=None)
model.train(iterations=50000)

x = geom.uniform_points(1000, True)
u = model.predict(x)
u_exact = func(x)

def L(x,y):
    dy_xx = dde.grad.hessian(y, x)

    Lu_k = -dy_xx
    return Lu_k
Lu = model.predict(x,operator=L)
gamma_pre =(np.dot(Lu.T,u) / np.dot(u.T,u))[0,0]
print(f"The predicted eigenvalue is: {gamma_pre}")
print(f"The real eigenvalue is: {np.pi**2}")
print(f"The relative error of eigenvalue is: {(gamma_pre-np.pi**2)/np.pi**2}")


plt.plot(x, np.sin(np.pi*x)/normalized_np(np.sin(np.pi*x)), label="ground_truth")
plt.plot(x, abs(u/normalized_np(u)), label="prediction")
plt.legend()
plt.show()

plt.plot(x, abs(u/normalized_np(u))-np.sin(np.pi*x)/normalized_np(np.sin(np.pi*x)), label="error")
plt.legend()
plt.show()
