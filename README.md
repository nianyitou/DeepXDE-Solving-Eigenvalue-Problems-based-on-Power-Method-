# DeepXDE: Solving Eigenvalue Problems based on Power Method

## installation:

>installation package: pip install deepxde <br>
><br>
>Use (python -m deepxde.backend.set_default_backend pytorch) to set the default backend pytorch

## 1. File: 1_D_Harmonic_Eigenvalue_Problem:

Solve 1D harmonic eigenvalue problem: <br>
u_xx + λ*u = 0 <br>
we solve for the nearest eigenvalue to gamma (here gamma=0):<br>
```python
gamma = tr.tensor([0], dtype=tr.float64)
```
<br>
Normalized function for pytorch and numpy:<br>

```python
def normalized(y):
    return (tr.sum(y * y) / y.shape[0])**0.5

def normalized_np(y):
    return (np.sum(y * y) / y.shape[0])**0.5
```
<br>
PDE function for deepxde (see Neural Networks Based on Power Method and Inverse Power Method for Solving Linear Eigenvalue Problems for details):<br>

```python
def pde(x, y):
    u_kold = y.clone().detach()
    dy_xx = dde.grad.hessian(y, x)
    Lu_k=-dy_xx - gamma*y

    return Lu_k/normalized(Lu_k) - u_kold/normalized(u_kold)
```
<br>
Ground truth:<br>

```python
def func(x):
    return np.sin(np.pi*x)
```
<br>
Boundary Conditions:<br>

```python
def transform(x, y):
    res = x * (1 - x)
    return res * y

def boundary(x, on_boundary):
    return on_boundary
```
<br>
The solution interval is (0,1). <br>
Use output transform set boundary conditions. <br>
Train points are linespace(0,1,8), valid points are linespace(0,1,100). <br>

```python
geom = dde.geometry.Interval(0, 1)
bc = []
data = dde.data.PDE(geom, pde, bc, 8, 0, solution=func, num_test=100, train_distribution="uniform")
```
<br>
layer_size four hidden layers with [5,5,5,1] neurons. <br>
The activation function is tanh. <br>
Use "Glorot uniform" to initialization parameters. <br>

```python
layer_size = [1] + [5] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
net.apply_output_transform(transform)
```
<br>
Learning rate = 1e-5, train for 10000 times. <br>

```python
model = dde.Model(data, net)
model.compile("adam", lr=0.00001, metrics=["l2 relative error"], external_trainable_variables=None)
model.train(iterations=10000)
```
<br>
Test points are linespace(0,1,1000), u represent the predicted values of the neural network and u_exact represent the ground truth values:<br>

```python
x = geom.uniform_points(1000, True)
u = model.predict(x)
u_exact = func(x)
```
<br>
gamma_pre represents the predicted eigenvalue (the nearest eigenvalue to gamma) of the neural network, and real value is pi**2:<br>

```python
def L(x,y):
    dy_xx = dde.grad.hessian(y, x)

    Lu_k = -dy_xx
    return Lu_k
Lu = model.predict(x,operator=L)
gamma_pre =(np.dot(Lu.T,u) / np.dot(u.T,u))[0,0]
print(f"The predicted eigenvalue is: {gamma_pre}")
print(f"The real eigenvalue is: {np.pi**2}")
print(f"The relative error of eigenvalue is: {(gamma_pre-np.pi**2)/np.pi**2}")
```
<br>
Plot u, u_exacy and relative error:<br>

```python
plt.plot(x, np.sin(np.pi*x)/normalized_np(np.sin(np.pi*x)), label="ground_truth")
plt.plot(x, abs(u/normalized_np(u)), label="prediction")
plt.legend()
plt.show()

plt.plot(x, abs(u/normalized_np(u))-np.sin(np.pi*x)/normalized_np(np.sin(np.pi*x)), label="error")
plt.legend()
plt.show()
```
### Result
<br>

![Image] (https://github.com/nianyitou/DeepXDE-Solving-Eigenvalue-Problems-based-on-Power-Method-/blob/main/1_D_Harmonic_Eigenvalue_Problem_Figure_1.png)
<br>

![Image] (https://github.com/nianyitou/DeepXDE-Solving-Eigenvalue-Problems-based-on-Power-Method-/blob/main/1_D_Harmonic_Eigenvalue_Problem_Figure_2.png)
<br>

![Image] (https://github.com/nianyitou/DeepXDE-Solving-Eigenvalue-Problems-based-on-Power-Method-/blob/main/1_D_Harmonic_Eigenvalue_Problem_Figure_3.png)
<br>


