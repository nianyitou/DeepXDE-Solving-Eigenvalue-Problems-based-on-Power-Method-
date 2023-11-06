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
PDE function for deepxde:<br>

```python
def pde(x, y):
    u_kold = y.clone().detach()
    dy_xx = dde.grad.hessian(y, x)
    Lu_k=-dy_xx - gamma*y

    return Lu_k/normalized(Lu_k) - u_kold/normalized(u_kold)
```
