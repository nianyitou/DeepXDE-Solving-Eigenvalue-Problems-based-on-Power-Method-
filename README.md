# DeepXDE: Solving Eigenvalue Problems based on Power Method

## installation:

>installation package: pip install deepxde <br>
><br>
>Use (python -m deepxde.backend.set_default_backend pytorch) to set the default backend pytorch

## 1. Power_Method:

Solve 1D harmonic eigenvalue problem: <br>
u_xx + λ*u = 0 <br>
See Power_Method/introduction.md for details. <br>

## 2. multiple_eigenvalues:

Solve 1D harmonic eigenvalue problem: <br>
u_xx + λ*u = 0 <br>
Calculate multiple eigenvalues and corresponding eigenfunctions simultaneously.
See multiple_eigenvalues/introduction.md for details. <br>

## 3. Gradient_Boosting:

Use multiple neural networks to approximate the solution (u = net_1(x) + net_2(x) + ...), greatly improved accuracy. <br>
See Gradient_Boosting/introduction.md for details. <br>
