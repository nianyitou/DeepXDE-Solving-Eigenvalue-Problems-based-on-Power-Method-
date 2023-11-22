## 1. File: nn_multi_net.py:

A neural network that stores multiple eigenfunctions simultaneously.<br>

```python
    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        x_list = []
        for eig in range(self.eig_num):
            x_1 = x
            x_1 = self.linears[eig](x_1)
            x_list.append(x_1)

        if self._output_transform is not None:
            for eig in range(self.eig_num):
                x_list[eig] = self._output_transform(inputs, x_list[eig])

        return torch.cat(x_list,dim=1)
```
<br>

output = [u_1, u_2, ...] ([batch,num_eigs])<br>
or output[:,0] = The first eigenfunction, output[:,1] = The second eigenfunction,...<br>

## 2. File: multi_max_eigs:

Solve 1D harmonic eigenvalue problem: <br>
u_xx + 1000.0u = Lu = λ*u <br>
Calculate multiple eigenvalues (5 eigenvalues with the largest value for example) and corresponding eigenfunctions simultaneously.<br>

```python
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
```
<br>
y = [eigvec_0, eigvec_1, eigvec_2,...]<br>
Lu_k = [L(eigvec_0), L(eigvec_1), L(eigvec_2),...]<br>
Schmidt orthogonalization of Lu_k:<br>
Lu_k[:,0] = L(eigvec_0)/norm(L(eigvec_0))<br>
Lu_k[:,1] = (L(eigvec_1) - dot(L(eigvec_0)/norm(L(eigvec_0)),L(eigvec_1))*L(eigvec_0)/norm(L(eigvec_0)))/norm(L(eigvec_1) - dot(L(eigvec_0)/norm(L(eigvec_0)),L(eigvec_1))*L(eigvec_0)/norm(L(eigvec_0)))<br>
Lu_k[:,2] = ...<br>
.<br>
.<br>
.<br>
loss = tr.abs(Lu_k[:, 0] - y[:,0]) +  tr.abs(Lu_k[:, 1] - y[:,1]) + ...<br>



### Result

![](/image/multiple_eigenvalues_multi_max_eigs_Figure_1.png)

![](/image/multiple_eigenvalues_multi_max_eigs_Figure_2.png)


## 3. File: multi_min_eigs:
