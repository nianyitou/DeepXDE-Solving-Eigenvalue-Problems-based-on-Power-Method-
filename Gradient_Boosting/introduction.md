# Use "gradient boosting" to enhance the performance of physics informed neural networks.

## 1. pde_gb.py:

Slightly modified version of deepxde.data.pde.py, the changes are shown below: <br>

```python
self.train_distribution = "uniform"
```

```python
def __init__(
        ...
        net_list = None,
        scale_list = None,
        current_index = 0
    ):
```
net_list：list of net. <br>
scale_list: scale factors of each net. <br>
current_index: which net in the list is training.

```python
        self.net_list = net_list
        self.scale_list = scale_list
        self.current_index = current_index
```

```python
def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
    if backend_name in ["tensorflow.compat.v1", "tensorflow", "pytorch", "paddle"]:
        outputs_pde = outputs/self.scale_list[self.current_index]
        for i in range(self.current_index):
            outputs_pde += self.net_list[i](inputs)/self.scale_list[i]
    elif backend_name == "jax":
        # JAX requires pure functions
        outputs_pde = (outputs, aux[0])

    f = []
    if self.pde is not None:
        if get_num_args(self.pde) == 2:
            f = self.pde(inputs, outputs_pde)*self.scale_list[self.current_index]
        elif get_num_args(self.pde) == 3:
            f = self.pde(inputs, outputs_pde, aux) * self.scale_list[self.current_index]
```

## 2. gradient_boosting.py:
Corresponding version of deepxde.model.py for gradient boosting, including: <br>

