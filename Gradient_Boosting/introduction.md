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

```python
class GB_PINNs():
    def __init__(self, geom, pde, bc, domain_num, boundary_num, net_list, scale_list, lr_list=None, solution=None, num_test=100):
        self.geom = geom
        self.pde = pde
        self.bc = bc
        self.domain_num = domain_num
        self.boundary_num = boundary_num
        self.lr_list = lr_list
        self.solution = solution
        self.num_test = num_test
        self.net_list = net_list
        self.scale_list = scale_list
```

lr_list: list of learning rate for each net in net_list. <br>
net_list: list of net (u = net_list[0] (x)/scale_list[0] + net_list[1] (x)/scale_list[1] + ...) <br>
scale_list: scale factors for nets in net_list. <br>

```python
    def compile(self, external_trainable_variables=None):
        for i in range(self.net_num):
            data = PDE(self.geom, self.pde, self.bc, self.domain_num,
                       self.boundary_num, solution=self.solution,
                       num_test=self.num_test, net_list = self.net_list,
                       scale_list = self.scale_list, current_index = i)
            self.data_list.append(data)
            model = dde.Model(self.data_list[i], self.net_list[i])
            self.model_list.append(model)
            model.compile("adam", lr=self.lr_list[i], metrics=["l2 relative error"],external_trainable_variables=external_trainable_variables)
```

GB_PINNs.compile(): Build model for each net and compile them (use deepxde.Model.compile), and store each model in list model_list. <br>

```python
    def train(self, train_step_list):
        if len(train_step_list)==1:
            for i in range(self.net_num):
                self.model_list[i].train(iterations=train_step_list[0])
        else:
            for i in range(self.net_num):
                self.model_list[i].train(iterations=train_step_list[i])
```

GB_PINNs.train(): Train each net in turn (use deepxde.Model.train()). <br>
train_step_list: list of training step for each net. <br>

```python
    def predict(self,x,operator=None):
        output = self.model_list[0].predict(x,operator=operator)/self.scale_list[0]
        for i in range(1,self.net_num):
            output += self.model_list[i].predict(x,operator=operator)/self.scale_list[i]
        return output
```

GB_PINNs.predict(): Output for all net(u = net_list[0] (x)/scale_list[0] + net_list[1] (x)/scale_list[1] + ...). <br>

## 3. Gradient_Boosting_1_D_Harmonic_Eigenvalue_Problem.py:

Use "gradient boosting" to enhance the performance of physics informed neural networks. <br>

```python
geom = dde.geometry.Interval(0, 1)
bc = []
layer_size = [1] + [10] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net_list = []
for i in range(2):
    net = dde.nn.FNN(layer_size, activation, initializer)
    net.apply_output_transform(transform)
    net_list.append(net)
GBP = GB_PINNs(geom, pde, bc, 32, 0, net_list, [1,1e3], lr_list=[1e-4]*len(net_list), solution=func, num_test=100)

GBP.compile()
GBP.train(train_step_list=[25000]*len(net_list))

x = geom.uniform_points(1000, True)
u = GBP.predict(x)
```

Use two net to represent the solution, each training 25000 times, other settings are the same as 1_D_Harmonic_Eigenvalue_Problem.py. <br>

### Result

![](/image/Gradient_Boosting_1_D_Harmonic_Eigenvalue_Problem_Figure_1.png)

![](/image/Gradient_Boosting_1_D_Harmonic_Eigenvalue_Problem_Figure_2.png)

![](/image/Gradient_Boosting_1_D_Harmonic_Eigenvalue_Problem_Figure_3.png)

![](/image/Gradient_Boosting_1_D_Harmonic_Eigenvalue_Problem_Figure_4.png)

Compared with 1_D_Harmonic_Eigenvalue_Problem.py, the accuracy improved by two orders of magnitude.
