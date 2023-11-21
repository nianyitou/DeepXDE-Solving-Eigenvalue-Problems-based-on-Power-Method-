import numpy as np
import torch
from deepxde.nn.pytorch.fnn import FNN
from deepxde.nn.pytorch import NN
from deepxde.nn import activations
from deepxde.nn import initializers
from deepxde import config

class Multi_FNN(NN):

    def __init__(self, eig_num, layer_sizes, activation, kernel_initializer):
        super().__init__()
        self.eig_num = eig_num

        self.linears = torch.nn.ModuleList()
        for _ in range(self.eig_num):
            self.linears.append(single_block(layer_sizes, activation, kernel_initializer))


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


class single_block(NN):
    def __init__(self, layer_sizes, activation, kernel_initializer):
        super().__init__()
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        self.linear = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linear.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)
                )
            )
            initializer(self.linear[-1].weight)
            initializer_zero(self.linear[-1].bias)


    def forward(self, x):
        for j, linear in enumerate(self.linear[:-1]):
            x = (
                self.activation[j](linear(x))
                if isinstance(self.activation, list)
                else self.activation(linear(x))
            )
        x = self.linear[-1](x)
        return x