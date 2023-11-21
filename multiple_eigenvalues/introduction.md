## 1. File: nn_multi_net.py:



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
