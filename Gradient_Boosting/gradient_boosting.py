import deepxde as dde
from pde_gb import PDE

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
        self.net_num = len(self.net_list)
        self.data_list = []
        self.model_list = []

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

    def train(self, train_step_list):
        if len(train_step_list)==1:
            for i in range(self.net_num):
                self.model_list[i].train(iterations=train_step_list[0])
        else:
            for i in range(self.net_num):
                self.model_list[i].train(iterations=train_step_list[i])


    def predict(self,x,operator=None):
        output = self.model_list[0].predict(x,operator=operator)/self.scale_list[0]
        for i in range(1,self.net_num):
            output += self.model_list[i].predict(x,operator=operator)/self.scale_list[i]
        return output