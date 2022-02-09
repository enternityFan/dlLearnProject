# @Time: 16:54
# @Author:Phalange
# @File:5_1Script.py
# @Software:PyCharm
# C’est la vie,enjoy it! :D

import torch
from torch import nn
from torch.nn import functional as F

#net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

#X = torch.rand(2, 20)
#net(X)

class MLP(nn.Module):
    # 用模型参数生命层。这里，我们生命两个全连接层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化
        # 这样，在类实例化时也可以指定其他函数参数，利于模型参数params
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self,X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义
        return self.out(F.relu(self.hidden(X)))

class MySequential(nn.Module):
    def __init__(self,*args):
        super(self).__init__()
        for idx,module in enumerate(args):
            # 这里，module是Module子类的一个实例，我们把它保存在'Module'类的成员
            # 变量 modules中。module的类型是OrderedDict
            self._modules[str(idx)] = module


    def forward(self,X):
        for block in self._modules.values():
            X = block(X)
        return X

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20,20)

    def forward(self,X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X,self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /=2
        return X.sum()

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,64),nn.ReLU(),
                                 nn.Linear(64,32),nn.ReLU())
        self.linear = nn.Linear(32,16)

    def forward(self,X):
        return self.linear(self.net(X))


X = torch.rand(2, 20)
#net = MLP()
#net = MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
#net = FixedHiddenMLP()
chimera = nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenMLP())
#print(net(X))
print(chimera(X))
#net(X)
