# @Time: 19:13
# @Author:Phalange
# @File:5_4Script.py
# @Software:PyCharm
# C’est la vie,enjoy it! :D

import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super(self).__init__()

    def forward(self,X):
        return X - X.mean()

class MyLinear(nn.Module):
    def __init__(self,in_units,units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units,units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self,X):
        linear = torch.matmul(X,self.weight.data) + self.bias.data
        return F.relu(linear)



layer = CenteredLayer()
layer(torch.FloatTensor([1,2,3,4,5]))
net = nn.Sequential(nn.Linear(8,128),CenteredLayer())
Y = net(torch.rand(4,8))
Y.mean()