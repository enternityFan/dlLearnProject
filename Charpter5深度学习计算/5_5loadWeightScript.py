# @Time: 19:33
# @Author:Phalange
# @File:5_5loadWeightScript.py
# @Software:PyCharm
# C’est la vie,enjoy it! :D

import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

X = torch.randn(size=(2, 20))
net = MLP()
Y = net(X)
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())
Y_clone = clone(X)
print(Y_clone == Y)