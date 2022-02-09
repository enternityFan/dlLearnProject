# @Time: 17:14
# @Author:Phalange
# @File:5_2Script.py
# @Software:PyCharm
# C’est la vie,enjoy it! :D

import torch
from torch import nn

net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
X = torch.rand(size=(2,4))
net(X)

print(net[2].state_dict())
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

print(*[(name,param.shape) for name,param in net[0].named_parameters()])
print(*[(name,param.shape) for name,param in net.named_parameters()])
print(net.state_dict()['2.bias'].data) # 一种访问网络参数的方式


def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),
                         nn.Linear(8,4),nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}',block1())
    return net

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,mean=0,std=0.01)
        nn.init.zeros_(m.bias)

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,1)
        nn.init.zeros_(m.bias)

def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,42)

def my_init(m):
    if type(m) == nn.Linear:
        print("Init",*[(name,param.shape)
                       for name,param in m.named_parameters()][0])
        nn.init.uniform_(m.weight,-10,10)
        m.weight.data *= m.weight.data.abs() >=5



rgnet = nn.Sequential(block2(),nn.Linear(4,1))
rgnet.apply(init_normal)

rgnet(X)
print(rgnet)
print(rgnet[0][0][0].weight.data[0],rgnet[0][0][0].bias.data[0])

'''
不同块应用不同的初始化方法
'''

net[0].apply(xavier)
net[2].apply(init_42)

net[0].weight.data[:] +=1
net[0].weight.data[0,0] = 42
net[0].weight.data[0]

shared = nn.Linear(8,8)
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),
                    shared,nn.ReLU(),
                    shared,nn.ReLU(),
                    nn.Linear(8,1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0,0] = 100
# 确保他们实际上是同一个对象，而不只是相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])

