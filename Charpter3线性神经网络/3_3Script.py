# @Time: 19:49
# @Author:Phalange
# @File:3_3Script.py
# @Software:PyCharm
# C’est la vie,enjoy it! :D

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
# nn是神经网络的缩写
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 10000)



def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)



net = nn.Sequential(nn.Linear(2, 1))
#loss = nn.MSELoss(reduction='sum')
loss = torch.nn.HuberLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.003)
num_epochs = 10

for epoch in range(num_epochs):
    #sum_l = 0.0
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        #sum_l +=l

    l = loss(net(features), labels)
    #sum_l /= (len(features) / batch_size)
    #print(f'epoch {epoch + 1}, loss {sum_l:f}')
    print(f'epoch {epoch + 1}, loss {l:f}')


w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)