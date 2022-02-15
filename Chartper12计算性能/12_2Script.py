# @Time : 2022-02-15 16:48
# @Author : Phalange
# @File : 12_2Script.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D



import os
import subprocess
import numpy
import torch
from torch import nn
from d2l import torch as d2l



# GPU计算热身
device = d2l.try_gpu()
a = torch.randn(size=(1000,1000),device=device)
b = torch.mm(a,a)

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000,1000))
        b = numpy.dot(a,a)


with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000,1000),device=device)
        b = torch.mm(a,a)

with d2l.Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)

x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2
z