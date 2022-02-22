# @Time : 2022-02-22 9:45
# @Author : Phalange
# @File : 2_3_13.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D
import torch

# 2.4
a = torch.arange(2*3*4).resize(2,3,4)
#print(a)
print(len(a))
b = torch.arange(9*3*2*1*1).resize(9,3,2,1,1)
print(len(b))

#2.5对于任意形状的张量X,len(X)是否总是对应于X特定轴的长度?这个轴是什么?

# answer: 根据上面a,b实验发现，len(x)返回第0维的轴的长度

# 2.6运行A/A.sum(axis=1)，看看会发生什么。你能分析原因吗？

A = torch.arange(3*3).resize(3,3)
print("A : "+str(A))
print("A.sum(axis=1):" + str(A.sum(axis=1)))
print(A/A.sum(axis=1))

# answer: 根据实验结果，A的每一列都除以A.sum(axis=1)的每一列,原因的话就是python的广播机制了。

# 2.7 考虑一个具有形状 (2,3,4) 的张量，在轴0、1、2上的求和输出是什么形状?
c = torch.arange(2*3*4).resize(2,3,4)
print("在0轴上的求和输出:"+ str(c.sum(axis=0))+"shape为:" +str(c.sum(axis=0).shape))
print("在1轴上的求和输出:"+ str(c.sum(axis=1))+"shape为:" +str(c.sum(axis=1).shape))
print("在2轴上的求和输出:"+ str(c.sum(axis=2))+"shape为:" +str(c.sum(axis=2).shape))

# 实验结果可以发现，在哪个轴上进行求和，那那一轴将消失

# 2.8 为linalg.norm函数提供3个或更多轴的张量，并观察其输出。对于任意形状的张量这个函数计算得到什么?

D = torch.randn(2,3,4)
print(torch.linalg.norm(D))
E = torch.ones(5,3,4,2,4)
print(torch.linalg.norm(E))
# 对于任意性状的张量，这个函数计算得到的是一个数值。
