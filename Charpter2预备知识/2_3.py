# @Time: 16:01
# @Author:Phalange
# @File:2_3.py
# @Software:PyCharm
# C’est la vie,enjoy it! :D

import torch
import numpy as np
x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(len(np.arange(4)))
print(x.shape)
A = np.arange(20).reshape(5,4)
print(A)
print(A.T)
A_sum_axis0 = A.sum(axis=0)
A_mean_axis0 = A.mean(axis=0)
sum_A = A.sum(axis=1,keepdims=True) # 保持矩阵二维特性
cum_A = A.cumsum(axis=0) # 沿某行的累计求和
print(sum_A)
print(A_sum_axis0)

x = np.arange(4)
y = np.ones(4)
print(np.dot(x,y))
print(np.dot(A,x))

B = np.ones(shape=(4,3))
print(np.dot(A,B))

u = np.array([3,-4])
np.linalg.norm(u) # L2范数
np.abs(u).sum() # L1范数
