import torch
import numpy


x = torch.arange(12) # 1..12
print(x)
print(x.shape) # 返回张良形状
print(x.numel()) # 返回张量中数据总数

X = x.reshape(3,4) # x.reshape(3,-1)
print(X)
zero = torch.zeros((2,3,4))
print(zero)
one = torch.ones((2,3,4))
print(one)
torch.randn(3,4) # 标准高斯分布中的随机采样

a = [[2,1,4,3],[1,2,3,4],[4,3,2,1]]
torch.tensor(a) # list to tensor
x = torch.tensor([1.0,2,4,8])
y = torch.tensor([2,2,2,2])
x +y,x - y,x * y,x / y,x ** y
torch.exp(x)

# 张量堆叠
X = torch.arange(12,dtype=torch.float32).reshape((3,4))
Y = torch.tensor(a)
torch.cat((X,Y),dim=0) # 按行
torch.cat((X,Y),dim=1) # 按列

'''
广播机制
'''
print("\n\n\n\n\n")
a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))
print(a,b)
print(a+b)

'''
索引和切片
'''
X[-1],X[1:3]

'''
节省内存
'''
Z = torch.zeros_like(Y)
print('id(Z):',id(Z))
Z[:] = X+Y
print('id(Z):',id(Z))

'''
转换为其他python对象

'''
A = X.numpy()
B = torch.tensor(A)
print(type(A),type(B))

'''
2.1.8练习题
'''

# 1
print(X < Y)
print(X > Y)

# 2
a = torch.arange(4).reshape((2,2))
b = torch.arange(2).reshape((1,2))
print(a,b)
print(a+b)