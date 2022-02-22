# @Time : 2022-02-22 10:25
# @Author : Phalange
# @File : 2_5_6.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import torch
import math
from d2l import torch as d2l
#1.为什么计算二阶导数比一阶导数的开销要更大？

# answer:废话了。。计算二阶导需要在一阶导的结果基础上进行求导。。开销肯定打了


#2.在运行反向传播函数之后，立即再次运行它，看看会发生什么。
x = torch.arange(4.0)
x.requires_grad_(True)
y = 2 * torch.dot(x,x)
print(y.backward())
#print(y.backward())

# answer:直接运行的话会有一个报错：RuntimeError: Trying to backward through the graph a second time, but the saved intermediate results have already been freed.


#3.在控制流的例子中，我们计算d关于a的导数，如果我们将变量a更改为随机向量或矩阵，会发生什么？
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(1,3),requires_grad=True)
d = f(a)
#d.backward()

# answer:  当反向传播时：RuntimeError: grad can be implicitly created only for scalar outputs
# 上面错误的原因：当输出不是标量时，调用.backward()就会出错，解决方法：显示声明输出的类型作为参数传入,且参数的大小必须要和输出值的大小相同

# 等价于d.backward(torch.ones(len(x)))
d.sum().backward() #梯度只能为标量（即一个数）输出隐式地创建
# answer: 这种形式可以工作



#4.重新设计一个求控制流梯度的例子，运行并分析结果。
def f(a):
    b = a * 2 + abs(a)
    c = b*3 - b ** (-1/4)
    return c

a = torch.randn(size=(3,1),requires_grad=True)
print(a.shape)
print(a)

e = f(a)
e.sum().backward()
print(a.grad)

# 5.使f(x)=sin(x)，绘制f(x)和df(x)/dx的图像，其中后者不使用f′(x)=cos(x)
xx = torch.range(-3,3,0.1,requires_grad=True)
y = torch.sin(xx)
y.sum().backward()
d2l.plot(xx.detach(),[y.detach(),xx.grad],xlabel='x',ylabel='y',legend=['y','y_grad'])
d2l.plt.show()
