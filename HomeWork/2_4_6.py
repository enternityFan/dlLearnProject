# @Time : 2022-02-22 10:11
# @Author : Phalange
# @File : 2_4_6.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D



import numpy as np
from IPython import display
from d2l import torch as d2l
import math

def f(x):
    return x ** 3 - 1/x

# 1.绘制函数 y=f(x)=x^3−1/x 和其在 x=1 处切线的图像
x = np.arange(0,5,0.1)
d2l.plot(x,[f(x),4 * x-4],'x','f(x)',legend=['f(x)','Tangent line (x=1)'])
d2l.plt.show()

# 2. 求函数f(x) = 3*x1^2 + 5*e(x2) 的梯度
def f2(x1,x2):
    return 3*x1 ** 2 + 5 * math.exp(x2)

# answer:梯度：[6*x1,5*exp(x2)]

# 3.函数f(x) = ||x||2 的梯度是什么

# answer: [x1/sqrt(x1^2+x2^2+...xn^2),x2/sqrt(x1^2+x2^2+...xn^2),...,xn/sqrt(x1^2+x2^2+...xn^2)]

# 4.你可以写出函数 u=f(x,y,z) ，其中 x=x(a,b) ， y=y(a,b) ， z=z(a,b) 的链式法则吗?

# answer:[du/dx * dx/da + du/dy * dy/da + du/dz * dz/da,du/dx * dx/db + du/dy * dy/db + du/dz * dz/db]
