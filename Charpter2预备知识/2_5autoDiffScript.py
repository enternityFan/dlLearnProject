# @Time: 17:22
# @Author:Phalange
# @File:2_5autoDiffScript.py
# @Software:PyCharm
# C’est la vie,enjoy it! :D
from mxnet import autograd,np,npx

npx.set_np()

x = np.arange(4.0)
x.attach_grad() # 为该张量的梯度分配内存

with autograd.record():
    y = 2 * np.dot(x,x)

y.backward()
print(x.grad)
print(x.grad == 4 * x)
print(x.grad)
with autograd.record():
    y = x.sum()
y.backward()
print(x.grad)
y.backward()


# 分离计算
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
