# @Time: 19:08
# @Author:Phalange
# @File:5_3Script.py
# @Software:PyCharm
# C’est la vie,enjoy it! :D

from mxnet import np, npx
from mxnet.gluon import nn

npx.set_np()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    return net

net = get_net()

print(net.collect_params)
print(net.collect_params())

net.initialize()
print(net.collect_params())

X = np.random.uniform(size=(2,20))
net(X)
print(net.collect_params())