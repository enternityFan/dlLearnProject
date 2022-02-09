# @Time: 19:24
# @Author:Phalange
# @File:3_2Script.py
# @Software:PyCharm
# C’est la vie,enjoy it! :D

import random
from mxnet import autograd,np,npx
from d2l import mxnet as d2l

npx.set_np()


def synthetic_data(w,b,num_examples): #@save
    """生成y=Xw+b+噪声"""
    X = np.random.normal(0,1,(num_examples,len(w)))

    y = np.dot(X,w) + b
    y += np.random.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))

def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = np.array(
            indices[i:min(i + batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices] # yeild直接由运行状态跳回就绪状态，然CPU重新调度。

def linreg(X,w,b):#@save
    """"线性回归模型"""
    return np.dot(X,w) + b

def squared_loss(y_hat,y):#@save
    """均方损失"""
    return(y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params,lr,batch_size):#@save
    """小批量随机梯度下降"""
    for param in params:
        param[:] = param - lr * param.grad / batch_size


true_w = np.array([2,-3.4])
true_b = 4.2
features,labels = synthetic_data(true_w,true_b,1000)
print('features:',features[0],'\nlabel:',labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:,(1)].asnumpy(),labels.asnumpy(),1)
d2l.plt.show()

batch_size = 10
for X,y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break

w = np.random.normal(0,0.01,(2,1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        with autograd.record():
            l = loss(net(X,w,b),y) # X和y的小批量损失

        # 计算l关于[w,b]的梯度
        l.backward()
        sgd([w,b],lr,batch_size)
    train_l = loss(net(features,w,b),labels)
    print(f'epoch {epoch + 1},loss {float(train_l.mean()):f}')