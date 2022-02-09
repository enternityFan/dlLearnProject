# @Time: 19:49
# @Author:Phalange
# @File:3_3Script.py
# @Software:PyCharm
# C’est la vie,enjoy it! :D

from mxnet import autograd,gluon,np,npx
from d2l import mxnet as d2l
from mxnet.gluon import nn
from mxnet import init
npx.set_np()

true_w = np.array([2,-3.4])
true_b = 4.2
features,labels = d2l.synthetic_data(true_w,true_b,1000)

def load_array(data_arrays,batch_size,is_train=True):#@save
    """构造一个Gluon数据迭代器"""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset,batch_size,shuffle=is_train)

batch_size = 10
data_iter = load_array((features,labels),batch_size)
next(iter(data_iter))

net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})
num_epochs = 3
for epoch in range(num_epochs):
    for X,y in data_iter:
        with autograd.record():
            l = loss(net(X),y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features),labels)
    print(f'epoch {epoch + 1},loss {l.mean().asnumpy():f}')

w = net[0].weight.data()
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
b = net[0].bias.data()
print(f'b的估计误差:{true_b - b}')
