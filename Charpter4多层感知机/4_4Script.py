# @Time: 11:56
# @Author:Phalange
# @File:4_4Script.py
# @Software:PyCharm
# C’est la vie,enjoy it! :D

import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

def evaluate_loss(net,data_iter,loss):#@save
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2) # 损失的总和，样本数量
    for X,y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out,y)
        metric.add(l.sum(),l.numel()) # 这一行现在我不是很理解
    return metric[0] / metric[1]

def train(train_features,test_features,train_labels,test_labels,num_epoches=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们在多项式中实现了他
    net = nn.Sequential(nn.Linear(input_shape,1,bias=False))
    batch_size = min(10,train_labels.shape[0])
    train_iter = d2l.load_array((train_features,train_labels.reshape(-1,1)),batch_size)
    test_iter = d2l.load_array((test_features,test_labels.reshape(-1,1)),batch_size,is_train=False)
    trainer = torch.optim.SGD(net.parameters(),lr=0.01)
    animator = d2l.Animator(xlabel='epoch',ylabel='loss',yscale='log',xlim=[1,num_epoches],ylim=[1e-3,1e2],
                           legend=['train','test'])
    for epoch in range(num_epoches):
        d2l.train_epoch_ch3(net,train_iter,loss,trainer)
        if epoch == 0 or (epoch + 1 ) % 20 ==0:
            animator.add(epoch + 1,(evaluate_loss(net,train_iter,loss),
                                    evaluate_loss(net,test_iter,loss)))
    print('weight:',net[0].weight.data.numpy())



max_degree = 20
n_train,n_test = 100,100 # 训练集和测试机大小
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5,1.2,-3.4,5.6])

features = np.random.normal(size=(n_train + n_test,1))
np.random.shuffle(features)
poly_features = np.power(features,np.arange(max_degree).reshape(1,-1))
for i in range(max_degree):
    poly_features[:,i] /=math.gamma(i+1) # mamma(n) = (n-1)!
# Labels的维度：(n_train+n_test,)
labels = np.dot(poly_features,true_w)
labels += np.random.normal(scale=0.1,size=labels.shape)

# Numpy to tensor
true_w,features,poly_features,labels = [torch.tensor(x,dtype=torch.float32) for x in [true_w,features,poly_features,labels]]

features[:2],poly_features[:2,:],labels[:2]

train(poly_features[:n_train,:2],poly_features[n_train:,:2],
      labels[:n_train],labels[n_train:])
