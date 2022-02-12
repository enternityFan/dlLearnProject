# @Time:2022-02-12 15:57
# @Author:Phalange
# @File:9_4Script.py
# @Software:PyCharm
# C’est la vie,enjoy it! :D
import torch
from torch import nn
from d2l import torch as d2l

# 加载数据
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# 通过设置"bidirective=True" 来定义LSTM模型
vocab_size,num_hiddens,num_layers = len(vocab),256,2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs,num_hiddens,num_layers,bidirectional=True)
model = d2l.RNNModel(lstm_layer,len(vocab))
model = model.to(device)
# 训练模型
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)