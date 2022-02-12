# @Time:2022-02-12 15:20
# @Author:Phalange
# @File:9_3Script.py
# @Software:PyCharm
# C’est la vie,enjoy it! :D

import torch
from torch import nn
from d2l import torch as d2l

batch_size,num_steps = 32,35
train_iter,vocab = d2l.load_data_time_machine(batch_size,num_steps)

vocab_size,num_hiddens,num_layers = len(vocab),256,2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs,num_hiddens,num_layers)
model = d2l.RNNModel(lstm_layer,len(vocab))
model = model.to(device)


num_epochs,lr = 500,2
d2l.train_ch8(model,train_iter,vocab,lr,num_epochs,device)
