# @Time : 2022-02-22 11:11
# @Author : Phalange
# @File : 3_2_9.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import random
import torch
from d2l import torch as d2l

#1.如果我们将权重初始化为零，会发生什么。算法仍然有效吗？

# answer：会报错：TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'
# 算法自然无效了,不过我报错的主要原因应该是torck.zeros(size=(2,1),requires_grad=True)没选择true


#2.假设你是乔治·西蒙·欧姆，试图为电压和电流的关系建立一个模型。你能使用自动微分来学习模型的参数吗?


#3.您能基于普朗克定律使用光谱能量密度来确定物体的温度吗？

#4.如果你想计算二阶导数可能会遇到什么问题？你会如何解决这些问题？

#5.为什么在squared_loss函数中需要使用reshape函数？

#6.尝试使用不同的学习率，观察损失函数值下降的快慢。

# answer:学习率过大，将导致发散，过小，将导致loss不下降或者说下降极其缓慢，在适当的范围内调节lr，lr大则下降快，小则下降慢。



#7.如果样本个数不能被批量大小整除，data_iter函数的行为会有什么变化？

# answer:假设样本数目为1008，批量大小为10，则最后一批数据将只有8个数据，indices[i: min(i + batch_size, num_examples)]因为这个函数
