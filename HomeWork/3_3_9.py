# @Time : 2022-02-22 13:20
# @Author : Phalange
# @File : 3_3_9.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D


# 1.如果我们用nn.MSELoss(reduction=‘sum’)替换 ，nn.MSELoss（）”为了使代码的行为相同，需要怎么更改学习速率？为什么？

# answer: lr = original_lr / batch_size，因为本来的是默认的损失的平均值，现在换成了损失的和，也就扩大了batch_size倍，除了就行


# 查看深度学习框架文档，它们提供了哪些损失函数和初始化方法？用Huber损失代替原损失。

# answer:有好多好多loss。。BCELoss、CTCLLoss、等等吧，直接搜torch的文档吧。  用huber损失替换，也是很简单的，不过我torch1.8的，没法实现，需要1.9才可以。

