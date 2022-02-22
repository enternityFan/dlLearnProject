# @Time : 2022-02-22 13:45
# @Author : Phalange
# @File : 3_6_9.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import torch

# 1.在本节中，我们直接实现了基于数学定义softmax运算的softmax函数。这可能会导致什么问题？提示：尝试计算 exp(50) 的大小。
print(torch.exp(torch.tensor(50)))

# answer: 当不合理的参数设置和较大噪声的影响，会导致输出结果可能非常大造成溢出，

# 2.本节中的函数cross_entropy是根据交叉熵损失函数的定义实现的。它可能有什么问题？提示：考虑对数的定义域。

# answer: y_hat中若某行最大的值也接近0的话，loss的值会超过long类型范围。

# 3.你可以想到什么解决方案来解决上述两个问题？

# answer:1.设置阈值让loss总处于一个范围之内。


# 4.返回概率最大的标签总是一个好主意吗？例如，医疗诊断场景下你会这样做吗？

# answer：不总是一个好主意，医疗诊断场景下不会这么做，因为假阳性错误判断太高了。。


# 5.假设我们希望使用softmax回归来基于某些特征预测下一个单词。词汇量大可能会带来哪些问题?

# answer:1.计算成本太高，2.每个单词的概率都大致相同，并且为很小的数值。。