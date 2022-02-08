# @Time: 17:40
# @Author:Phalange
# @File:2_5possibility.py
# @Software:PyCharm
# C’est la vie,enjoy it! :D

import random
from mxnet import np,npx
from d2l import mxnet as d2l

npx.set_np()

fair_probs = [1.0 / 6] * 6
print(np.random.multinomial(10,fair_probs)) # 1为试验次数，pvals参数为P个不同结果的概率

counts = np.random.multinomial(1000,fair_probs).astype(np.float32)
print(counts / 1000)

# 500组实验，每组抽10个样本
counts = np.random.multinomial(50,fair_probs,size=700)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1,keepdims = True)
d2l.set_figsize((6,4.5))
for i in range(6):
    d2l.plt.plot(estimates[:,i].asnumpy(),label=("P(die=" + str(i+1)+")"))

d2l.plt.axhline(y=0.167,color='black',linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()