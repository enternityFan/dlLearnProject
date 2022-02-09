# @Time: 18:19
# @Author:Phalange
# @File:3_1Script.py
# @Software:PyCharm
# C’est la vie,enjoy it! :D

import math
import time
from mxnet import np
from d2l import mxnet as d2l

def normal(x,mu,sigma):
    p = 1/math.sqrt(2 * math.pi * sigma ** 2)
    return p*np.exp(-0.5/sigma**2 * (x - mu)**2)



class Timer: #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动定时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()



n = 10000
a = np.ones(n)
b = np.ones(n)
c = np.zeros(n)
timer = Timer()
timer.start()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.5f} sec')
timer.start()
d = a+b
print(f'{timer.stop():.5f} sec')

# 再次使用numpy进行可视化
x = np.arange(-7,7,0.01)
params = [(0,1),(0,2),(3,1)]
d2l.plot(x,[normal(x,mu,sigma) for mu,sigma in params],xlabel='x',ylabel='p(x)',figsize=(4.5,2.5),
         legend=[f'mean {mu},std{sigma}' for mu,sigma in params])
d2l.plt.show()