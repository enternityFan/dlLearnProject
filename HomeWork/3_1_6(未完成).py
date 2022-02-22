# @Time : 2022-02-22 11:00
# @Author : Phalange
# @File : 3_1_6(未完成).py
# @Software: PyCharm
# C'est la vie,enjoy it! :D


import math
import time
import numpy as np
import torch
from d2l import torch as d2l


# 1.假设我们有一些数据 x1,…,xn∈R 。我们的目标是找到一个常数 b ，使得最小化 ∑i(xi−b)2 。
#
# 1）找到最优值 b 的解析解。
#
# 2)这个问题及其解与正态分布有什么关系?


"""
首先写出： y = (x1 - b) **2 + (x2 - b) **2 + ... ( xn - b) ** 2
         为了求最小的y，可以求y对b求偏导：
         (x1-b) + (x2 - b) + ...(xn - b) =0  # 化简后
         则： x1 + x2 + .. + xn = n * b
         则 b为x1 + x2 + ... + xn的平均值

2)   某个大佬通过P(xi|b)求最大似然估计的方法转换为题干问题。。牛
"""

