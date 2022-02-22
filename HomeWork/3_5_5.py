# @Time : 2022-02-22 13:37
# @Author : Phalange
# @File : 3_5_5.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D


# 1.减少batch_size（如减少到1）是否会影响读取性能？

# answer: 当batch_size为256时，读取的时间为4.09s,当设置为1时，时间需要为25.48s，也就是差不多提高了256倍吧。

# 2.数据迭代器的性能非常重要。你认为当前的实现足够快吗？探索各种选择来改进它。

# answer: 提高进程数目，提高batch_size


# 3.查阅框架的在线API文档。还有哪些其他数据集可用？

# answer：torchvision.datasets中还有COCO、CIFAR、FakeData、LFW。。。。很多很多。。