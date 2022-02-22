# @Time : 2022-02-22 9:03
# @Author : Phalange
# @File : jiebaScript.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import jieba

seg_str = "好好学习，天天向上。"

print("/".join(jieba.lcut(seg_str))) # 精简模式，返回一个列表类型的结果
print("/".join(jieba.lcut(seg_str,cut_all=True))) # 全模式
print("/".join(jieba.lcut_for_search(seg_str))) # 搜索引擎模式


# 对三国演义进行分词
txt = []
with open("threskingdoms.txt","r",encoding='utf-8') as f:
    txt = f.read()
words = jieba.lcut(txt)
counts={}
for word in words:
    if len(word) == 1: # 单个词语不计算在内
        continue
    else:
        counts[word] = counts.get(word,0) +1 # get(key,default) 当key不存在值时返回default的值,这里是0

items = list(counts.items())
items.sort(key=lambda x:x[1],reverse=True) # 从大到小进行排序

for i in range(10):
    word,count = items[i]
    print("{0:<5}{1:>5}".format(word,count))


# 对哈姆雷特进行分词
txt = []
with open("hamlet.txt","r",encoding='utf-8') as f:
    txt = f.read()

# 预处理
for ch in '!"#$%&()*+,-./:;<=>?@[\\]^_‘{|}~':
    txt = txt.replace(ch, " ")      # 将文本中特殊字符替换为空格
words = txt.split()
counts={}
for word in words:
    if len(word) == 1: # 单个词语不计算在内
        continue
    else:
        counts[word] = counts.get(word,0) +1 # get(key,default) 当key不存在值时返回default的值,这里是0

items = list(counts.items())
items.sort(key=lambda x:x[1],reverse=True) # 从大到小进行排序

for i in range(10):
    word,count = items[i]
    print("{0:<5}{1:>5}".format(word,count))
