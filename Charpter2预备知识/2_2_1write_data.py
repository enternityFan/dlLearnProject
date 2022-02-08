import os
import pandas as pd
import torch
if False:
    os.mkdir(os.path.join('..','data'))
    data_file = os.path.join('..','data','house_tiny.csv')
    with open(data_file,'w') as f:
        f.write('NumRooms,Alley,Price\n')  # 列名
        f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')


data_file = os.path.join('..','data','house_tiny.csv')
data = pd.read_csv(data_file)
print(data)


print(data)

'''
处理缺失值
'''
inputs,outputs = data.iloc[:,0:2],data.iloc[:,2] # iloc为位置索引o

inputs = inputs.fillna(inputs.mean()) # 填充缺失值使用同一列的均值
print(inputs)
inputs = pd.get_dummies(inputs,dummy_na=True) # Convert categorical variable into dummy/indicator variables.
print(inputs)

X,y = torch.tensor(inputs.values),torch.tensor(outputs.values)
print(X)
print(y)

'''
删除最多nan的列
'''
def deleteMaxNa(data):
    num_na = data.isna().sum()
    num_dict = dict(num_na)
    return data.drop(max(num_dict, key=num_dict.get), axis=1)
