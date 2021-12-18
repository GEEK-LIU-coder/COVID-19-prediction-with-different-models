import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# 约翰霍普金斯大学美国疫情数据2020-01-21——2021-07-01
#一共528行数据，这里设置每20个数据为一个间隔
#读入csv文件

n='us_virus.csv'
data=pd.read_csv(n)


#把日期，确诊人数存入列表
date_list=list(data['date'])
confirm_list=list(data['cases'])

#处理为单日新增(迭代)
for i in range(len(confirm_list)-1,0):
    confirm_list[i]=confirm_list[i]-confirm_list[i-1]

print(confirm_list)


#设置训练集和验证集的数据
test_data_size = 20
train_data = confirm_list[:-test_data_size]
test_data = confirm_list[-test_data_size:]
print(len(train_data))
train_data=np.array(train_data)
test_data=np.array(test_data)

#训练集做归一化处理，减小误差
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
train_window = 20

#通过函数，得到一个’输入输出列表‘，每个元素是(长度为20)训练序列和训练标签(21)的元组
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

#定义LSTM类，继承nn的module模块
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=45, output_size=1):
        super().__init__()
        #*****(传参)隐藏神经元/节点的的个数
        self.hidden_layer_size = hidden_layer_size
        #*****传 输入节点数(就是1，单变量)和隐藏节点数
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        #*****写一个全连接层，因为input_size和hidden——layer_size的叠加使得输出维度并不是1，所以通过...转换到需要的输出维度
        self.linear = nn.Linear(hidden_layer_size, output_size)
        #包含先前的隐藏状态和单元状态
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    #写前向传播
    def forward(self, input_seq):
        #用前面写的lstm属性
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)

        #全连接层输出
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)


model = LSTM()
#初始化损失函数
loss_function = nn.MSELoss()
#优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#训练次数
epochs = 100

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')


#后面调用函数进行预测并画图的部分就跳过
fut_pred = 20

test_inputs = train_data_normalized[-train_window:].tolist()

#不更新梯度
model.eval()

for i in range(fut_pred):
    # 取出(测试集)最后20个数据
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))
print(actual_predictions)
x = np.arange(509, 529, 1)

plt.title('COVID-19')
plt.ylabel('Newly Confimred cases')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
#plt.plot(confirm_list,'b',label='real')
plt.plot(x,test_data,'b',label='real')
#plt.plot(x,actual_predictions,'r',label='estimation')
plt.plot(x,actual_predictions,'r',label='estimation')
plt.show()