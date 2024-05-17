import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout,Flatten
import matplotlib.pyplot as plt
import glob , os
import seaborn as sns
import sys
from sklearn.preprocessing import MinMaxScaler

data_path = ".\\industry_timeseries\\"

# 读取数据
columns = ['YEAR','MONTH','DAY','TEMP_HIG','TEMP_COL',
                    'AVG_TEMP','AVG_WET','DATA_COL']
data = pd.read_csv(data_path + "timeseries_train_data\\2.csv",
                    names=columns)

# 数据预处理：将序列数据转化为监督问题数据
from pandas import DataFrame
from pandas import concat

def convert_data_to_supervised(data, num_in=1,num_out=1,dropnan=True):
    num_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = [],[]
    for i in range(num_in, 0 ,-1):
        cols.append(df.shift(i))
        names+=[('列 %d(t-%d)' % (j+1,i)) for j in range(num_vars)]
    for i in range(0,num_out):
        cols.append(df.shift(-i))
        if i==0:
            names+=[('列 %d(t)' % (j+1)) for j in range(num_vars)]
        else:
            names+=[('列 %d(t+%d)' % (j+1,i)) for j in range(num_vars)]
    formal_data = concat(cols,axis=1)
    formal_data.columns = names

    if dropnan:
        formal_data.dropna(inplace=True)
    return formal_data

# 数据归一化
scaler = MinMaxScaler(feature_range=(0,1))
scalered_data = scaler.fit_transform(data[['DATA_COL','TEMP_HIG',
                    'TEMP_COL','AVG_TEMP','AVG_WET']].values)

# 将数据转化为 LSTM可识别的形式
reframed = convert_data_to_supervised(scalered_data,1,1)
reframed.drop(reframed.columns[[6,7,8,9]], axis=1,inplace=True)

# 数据集划分
train_days = 400
valid_days = 150
values = reframed.values
train_data = values[:train_days, :] 
valid_data = values[train_days:train_days+valid_days, :]     
test_data = values[train_days+valid_days:, :]       

# 训练集、验证集、测试集  划分  自变量 和 因变量
train_data_X, train_data_Y = train_data[:, :-1], train_data[:, -1]  
valid_data_X, valid_data_Y = valid_data[:, :-1], valid_data[:, -1]
test_data_X, test_data_Y = test_data[:, :-1], test_data[:, -1]

# 重构数据的维度
train_data_X = train_data_X.reshape((train_data_X.shape[0],
                    1, train_data_X.shape[1]))
valid_data_X = valid_data_X.reshape((valid_data_X.shape[0],
                    1, valid_data_X.shape[1]))
test_data_X = test_data_X.reshape((test_data_X.shape[0], 
                    1, test_data_X.shape[1]))

model = Sequential()
model.add(LSTM(1024, activation='relu',
                    input_shape=(train_data_X.shape[1],
                    train_data_X.shape[2]), return_sequences=True))
model.add(Flatten())
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# fit network
hist = model.fit(train_data_X,train_data_Y,
                    epochs=350, batch_size=16, 
                    validation_data=(valid_data_X,valid_data_Y), 
                    verbose=1, shuffle=False)
# 损失可视化
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='valid')
plt.legend()
plt.show()
plt.figure(figsize=(24,8))

train_predict = model.predict(train_data_X)
valid_predict = model.predict(valid_data_X)
test_predict = model.predict(test_data_X)

with open('Origin_data.txt', 'w', encoding='utf-8') as file:
    for item in values[:,-1]:
        file.write("%s\n" % item)  # 每个元素后写入换行符

train_show = []
for i in train_predict:
    train_show.append(i)

valid_show = []
for i in valid_predict:
    valid_show.append(i)

test_show = []
for i in test_predict:
    test_show.append(i)

# 记录所有的结果
with open('Train_Predict.txt', 'w', encoding='utf-8') as file:
    for item in train_predict[:,-1]:
        file.write("%s\n" % item)  # 每个元素后写入换行符
with open('Valid_Predict.txt', 'w', encoding='utf-8') as file:
    for item in valid_predict[:,-1]:
        file.write("%s\n" % item)  # 每个元素后写入换行符
with open('Test_Predict.txt', 'w', encoding='utf-8') as file:
    for item in test_predict[:,-1]:
        file.write("%s\n" % item)  #

plt.plot(np.arange(577),values[:,-1], label="Train set (actual)")
plt.plot(np.arange(400),train_show, label="Train set (predict)")
plt.plot(np.arange(400,550),valid_show, label="Valid set (predict)")
plt.plot(np.arange(550,577),test_predict, label="Test set (predict)")
label="Test set (predict)")
plt.legend(fontsize=16)
plt.savefig("./New_pig.svg",bbox_inches='tight')
