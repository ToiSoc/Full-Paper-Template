import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

#设置随机种子
import random
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 导入数据
train_df=pd.read_csv("./DailyDelhiClimateTrain.csv")

# 绘制相关性热力图
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']    
plt.rcParams['axes.unicode_minus'] = False      
for_heatmap_data = train_df[['meantemp','humidity',
                'wind_speed','meanpressure']]
scaler = MinMaxScaler(feature_range=(0,1))
scalered_data = scaler.fit_transform(for_heatmap_data.values)

# 数据可视化
scaled_df = pd.DataFrame(scalered_data, 
                columns=['日平均气温','相对湿度','风速','平均气压'])
scalered_data_corr = scaled_df.corr()
fontsize_main = 22
fig = plt.figure(figsize=(12,10))
ax = sns.heatmap(scalered_data_corr,cmap="BuGn",annot = True,
                annot_kws={"fontsize": fontsize_main})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

# 设置x轴、y轴坐标
for item in ax.get_xticklabels():
    item.set_rotation(0)
for item in ax.get_yticklabels():
    item.set_rotation(0)
ax.tick_params(labelsize=fontsize_main) 
plt.savefig("./CNN-LSTM相关性热力图.pdf",bbox_inches='tight')

# 数据归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
meantemp = scaler.fit_transform(meantemp.reshape(-1,1))

# 数据划分
def split_data(data,time_step=12):
    dataX=[]
    datay=[]
    for i in range(len(data)-time_step):
        dataX.append(data[i:i+time_step])
        datay.append(data[i+time_step])
    dataX=np.array(dataX).reshape(len(dataX),time_step,-1)
    datay=np.array(datay)
    return dataX,datay

dataX,datay=split_data(meantemp,time_step=12)

#划分训练集和测试集的函数
def train_test_split(dataX,datay,shuffle=True,percentage=0.8):
    """
    将训练数据X和标签y以numpy.array数组的形式传入
    划分的比例定为训练集:测试集=8:2
    """
    if shuffle:
        random_num=[index for index in range(len(dataX))]
        np.random.shuffle(random_num)
        dataX=dataX[random_num]
        datay=datay[random_num]
    split_num=int(len(dataX)*percentage)
    train_X=dataX[:split_num]
    train_y=datay[:split_num]
    test_X=dataX[split_num:]
    test_y=datay[split_num:]
    return  train_X,train_y,test_X,test_y
            train_X,train_y,test_X,
            test_y=train_test_split(dataX,datay,
            shuffle=False,percentage=0.8)

train_X,train_y,test_X,test_y=train_test_split(dataX,datay,
            shuffle=False,percentage=0.8)

# 定义CNN+LSTM模型类
class CNN_LSTM(nn.Module):
    def __init__(self, conv_input,input_size, hidden_size, num_layers, output_size):
        super(CNN_LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv=nn.Conv1d(conv_input,conv_input,1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x=self.conv(x)
        # 初始化隐藏状态h0
        h0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size)
        # 初始化记忆状态c0
        c0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size) 
        #print(f"x.shape:{x.shape},h0.shape:{h0.shape},c0.shape:{c0.shape}")
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 网络训练
test_X1=torch.Tensor(test_X)
test_y1=torch.Tensor(test_y)

# 定义输入、隐藏状态和输出维度
input_size = 1
conv_input=12
hidden_size = 64
num_layers = 5
output_size = 1

# 创建CNN_LSTM模型实例
model =CNN_LSTM(conv_input,input_size, hidden_size, num_layers, output_size)

#训练周期为500次
num_epochs=1000
batch_size=16
optimizer=optim.Adam(model.parameters(),lr=0.0001,betas=(0.5,0.999))
criterion=nn.MSELoss()

train_losses=[]
test_losses=[]
for epoch in range(num_epochs):
    random_num=[i for i in range(len(train_X))]
    np.random.shuffle(random_num)
    
    train_X=train_X[random_num]
    train_y=train_y[random_num]
    train_X1=torch.Tensor(train_X[:batch_size])
    train_y1=torch.Tensor(train_y[:batch_size])
    
    model.train()
    optimizer.zero_grad()
    output=model(train_X1)
    train_loss=criterion(output,train_y1)
    train_loss.backward()
    optimizer.step()
    
    if epoch%2==0:
        model.eval()
        with torch.no_grad():
            output=model(test_X1)
            test_loss=criterion(output,test_y1)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"epoch:{epoch},train_loss:{train_loss},test_loss:{test_loss}")

# 训练结果反归一化并可视化
train_X1=torch.Tensor(X_train)
train_pred=model(train_X1).detach().numpy()
test_pred=model(test_X1).detach().numpy()
pred_y=np.concatenate((train_pred,test_pred))
pred_y=scaler.inverse_transform(pred_y).T[0]
true_y=np.concatenate((y_train,test_y))
true_y=scaler.inverse_transform(true_y).T[0]

#可视化
plt.title("CNN_LSTM")
x=[i for i in range(len(true_y))]
plt.plot(x,pred_y,marker="o",markersize=1,label="pred_y")
plt.plot(x,true_y,marker="x",markersize=1,label="true_y")
plt.legend()
plt.show()
