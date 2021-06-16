import math
import torch
import numpy
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
train_data_size = 8000
sliding_window_size = 32
hidden_size = sliding_window_size*2
# 隐层单元数用于提取特征 故设置为input的2倍
output_size = 1     # 最终输出一个结果
iteration = 550
loss_list = []      # loss_list用于收集迭代过程中产生的loss


# define get data function
def get_data(train_data_size, sliding_window_size):
    data_read = pd.DataFrame(pd.read_csv('./Data/input_data.csv', header=None, usecols=[0]))
    data_save = []
    for col in data_read.columns:
        data_save = data_read[col]

    # data_save = data_save[:train_data_size]
    print(len(data_save))
    print(data_save.shape)

    linear_input = []
    linear_target = []
    for number in range(0, train_data_size):
        linear_input.append(data_save[number: number + sliding_window_size])
        linear_target.append(data_save[number + sliding_window_size])

    return linear_input, linear_target


# define train_data and test_data
def get_train_and_test(linear_input, linear_target, split_size, sliding_window_size=32):
    # 划分训练集和测试集，70% 作为训练集
    # split_size between (0,1)
    train_size = int(len(linear_input) * split_size)
    train_input = linear_input[:train_size]
    train_label = linear_target[:train_size]


    train_input = numpy.array(train_input)
    train_label = numpy.array(train_label)
    train_input = train_input.reshape(-1, sliding_window_size)
    train_label = train_label.reshape(-1, 1)
    # reshape(-1,x) 表示先不计算行，先按x分 再计算行数

    # 转换为torch
    train_input = torch.from_numpy(train_input).float()
    train_label = torch.from_numpy(train_label).float()

    print(train_input.shape)
    print(train_label.shape)

    return train_input, train_label


# define Linear Network
# 两层：第一层为隐藏层 64个隐藏单元 第二层为输出层 1个输出单元 激活函数为sigmoid函数
class LinearNet(nn.Module):
    def __init__(self, input_size=32, hidden_size=64, output_size=1, num_layers=1):
        super(LinearNet, self).__init__()

        # define hidden layer
        self.hidden = nn.Linear(input_size, hidden_size)
        # define output layer
        self.output = nn.Linear(hidden_size, output_size)
        # define activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, src):
        src = self.hidden(src)
        src = self.sigmoid(src)
        src = self.output(src)
        output = self.sigmoid(src)

        return output


# define Train function
def train(input_torch, outcome_torch, iteration):
    net = LinearNet(sliding_window_size, hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

    # 开始训练
    for e in range(iteration):

        var_x = Variable(input_torch)
        var_y = Variable(outcome_torch)
        # 前向传播
        out = net(var_x)
        loss = criterion(out, var_y)
        loss_list.append(loss)
        # loss_list used to record loss value in Iteration
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (e + 1) % 20 == 0:  # 每 10 次输出结果
            print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))

    return net


# define evaluate part
def evaluate(net, data_input, data_outcome, sliding_window_size=32):
    net = net.eval()  # 转换成测试模式
    # 将 data_input 转换为 numpy.array
    data_input = numpy.array(data_input)
    data_input = data_input.reshape(-1, sliding_window_size)
    data_input = torch.from_numpy(data_input).float()

    var_data = Variable(data_input)
    print(var_data.shape)
    # 测试集的预测结果
    pred_test = net(var_data)
    # 改变输出的格式
    pred_test = pred_test.view(-1).data.numpy()
    print("prediciton set shape:", pred_test.shape)
    # 画出实际结果和预测的结果
    prediction_plot = pred_test[-200:]
    # plt.plot(prediction_plot, 'r', label='Prediction')
    # plt.legend(loc='upper right')
    # real = data_outcome[7800:]
    # plt.plot(real, 'b', label='Ground Truth')
    # plt.xlabel("Data Index")
    # plt.ylabel("Data Value")
    # plt.legend(loc='upper right')
    # # plt.show()
    # plt.close()
    # plt.plot(loss_list, 'y', label='Loss')
    # plt.legend(loc='upper right')
    # plt.xlabel("Epoch")
    # plt.ylabel("MSE")
    # #plt.show()
    # plt.close()
    # 计算误差
    mse = mean_squared_error(pred_test[5600:], data_outcome[5600:])
    mae = mean_absolute_error(pred_test[5600:], data_outcome[5600:])
    mape = mean_absolute_percentage_error(pred_test[5600:], data_outcome[5600:])
    print(f"均方误差(MSE)：{mse}")
    print(f"平均绝对误差(MAE):{mae}")
    print(f"平均绝对分差(MAPE):{mape}")
    # 保存预测结果
    # print(data_outcome[5600])
    data_csv = pd.DataFrame(prediction_plot)
    data_csv.to_csv('./Data/Transformer_predict_data.csv', index=False, header=False)


input, target = get_data(train_data_size, sliding_window_size)
train_input, train_target = get_train_and_test(input, target, 0.7, sliding_window_size)
net = train(train_input, train_target, iteration)
evaluate(net, input, target, sliding_window_size)