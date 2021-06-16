import torch
import numpy
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Parameters
train_data_size = 8001
sliding_window_size = 32
input_size = 1
hidden_size = 32  # 等同于输入32个
output_size = 1
num_layers = 1
batch_size = 1
iteration = 300


# 定义模型 rnn + linear_regression
class rnn_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(rnn_reg, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='relu', batch_first=True)  # rnn
        self.reg = nn.Linear(hidden_size, output_size)  # 回归

    def forward(self, x, h_init):
        x, hn = self.rnn(x, h_init)  # (seq, batch, hidden)
        # print(x.shape)
        # print(hn.shape)
        s, b, h = x.shape
        hn = hn.reshape(-1, h)
        x = self.reg(hn)
        # print(x.shape)
        return x

    def init_hidden(self, batch, hidden_size):
        return torch.zeros(1, batch, hidden_size)


# define get data function
def get_data(train_data_size, sliding_window_size):
    data_read = pd.DataFrame(pd.read_csv('./Data/input_data.csv', header=None, usecols=[0]))
    data_save = []
    for col in data_read.columns:
        data_save = data_read[col]

    # data_save = data_save[:train_data_size]
    print(len(data_save))
    print(data_save.shape)

    data_input = []
    data_outcome = []
    for number in range(0, train_data_size - 1):
        data_input.append(data_save[number:number + sliding_window_size])
        data_outcome.append(data_save[number + sliding_window_size])
    print(data_outcome[0])
    print(data_save[32])
    print("data_outcome length:", len(data_outcome))
    return data_input, data_outcome


# define train_data and test_data
def get_train_and_test(input, label, split_size):
    # 划分训练集和测试集，70% 作为训练集
    # split_size between (0,1)
    train_size = int(len(input) * split_size)
    train_x = input[:train_size]
    train_y = label[:train_size]
    test_x = input[train_size:]
    test_y = label[train_size:]

    train_x = numpy.array(train_x).reshape(-1, sliding_window_size, 1)
    train_y = numpy.array(train_y).reshape(-1, 1)
    text_x = numpy.array(test_x).reshape(-1, sliding_window_size, 1)
    test_y = numpy.array(test_y).reshape(-1, 1)
    # reshape(-1,x) 表示先不计算行，先按x分 再计算行数

    # 转换为torch
    input_torch = torch.from_numpy(train_x).float()
    outcome_torch = torch.from_numpy(train_y).float()
    test_x = torch.from_numpy(text_x).float()
    test_y = torch.from_numpy(test_y).float()

    return input_torch, outcome_torch, test_x, test_y


# define Train function
def train(input_torch, outcome_torch, input_size, hidden_size, iteration):
    net = rnn_reg(input_size, hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    # Record
    loss_list = []
    # batch_size
    b, _, _ = input_torch.shape

    # 开始训练
    for e in range(iteration):

        var_x = Variable(input_torch)
        var_y = Variable(outcome_torch)
        hidden_init = Variable(net.init_hidden(b, hidden_size))
        # 前向传播
        out = net(var_x, hidden_init)
        loss = criterion(out, var_y)
        loss_list.append(loss)
        # loss_list used to record loss value in Iteration
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (e + 1) % 10 == 0:  # 每 100 次输出结果
            print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))

    return net, loss_list


# define evaluate part
def evaluate(net, data_input, data_outcome, sliding_window_size):
    net = net.eval()  # 转换成测试模式
    # 将 data_input 转换为 numpy.array
    data_input = numpy.array(data_input)
    data_input = data_input.reshape(-1, sliding_window_size, 1)
    data_input = torch.from_numpy(data_input).float()
    var_data = Variable(data_input)
    # 测试集的预测结果
    b, _, _ = data_input.shape
    h_init = Variable(net.init_hidden(b, hidden_size))
    pred_test = net(var_data, h_init)
    # 改变输出的格式
    pred_test = pred_test.view(-1).data.numpy()
    data_outcome = data_outcome.view(-1).data.numpy()
    print("prediciton set shape:", pred_test.shape)
    # 画出实际结果和预测的结果
    prediction_plot = pred_test[-200:]
    plt.plot(prediction_plot, 'r', label='Prediction')
    plt.legend(loc='upper right')
    origin_plot = data_outcome[-200:]
    plt.plot(origin_plot, 'b', label='GroundTruth')
    plt.xlabel("Data Index")
    plt.ylabel("Data Value")
    plt.legend(loc='upper right')
    plt.show()
    plt.close()
    plt.plot(loss_list, 'y', label='loss')
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()
    plt.close()
    # 计算误差
    mse = mean_squared_error(pred_test, data_outcome)
    mae = mean_absolute_error(pred_test, data_outcome)
    mape = mean_absolute_percentage_error(pred_test, data_outcome)
    print(f"均方误差(MSE)：{mse}")
    print(f"平均绝对误差(MAE):{mae}")
    print(f"平均绝对分差(MAPE):{mape}")
    # 保存预测结果
    # print(data_outcome[0])
    data_csv = pd.DataFrame(prediction_plot)
    data_csv.to_csv('./Data/RNN_predict_data.csv', index=False, header=False)


# Get Data
data_input, data_outcome = get_data(train_data_size, sliding_window_size)
input_torch, outcome_torch, test_input, test_outcome = get_train_and_test(data_input, data_outcome, 0.7)
# Training
print(input_torch.shape, outcome_torch.shape, test_input.shape, test_outcome.shape)
net, loss_list = train(input_torch, outcome_torch, input_size, hidden_size, iteration)
# Evaluating
evaluate(net, test_input, test_outcome, sliding_window_size)
