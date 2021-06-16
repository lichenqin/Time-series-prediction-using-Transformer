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
hidden_size = 32  # 等同于输入32个
origin_size = 1
extend_size = 32
output_size = 1
iteration = 1000
# Record
loss_list = []


# Position Encoding Part
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


# Input Fully Connect Layer
# input: (seq_len, batch_size, input_size)
# output: (seq_len, batch_size, output_size)
class InputLayer(nn.Module):

    def __init__(self, input_size, output_size):
        super(InputLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        # print("after extend:", x.shape)
        return x

    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(0, initrange)


# Output Fully Connect Layer
# input: (seq_len, batch_size, output_size)
# first_output: (seq_len, batch_size, origin_size)
# second_output: (batch_size, origin_size)
class OutputLayer(nn.Module):
    def __init__(self, input_size, output_size, sequence_length):
        super(OutputLayer, self).__init__()

        # define hidden layer
        self.hidden = nn.Linear(input_size, output_size)
        # define output layer
        self.output = nn.Linear(sequence_length, output_size)
        # define activation function
        self.sigmoid = nn.Sigmoid()
        # save input_size
        self.sequence = sequence_length
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.hidden.bias.data.zero_()
        self.output.bias.data.zero_()
        self.hidden.weight.data.uniform_(0, initrange)
        self.hidden.weight.data.uniform_(0, initrange)

    def forward(self, src):
        # print('before output layer:', src.shape)
        src = self.hidden(src)
        src = self.sigmoid(src)
        # print('output layer middle shape:', src.shape)
        src = src.reshape(self.sequence, -1)
        # print(src.shape)
        src = torch.transpose(src, 0, 1)
        # print("after transpose:", src.shape)
        src = self.output(src)
        output = self.sigmoid(src)
        # print('output layer final shape:', output.shape)
        return output


# define Transformer_Encoder + Linear part
class TransAm(nn.Module):
    def __init__(self, sequence_length=32, origin_size=1, feature_size=8, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.input_layer = InputLayer(origin_size, feature_size)
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=2, dropout=dropout, dim_feedforward=feature_size*4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = OutputLayer(feature_size, origin_size, sequence_length)
        # self.init_weights()

    # def init_weights(self):
    #     initrange = 0.1
    #     self.decoder.bias.data.zero_()
    #     self.decoder.weight.data.uniform_(0, initrange) # 定义decoder为Linear

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        # print(src.shape)
        src = self.input_layer(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# define get data function
def get_data(train_data_size, sliding_window_size):
    data_read = pd.DataFrame(pd.read_csv('./Data/input_data.csv', header=None, usecols=[0]))
    data_save = []
    for col in data_read.columns:
        data_save = data_read[col]

    input = []
    target = []
    # 构造标签数据
    for number in range(0, train_data_size):
        target.append(data_save[number + sliding_window_size])

    # 构造输入数据(seq_length, batch_size, feature_size)
    # 其中sequence_length 代表输入数组一个句子输入多少单词
    # batch_size 代表共有多少个句子
    # 注意不要被网络上的信息混淆，目前pytorch框架还没有对Transformer模型增加batch_first参数
    for seq in range(0, sliding_window_size):
        for number in range(seq, train_data_size+seq):
            input.append(data_save[number])

    input = numpy.array(input).reshape(-1, train_data_size, 1)
    target = numpy.array(target).reshape(train_data_size, 1)

    print("After get data:")
    print(input.shape)  # should be (32, 8000, 1)
    print(target.shape) # should be (8000, 1)

    return input, target


# define train_data and test_data
def get_train_and_test(input_src, target_src, train, split_size):
    # 划分训练集和测试集，70% 作为训练集
    # split_size between (0,1)
    train_size = int(train * split_size)
    input = input_src[:, :train_size, :]
    target = target_src[:train_size, :]
    test_input = input_src[:, train_size:8000, :]
    test_target = target_src[train_size:8000, :]

    # 转换为torch
    src = torch.from_numpy(input).float()
    tgt = torch.from_numpy(target).float()
    test_input = torch.from_numpy(test_input).float()
    test_target = torch.from_numpy(test_target).float()

    print("After split:")
    print(src.shape)
    print(tgt.shape)
    print(test_input.shape)
    print(test_target.shape)

    return src, tgt, test_input, test_target


# define Transformer Train Part
def trans_train(src, tgt, iteration):
    # transformer_model = nn.Transformer(d_model=32, dropout=0.5)
    transformer_model = TransAm().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.01)
    print("Start Training:")

    for epoch in range(1, iteration + 1):
        src = Variable(src)
        tgt = Variable(tgt)
        # 前向传播
        out = transformer_model(src)
        # print(out.shape)
        loss = criterion(out, tgt)
        loss_list.append(loss)
        # loss_list used to record loss value in Iteration
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch % 10 == 0):
            print(f'Epoch:{epoch}    Loss:{loss.item()} ')
            if loss.item() < 0.002:
                break
            if (loss.item() > 0.02 and epoch > 170):
                break

    return transformer_model


# define evaluate part
def evaluate(net, data_input, data_outcome):
    net = net.eval()  # 转换成测试模式
    # 测试集的预测结果
    var_data = Variable(data_input)
    pred_test = net(var_data)
    print("prediciton set shape:", pred_test.shape)
    # 改变输出的格式
    pred_test = pred_test.view(-1).data.numpy()
    data_outcome = data_outcome.view(-1).data.numpy()
    # 画出实际结果和预测的结果
    prediction_plot = pred_test[-200:]
    # plt.plot(prediction_plot, 'r', label='prediction')
    # plt.legend(loc='upper right')
    # origin_plot = data_outcome[-200:]
    # plt.plot(origin_plot, 'b', label='origin')
    # plt.legend(loc='upper right')
    # plt.show()
    # plt.close()
    # plt.plot(prediction_plot, 'r', label='Prediction')
    # plt.ylabel("Data Value")
    # plt.xlabel("Data Index")
    # plt.legend(loc='upper right')
    # plt.plot(origin_plot, 'b', label='Ground Truth')
    # plt.legend(loc='upper right')
    # plt.show()
    # plt.close()
    # plt.plot(loss_list, 'y', label='loss')
    # plt.legend(loc='upper right')
    # plt.show()
    # plt.close()
    # plt.plot(loss_list, 'y', label='Loss')
    # plt.xlabel("Epoch")
    # plt.ylabel("MSE")
    # plt.legend(loc='upper right')
    # plt.show()
    # plt.close()
    # 计算误差
    mse = mean_squared_error(pred_test, data_outcome)
    mae = mean_absolute_error(pred_test, data_outcome)
    mape = mean_absolute_percentage_error(pred_test, data_outcome)
    print(f"均方误差(MSE)：{mse}")
    print(f"平均绝对误差(MAE):{mae}")
    print(f"平均绝对分差(MAPE):{mape}")
    # 保存预测结果
    print(data_outcome[0])
    data_csv = pd.DataFrame(pred_test[-200:])
    data_csv.to_csv('./Data/Attention_predict_data.csv', index=False, header=False)


input_numpy, target_numpy = get_data(train_data_size=train_data_size, sliding_window_size=sliding_window_size)
input_torch, target_torch, test_torch, test_target = get_train_and_test(input_numpy, target_numpy, train=train_data_size, split_size=0.7)
net = trans_train(input_torch, target_torch, iteration=iteration)
evaluate(net, test_torch, test_target)
