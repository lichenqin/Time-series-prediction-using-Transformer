# 导入模块
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error

# 获取数据
data_read = pd.DataFrame(pd.read_csv('./Data/origin_data.csv', header=None, usecols=[0]))


# 原始数据绘图
def plot_origin_data(data):
    plt.hist(data, bins=500, color='steelblue', label='Origin data histogram')  # 返回值元组
    plt.xlabel('Data Value')
    plt.ylabel('Number of Data in same Range')
    plt.legend(loc='upper right')
    # plt.savefig('./Result/1.png', bbox_inches='tight')
    plt.show()
    plt.close()
    plt.plot(data, label='Origin data')
    plt.xlabel('Data Index')
    plt.ylabel('Data Value')
    plt.legend(loc='upper right')
    plt.show()
    plt.close()


def plot_changed_data(data):
    plt.hist(data, bins=500, color='steelblue', label='Changed data histogram')  # 返回值元组
    plt.xlabel('Data Value')
    plt.ylabel('Number of Data in same Range')
    plt.legend(loc='upper right')
    plt.show()
    plt.close()
    plt.plot(data, label='Changed data')
    plt.xlabel('Data Index')
    plt.ylabel('Data Value')
    plt.legend(loc='upper right')
    plt.show()
    plt.close()


def data_log_process(data):
    data_log = []
    for index in range(len(data)):
        data_log.append(float(math.log(data[index])))

    return data_log


# 计算均值 方差 三次方均值
def calc(data):
    n = len(data) # 10000个数
    miu = 0.0   # niu表示平均值,即期望.
    miu2 = 0.0  # niu2表示平方的平均值
    miu3 = 0.0  # niu3表示三次方的平均值
    for a in data:
        miu += a
        miu2 += a**2
        miu3 += a**3
    miu /= n
    miu2 /= n
    miu3 /= n
    sigma = math.sqrt(miu2 - miu*miu)
    return [miu, sigma, miu3]


# 计算偏度 峰度
def calc_stat(data):
    [miu, sigma, miu3] = calc(data)
    n = len(data)
    miu4 = 0.0  # niu4计算峰度计算公式的分子
    for a in data:
        a -= miu
        miu4 += a ** 4
    miu4 /= n

    skew = (miu3 - 3 * miu * sigma ** 2 - miu ** 3) / (sigma ** 3)  # 偏度计算公式
    kurt = miu4 / (sigma ** 4)  # 峰度计算公式:下方为方差的平方即为标准差的四次方
    return [miu, sigma, skew, kurt]


def min_max_scaling(data, max, min):

    for index in range(len(data)):
        if max < data[index]:
            max = data[index]
        if min > data[index]:
            min = data[index]

    print(f'Max:{max},Min:{min}')
    difference = max - min
    print(len(data))
    print(difference)
    for index in range(len(data)):
        data[index] = (data[index]-min)/difference

    plt.hist(data, bins=500, color='steelblue', label='min_max scaling histogram')  # 返回值元组
    plt.legend(loc='upper right')
    plt.show()
    plt.close()
    plt.plot(data, label='min_max scaling data')
    plt.legend(loc='upper right')
    plt.show()
    plt.close()
    return data, max, min


def test_sg_parameter(data, window_length, k):
    data_smooth = savgol_filter(data, window_length*2+1, k)
    mse = mean_squared_error(data, data_smooth)
    print(f'window_length:{window_length},k:{k},mse:{mse}')
    return mse


def find_sg_parameter(data):
    mse_temp = 0.0
    mse_list = []

    for k in range(3, 23, 2):
        for window_length in range(2*k+1, 2*k+2, 2):
            if window_length * 2 + 1 > k:
                mse_temp = test_sg_parameter(data, window_length, k)
                mse_list.append(mse_temp)

    plt.plot(mse_list, 'y', label='SG_filter_mse')
    plt.legend(loc='upper right')
    plt.xlabel('Sliding_window = 2k+1')
    plt.ylabel('MSE')
    plt.show()
    plt.close()


# Parameter
max = 0.0
min = 100.0

data = []
for col in data_read.columns:
    data = data_read[col]

data_log_processed = data_log_process(data)
find_sg_parameter(data)
data_min_max, max, min = min_max_scaling(data_log_processed, max, min)
data_smooth = savgol_filter(data_min_max, 21, 15)

data_csv = pd.DataFrame(data_smooth)
data_csv.to_csv('./Data/input_data.csv', index=False, header=False)

data_smooth = data_smooth[7832:8032]
plt.plot(data_smooth)
plt.show()
plt.close()
difference = max - min
for number in range(0, len(data_smooth)):
    data_smooth[number] = data_smooth[number]*difference + min
plt.plot(data_smooth)
plt.show()
plt.close()
data_csv = pd.DataFrame(data_smooth[-200:])
data_csv.to_csv('./Data/origin_slice_data.csv', index=False, header=False)

