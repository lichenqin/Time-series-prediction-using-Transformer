# 导入模块
import math
import pandas as pd
import matplotlib.pyplot as plt

# 获取数据
origin_data = pd.DataFrame(pd.read_csv('C:/Users/11981/Desktop/Data/origin_slice_data.csv', header=None, usecols=[0]))
linear_predict = pd.DataFrame(pd.read_csv('C:/Users/11981/Desktop/Data/Linear_predict_data.csv', header=None, usecols=[0]))
RNN_predict = pd.DataFrame(pd.read_csv('C:/Users/11981/Desktop/Data/RNN_predict_data.csv', header=None, usecols=[0]))
Attention_predict = pd.DataFrame(pd.read_csv('C:/Users/11981/Desktop/Data/Attention_predict_data.csv', header=None, usecols=[0]))
Transformer_predict = pd.DataFrame(pd.read_csv('C:/Users/11981/Desktop/Data/Transformer_predict_data.csv', header=None, usecols=[0]))
# Parameter
max = 11.4628629917588
min = 6.85751406254539


# 获取list数据
def get_data(data):
    data_list = []
    for col in data.columns:
        data_list = data[col]

    return data_list


# 逆归一化
def data_process(data, min, max):
    difference = max - min
    for number in range(0, len(data)):
        data[number] = data[number]*difference+min
    return data


# 获取list
linear_predict = get_data(linear_predict)
RNN_predict = get_data(RNN_predict)
Attention_predict = get_data(Attention_predict)
Transformer_predict = get_data(Transformer_predict)
# 逆归一化
linear_predict = data_process(linear_predict, min, max)
RNN_predict = data_process(RNN_predict, min, max)
Attention_predict = data_process(Attention_predict, min, max)
Transformer_predict = data_process(Transformer_predict, min, max)

plt.plot(origin_data[:50], 'b', label='Ground Truth')
plt.plot(linear_predict[:50], 'orange', label='Linear')
plt.plot(RNN_predict[:50], 'pink', label='RNN')
plt.plot(Attention_predict[:50], 'green', label='Linear_Attention')
plt.plot(Transformer_predict[:50], 'red', label='Transformer')
plt.xlabel('Data Index')
plt.ylabel('Data Value')
plt.legend(loc='best')
# plt.show()
plt.savefig('C:/Users/11981/Desktop/机器学习/论文写作素材/图片素材/Compare.png', bbox_inches='tight')
plt.close()