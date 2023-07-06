import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

data = []  # 用于存储每株植物的序列数据
labels = []
raw_data = []
df = pd.read_excel('final_data.xlsx').drop('高度', axis=1)
offset = 0
data_size = 3276
# 将每7天的数据作为一组，其中前6天为输入序列，第7天为标签序列
# 已改为14天一组
for i in range(0,data_size,14):
        input_sequence = df.iloc[i+offset:i+offset+10,:-2].values  # 输入序列为前10天的数据
        raw_data_seq = [df.iloc[i+offset+0]["high"],df.iloc[i+offset+1]["high"],df.iloc[i+offset+2]["high"],df.iloc[i+offset+3]["high"],
                    df.iloc[i+offset+4]["high"],df.iloc[i+offset+5]["high"],df.iloc[i+offset+6]["high"],df.iloc[i+offset+7]["high"],
                    df.iloc[i+offset+8]["high"],df.iloc[i+offset+9]["high"]]
        label_sequence = [df.iloc[i+offset+10]["high"],df.iloc[i+offset+11]["high"],df.iloc[i+offset+12]["high"],df.iloc[i+offset+13]["high"]]    # 标签序列为第11-14天的数据
        data.append(input_sequence)
        labels.append(label_sequence)
        raw_data.append(raw_data_seq)

# 生成随机排列索引数组
random_indices = np.random.permutation(len(data))

# 使用随机排列索引数组来重排 data 和 labels 列表
data_shuffled = [data[i] for i in random_indices]
labels_shuffled = [labels[i] for i in random_indices]
raw_data_shuffled = [raw_data[i] for i in random_indices]

np.save('data_vector_14.npy',data_shuffled)
np.save('labels_vector_14.npy',labels_shuffled)
np.save('raw_vector_14.npy',raw_data_shuffled)

print(len(data_shuffled),len(labels_shuffled))
