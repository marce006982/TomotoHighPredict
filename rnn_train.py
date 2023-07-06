import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 假设您已经读取并处理了Excel数据，其中每行对应一天的数据
# 将数据按照植物编号进行分组

data = np.load('data_vector_14.npy')
labels = np.load('labels_vector_14.npy')
raw = np.load('raw_vector_14.npy')

# 定义 RNN 模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])

        return out


# 转换为 PyTorch 张量
data_tensor = torch.from_numpy(data).float()
labels_tensor = torch.from_numpy(labels).float()
raw_tensor = torch.from_numpy(raw).float()

# 划分训练集和测试集
train_ratio = 0.8  # 训练集所占比例
train_size = int(len(data_tensor) * train_ratio)

train_data = data_tensor[:train_size]
train_labels = labels_tensor[:train_size]
test_data = data_tensor[train_size:]
test_labels = labels_tensor[train_size:]
test_raw = raw_tensor[train_size:]


for  n in range(2,3):

    # # 设置训练超参数
    input_size = 33  # 输入特征的维度
    hidden_size = 30 # RNN 隐含层的维度
    num_layers = n  # RNN 的层数
    output_size = 4  # 输出标签的维度
    batch_size = 16  # 批次大小
    num_epochs = 1000  # 训练轮数
    learning_rate = 0.001  # 学习率

    # # 创建 RNN 模型实例
    model = RNN(input_size, hidden_size, num_layers, output_size)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # # 训练模型
    for epoch in range(num_epochs):
        for i in range(0, len(train_data), batch_size):
            # 获取当前批次的数据和标签
            inputs = train_data[i:i+batch_size]
            targets = train_labels[i:i+batch_size]

            # 向前传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 打印每轮训练的损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'model_vector14_1.pth')

    # 在测试集上进行预测
    # 载入已训练好的模型
    # model.load_state_dict(torch.load('model_vector14_1.pth'))
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data)
        loss = criterion(test_outputs, test_labels)
        print(loss.item())
#**************************下面是画图部分**************************#
    # true = np.concatenate((test_raw, test_labels), axis=1)
    # predict = np.concatenate((test_raw, test_outputs), axis=1)
    # x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    # plt.figure()

    # 绘制第一个数据的图像，使用红色线条
    # plt.plot(true[3], color='red', label='true')

    # # 绘制第二个数据的图像，使用蓝色线条
    # plt.plot(predict[3], color='blue', label='predict')

    # # 添加图例
    # plt.legend()

    # # 显示图像
    # plt.show()
    # test_outputs = np.array(test_outputs)
    # test_labels  = np.array(test_labels)
    # size  = test_outputs.shape[0]
    # mpe = np.zeros((94,1))
    # for i in range(0,size):
    #    mpe[i] = abs((test_outputs[i]-test_labels[i])/test_labels[i]) 
    # print(mpe.mean())
        #test_loss = criterion(test_outputs, test_labels)
    # print(test_loss.mean())
        #print(f'Test Loss: {test_loss.item():.4f}')
    # # 生成自然步长的横坐标

    # # 将 test_outputs 和 test_labels 转换为 NumPy 数组
    # test_outputs = test_outputs.numpy()
    # test_labels = test_labels.numpy()

    # # 根据 test_outputs 的大小对 test_outputs 和 test_labels 进行排序
    # index = test_labels.copy()
    # index = np.argsort(index,axis=0)
    # sorted_outputs = test_outputs[index]
    # sorted_labels = test_labels[index]

    # # 可视化显示排序后的点
    # plt.scatter(range(len(test_outputs)), sorted_outputs, label='Test Outputs')
    # plt.scatter(range(len(test_labels)), sorted_labels, label='Test Labels')

    # # 设定 x 轴范围
    # plt.xlim(0, len(test_outputs))

    # # 设定 y 轴范围
    # plt.ylim(0, 200)

    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.legend()
# plt.show()

