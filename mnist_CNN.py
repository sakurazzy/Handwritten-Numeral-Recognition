import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

# # 从torchvision下载mnist手写数字数据集
# train_data = torchvision.datasets.MNIST(
#     root='./mnist/',                                # 根目录
#     train=True,                                     # 选择训练集
#     transform=torchvision.transforms.ToTensor(),    # 转换为tensor数据类型，维度为(C*H*W)，并归一化到[0,1]
#     download=False,                                 # 已经下载过了，不再下载数据集
# )
# print(train_data.data.size())                 # (60000, 28, 28)
# print(train_data.targets.size())               # (60000)
#
# # 创建数据集
# # 建立数据集Loader，每一个batch为50个样本，则数据维度为(50,1,28,28),shuffle打乱样本
# train_loader = Data.DataLoader(dataset=train_data, batch_size=50, shuffle=True)
# # # 选取2000个样本进行测试
# test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)    # 选择根目录，选择测试集
# # torch.unsqueeze扩充维度，进行归一化
# test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000]/255.
# test_y = test_data.targets[:2000]

# 创建CNN网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # 输入维度为(1,28,28)
            nn.Conv2d(
                in_channels=1,              # 输入维度
                out_channels=16,            # 输出维度
                kernel_size=5,              # 卷积核大小
                stride=1,                   # 卷积步长
                padding=2,                  # 为了使卷积操作后图像大小不变，则padding=(kernel_size-1)/2
            ),                              # 经过第一个卷积层，维度变为(16,28,28)
            nn.ReLU(),                      # relu激活函数
            nn.MaxPool2d(kernel_size=2),    # 最大值池化层，维度变为(16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # 输入维度(16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # 维度变为(32, 14, 14)
            nn.ReLU(),                      # relu激活函数
            nn.MaxPool2d(2),                # 最大值池化层，维度变为(32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)# 全连接层，输出维度为10

    # 定义前向通道
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # 把x降维到(batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x

# # 定义训练过程
# cnn = CNN() # 创建CNN网络
# optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)   # 选用Adam优化器
# loss_func = nn.CrossEntropyLoss()                          # loss函数选用交叉熵函数
#
# for epoch in range(1):                                     # 这里只训练一个epoch，已经有比较好的效果
#     for step, (b_x, b_y) in enumerate(train_loader):
#         output = cnn(b_x)[0]            # CNN网络的输出
#         loss = loss_func(output, b_y)   # 计算交叉熵损失函数
#         optimizer.zero_grad()           # 手动清零梯度
#         loss.backward()                 # 向后传播，计算梯度
#         optimizer.step()                # 更新权重参数
#         if step % 50 == 0:              # 每50次对网络进行测试
#             test_output, last_layer = cnn(test_x)
#             pred_y = torch.max(test_output, 1)[1].data.numpy()
#             accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
#             print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
#
# # 取十个测试样本进行测试
# test_output, _ = cnn(test_x[:10])
# pred_y = torch.max(test_output, 1)[1].data.numpy()
# print(pred_y, 'prediction number')
# print(test_y[:10].numpy(), 'real number')
#
# # 保存模型
# torch.save(cnn,'./cnn_model/model.pth')

# 加载模型
cnn = torch.load('./cnn_model/model.pth')

Imagelist = []      # 用于存储待识别图像
# 这里取了10*9个图像，存储真实值
train_y = [[7,3,6,5,8,2,4,9,1],
           [9,7,2,3,8,4,1,4,5],
           [9,7,2,3,8,4,1,4,5],
           [8,6,4,1,3,7,9,2,5],
           [1,9,7,5,4,3,2,8,6],
           [1,9,5,7,4,3,2,6,8],
           [9,1,8,3,2,5,4,7,6],
           [9,1,8,3,2,5,4,7,6],
           [9,1,8,3,2,5,7,4,6],
           [9,1,8,3,2,5,7,4,6]]

# 将待识别图像转换为28*28的归一化灰度图像
for i in range(10):
    for j in range(9):
        img = Image.open('./singblock/block{}_{}.jpg'.format(i,j)).convert('L').resize((28,28))
        arr = []
        for m in range(28):
            for n in range(28):
                pixel = 1.0 - float(img.getpixel((n, m))) / 255.0
                arr.append(pixel)
        arr1 = np.array(arr).reshape((28, 28))
        Imagelist.append(arr1)

train_x = torch.from_numpy(np.array(Imagelist).astype(np.float32))
train_x = torch.unsqueeze(train_x, dim=1)
# print(train_x.shape)

# 对输入数据进行预测
test_output, _ = cnn(train_x)   # 输出的是概率，需要通过索引进行提取
pred_y = torch.max(test_output, 1)[1].data.numpy()
pred_y = np.array(pred_y)

# 取第一幅图像打印预测值和真实值
test_y = []
for i in range(9):
    test_y.append(pred_y[i])
print("第一幅图像预测值：",test_y)
print("第一幅图像真实值:",train_y[0])

# 计算全部测试图像的准确率
count = 0
for i in range(10):
    for j in range(9):
        if pred_y[j+i*9] == train_y[i][j]:
            count += 1
acc = count/90
print('accuracy:',acc)
