import torch.nn as nn
import torch.nn.functional as F
import torch

# 定义模型
class GoCNN(nn.Module):
    def __init__(self):
        super(GoCNN, self).__init__()

        # 设定更少的通道数
        self.in_channels = 32

        # 输入层
        self.conv1 = nn.Conv2d(3, self.in_channels,
                               kernel_size=3, stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # 中间卷积层
        self.conv2 = nn.Conv2d(self.in_channels,
                               self.in_channels,
                               kernel_size=3, stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(self.in_channels)

        # 输出层
        self.fc1 = nn.Linear(19 * 19 * self.in_channels,
                             128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


