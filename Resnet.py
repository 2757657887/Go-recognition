import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
import torch.nn as nn
import torch


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        # 使用ResNet18的权重
        self.model = models.resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.conv1 = nn.Conv2d(3, 64,
                                     kernel_size=(3, 3),
                                     stride=(1, 1),
                                     padding=(1, 1))

        # 更改最后一层以用于回归问题
        # 假设您的回归问题是预测一个值
        self.model.fc = nn.Linear(in_features=512,
                                  out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x


# model = ResNet()
#
# x = torch.randn(1, 3, 19, 19)
# out = model(x)
# print(out)
