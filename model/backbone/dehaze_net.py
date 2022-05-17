import torch
import torch.nn as nn
import math
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader  # 我们要加载数据集的
from torchvision import transforms  # 数据的原始处理
from torchvision import datasets
import torch.nn.functional as F  # 激活函数
import torch.optim as optim


class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        # [64,1,28,28]-[64,16,28,28] 通过卷积核为一的卷积，图像尺寸不变
        self.branch1x1 = nn.Conv2d(in_channels, 3, kernel_size=1, padding=0)

        self.branch5x5_1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        # [64,16,28,28]-[64,24,28,28] 通过卷积核为5的卷积，因为w，h分别填充了2，所以图像尺寸不变
        # self.branch5x5_2 = nn.Conv2d(16,24, kernel_size=5,padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 3, kernel_size=5, padding=2)
        # [64,16,28,28]-[64,24,28,28] 通过卷积核为3的卷积，因为填充了1，所以图像尺寸不变
        # self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        # [64,24,28,28]-[64,24,28,28] 通过卷积核为3的卷积，因为填充了1，所以图像尺寸不变
        # self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        # [64,1,28,28]-[64,24,28,28]
        # self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        # branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        # branch3x3 = self.branch3x3_2(branch3x3)
        # branch3x3 = self.branch3x3_3(branch3x3)

        # 平均池化尺寸不变
        # branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        # branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3]
        # [b,c,h,w] dim=1是维度方向拼接 ，这里返回的维度是上述几个维度相加(16+24+24+24=88)
        return torch.cat((branch1x1, branch5x5, branch3x3), dim=1)


class dehaze_net(nn.Module):

    def __init__(self):
        super(dehaze_net, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        # self.e_conv1 = InceptionA(in_channels=3)
        # 输出9

        self.block0 = nn.Sequential(nn.Conv2d(1, 3, 1, 1, 0, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(3, 6, 3, 1, 1, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(6, 6, 5, 1, 2, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(6, 3, 5, 1, 2, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(3, 1, 5, 1, 2, bias=True))

        self.block1 = nn.Sequential(nn.Conv2d(1, 3, 1, 1, 0, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(3, 6, 3, 1, 1, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(6, 6, 5, 1, 2, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(6, 3, 5, 1, 2, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(3, 1, 5, 1, 2, bias=True))

        self.block2 = nn.Sequential(nn.Conv2d(1, 3, 1, 1, 0, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(3, 6, 3, 1, 1, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(6, 6, 5, 1, 2, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(6, 3, 5, 1, 2, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(3, 1, 5, 1, 2, bias=True))

    def forward(self, x):
        source = []
        # print(x.size())
        # print(x[:, :, :,:].size())
        # print(x[:, 1, :,:].size())
        # print(x[:, 0, :,:].size())
        x0 = x[:, 0, :, :]
        x1 = x[:, 1, :, :]
        x2 = x[:, 2, :, :]
        # print(x0.size())
        # print(x1.size())
        # print(x2.size())
        a1 = x0.unsqueeze(1)#升维操作
        a2 = x1.unsqueeze(1)
        a3 = x2.unsqueeze(1)

        source.append(x)

        b0 = self.relu(self.block0(a1))

        b1 = self.relu(self.block1(a2))

        b2 = self.relu(self.block2(a3))
        c3 = torch.cat((b0, b1, b2), 1)
        # print(c3.size())

        clean_image = self.relu((c3 * x) - c3 + 1)
        return clean_image


if __name__ == "__main__":
    x = torch.randn(2, 3, 28, 28)
    dehaze  = dehaze_net()
    out = dehaze.forward(x)
    print(out.shape)