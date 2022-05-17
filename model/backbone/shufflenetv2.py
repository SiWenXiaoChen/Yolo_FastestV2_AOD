import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class dedust_net(nn.Module):

    def __init__(self):
        super(dedust_net, self).__init__()

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
        a1 = x0.unsqueeze(1)  # 升维操作
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


class AODnet(nn.Module):
    def __init__(self):
        super(AODnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, padding=1)
        self.b = 1

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = F.relu(self.conv5(cat3))

        if k.size() != x.size():
            raise Exception("k, haze image are different size!")

        output = k * x - k + self.b
        return F.relu(output)


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]  # 判断语句是否正确，不正确直接返回错误

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            # 深度分组卷积，分组数为mid_channles
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)  # 为什么通道数要满足整除4的条件
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)#改变矩阵行列
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleNetV2(nn.Module):
    def __init__(self, stage_out_channels, load_param):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = stage_out_channels

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage2", "stage3", "stage4"]
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            stageSeq = []
            for i in range(numrepeat):
                if i == 0:
                    stageSeq.append(ShuffleV2Block(input_channel, output_channel,
                                                   mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    stageSeq.append(ShuffleV2Block(input_channel // 2, output_channel,
                                                   mid_channels=output_channel // 2, ksize=3, stride=1))
                input_channel = output_channel
            setattr(self, stage_names[idxstage], nn.Sequential(*stageSeq))#改变对象属性的值：self.stage2 = nn.Sequential(*stageSeq)

        if load_param == False:
            self._initialize_weights()
        else:
            print("load param...")

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        C1 = self.stage2(x)
        C2 = self.stage3(C1)
        C3 = self.stage4(C2)

        return C2, C3

    def _initialize_weights(self):
        print("initialize_weights...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load("./model/backbone/backbone.pth", map_location=device), strict=True)


class ShuffleNetV2_dehaze(nn.Module):
    def __init__(self, stage_out_channels, load_param):
        super(ShuffleNetV2_dehaze, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = stage_out_channels

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.dehaze = nn.Sequential(
            # nn.Conv2d(176, 3, kernel_size=1, padding=88),#上采样，还原图片尺寸
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(24, 3, kernel_size=1),
            AODnet()
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage2", "stage3", "stage4"]
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            stageSeq = []
            for i in range(numrepeat):
                if i == 0:
                    stageSeq.append(ShuffleV2Block(input_channel, output_channel,
                                                   mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    stageSeq.append(ShuffleV2Block(input_channel // 2, output_channel,
                                                   mid_channels=output_channel // 2, ksize=3, stride=1))
                input_channel = output_channel
            setattr(self, stage_names[idxstage], nn.Sequential(*stageSeq))

    def forward(self, x):
        x = self.first_conv(x)
        clean_image = self.dehaze(x)
        x = self.maxpool(x)
        C1 = self.stage2(x)
        C2 = self.stage3(C1)
        C3 = self.stage4(C2)

        return C2, C3, clean_image

    def _initialize_weights(self):
        print("initialize_weights...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load("./model/backbone/backbone.pth", map_location=device), strict=True)


class ShuffleNetV2_dehaze_test(nn.Module):
    def __init__(self, stage_out_channels, load_param):
        super(ShuffleNetV2_dehaze_test, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = stage_out_channels

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage2", "stage3", "stage4"]
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            stageSeq = []
            for i in range(numrepeat):
                if i == 0:
                    stageSeq.append(ShuffleV2Block(input_channel, output_channel,
                                                   mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    stageSeq.append(ShuffleV2Block(input_channel // 2, output_channel,
                                                   mid_channels=output_channel // 2, ksize=3, stride=1))
                input_channel = output_channel
            setattr(self, stage_names[idxstage], nn.Sequential(*stageSeq))

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        C1 = self.stage2(x)
        C2 = self.stage3(C1)
        C3 = self.stage4(C2)

        return C2, C3

    def _initialize_weights(self):
        print("initialize_weights...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load("./model/backbone/backbone.pth", map_location=device), strict=True)


if __name__ == "__main__":
    model = ShuffleNetV2_dehaze_test([-1, 24, 48, 96, 192], True)
    print(model)
    test_data = torch.rand(1, 3, 352, 352)

    test_outputs = model(test_data)
    # for out in test_outputs:
    # print(out.size())
