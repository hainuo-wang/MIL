import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, hw, chs):
        super(Attention, self).__init__()
        self.in_chs = hw
        # self.se1 = SENet(chs=chs)
        # self.se2 = SENet(chs=chs)
        self.att = nn.Sequential(
            nn.Linear(hw, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x = cat([instance1, instance2])
        # x: batch_size, (512, 512), 32, 32
        # x: batch_size, (256, 256), 64, 64
        batch_size, chs, h, w = x.size()
        # instance_1 = self.se1(x[:, :chs//2])
        # instance_2 = self.se2(x[:, chs//2:])
        instance_1 = x[:, :chs // 2]
        instance_2 = x[:, chs // 2:]
        instance_1 = torch.mean(instance_1, dim=1).unsqueeze(1)
        instance_2 = torch.mean(instance_2, dim=1).unsqueeze(1)
        bag = torch.cat([instance_1, instance_2], dim=1)  # (batch_size, 2, h, w)
        bag = bag.view(batch_size, 2, h * w)  # (batch_size, 2, h*w)
        weight = self.att(bag)  # (batch_size, 2, h*w) -> (batch_size, 2, 1)
        weight = weight.squeeze(2)  # (batch_size, 2)
        weight = torch.softmax(weight, dim=1)
        w1 = weight[:, 0].unsqueeze(1).repeat(1, chs // 2).view(batch_size, chs // 2, 1, 1)
        w2 = weight[:, 1].unsqueeze(1).repeat(1, chs // 2).view(batch_size, chs // 2, 1, 1)
        weight = torch.cat([w1, w2], dim=1)
        x = x * weight
        return x[:, :chs // 2] + x[:, chs // 2:]


# model = Attention(32 * 32, 3)
# x = torch.rand(64, 1, 32, 32)
# x = model(x)
# print(model)
# print(x)
