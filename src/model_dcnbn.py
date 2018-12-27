import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv11_bn = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv12_bn = nn.BatchNorm2d(64)
        self.conv13 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv13_bn = nn.BatchNorm2d(64)
        self.conv21 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.conv21_bn = nn.BatchNorm2d(256)
        self.conv22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv22_bn = nn.BatchNorm2d(256)
        self.conv23 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv23_bn = nn.BatchNorm2d(256)
        self.conv24 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv24_bn = nn.BatchNorm2d(256)
        self.conv25 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv25_bn = nn.BatchNorm2d(256)
        self.conv26 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv26_bn = nn.BatchNorm2d(256)
        self.conv27 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv27_bn = nn.BatchNorm2d(256)
        self.conv28 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv28_bn = nn.BatchNorm2d(256)
        self.conv31 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv31_bn = nn.BatchNorm2d(512)
        self.conv32 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv32_bn = nn.BatchNorm2d(512)
        self.conv33 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv33_bn = nn.BatchNorm2d(512)
        self.conv34 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv34_bn = nn.BatchNorm2d(512)
        self.conv35 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv35_bn = nn.BatchNorm2d(512)
        self.conv36 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv36_bn = nn.BatchNorm2d(512)
        self.conv37 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv37_bn = nn.BatchNorm2d(512)
        self.conv38 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv38_bn = nn.BatchNorm2d(512)

        self.ave1 = nn.AvgPool2d(kernel_size=8)

        self.fc1 = nn.Linear(512, nclasses)

    def forward(self, x):
        x = F.relu(self.conv11_bn(self.conv11(x)))
        x = F.relu(self.conv12_bn(self.conv12(x)))
        x = F.relu(self.conv13_bn(self.conv13(x)))
        x = F.max_pool2d(self.conv21_bn(self.conv21(x)), 2)
        x = F.relu(x)
        x = F.relu(self.conv22_bn(self.conv22(x)))
        x = F.relu(self.conv23_bn(self.conv23(x)))
        x = F.relu(self.conv24_bn(self.conv24(x)))
        x = F.relu(self.conv25_bn(self.conv25(x)))
        x = F.relu(self.conv26_bn(self.conv26(x)))
        x = F.relu(self.conv27_bn(self.conv27(x)))
        x = F.relu(self.conv28_bn(self.conv28(x)))
        x = F.max_pool2d(self.conv31_bn(self.conv31(x)), 2)
        x = F.relu(x)
        x = F.relu(self.conv32_bn(self.conv32(x)))
        x = F.relu(self.conv33_bn(self.conv33(x)))
        x = F.relu(self.conv34_bn(self.conv34(x)))
        x = F.relu(self.conv35_bn(self.conv35(x)))
        x = F.relu(self.conv36_bn(self.conv36(x)))
        x = F.relu(self.conv37_bn(self.conv37(x)))
        x = F.relu(self.conv38_bn(self.conv38(x)))

        x = self.ave1(x)

        x = x.view(-1, 512)

        x = F.relu(self.fc1(x))

        return F.log_softmax(x, dim=1)
