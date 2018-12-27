import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        #2
        self.res2a_branch1 = nn.Conv2d(64, 256, kernel_size=1)
        self.bn2a_branch1 = nn.BatchNorm2d(256)

        self.res2a_branch2a = nn.Conv2d(64, 64, kernel_size=1)
        self.bn2a_branch2a = nn.BatchNorm2d(64)
        self.res2a_branch2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2a_branch2b = nn.BatchNorm2d(64)
        self.res2a_branch2c = nn.Conv2d(64, 256, kernel_size=1)
        self.bn2a_branch2c = nn.BatchNorm2d(256)

        #3
        self.res2b_branch2a = nn.Conv2d(256, 64, kernel_size=1)
        self.bn2b_branch2a = nn.BatchNorm2d(64)
        self.res2b_branch2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b_branch2b = nn.BatchNorm2d(64)
        self.res2b_branch2c = nn.Conv2d(64, 256, kernel_size=1)
        self.bn2b_branch2c = nn.BatchNorm2d(256)

        #4
        self.res2c_branch2a = nn.Conv2d(256, 64, kernel_size=1)
        self.bn2c_branch2a = nn.BatchNorm2d(64)
        self.res2c_branch2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2c_branch2b = nn.BatchNorm2d(64)
        self.res2c_branch2c = nn.Conv2d(64, 256, kernel_size=1)
        self.bn2c_branch2c = nn.BatchNorm2d(256)

        #5
        self.res2d_branch2a = nn.Conv2d(256, 64, kernel_size=1)
        self.bn2d_branch2a = nn.BatchNorm2d(64)
        self.res2d_branch2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2d_branch2b = nn.BatchNorm2d(64)
        self.res2d_branch2c = nn.Conv2d(64, 256, kernel_size=1)
        self.bn2d_branch2c = nn.BatchNorm2d(256)

        #6
        self.res3a_branch1 = nn.Conv2d(256, 512, kernel_size=1, stride=2)
        self.bn3a_branch1 = nn.BatchNorm2d(512)

        self.res3a_branch2a = nn.Conv2d(256, 128, kernel_size=1, stride=2)
        self.bn3a_branch2a = nn.BatchNorm2d(128)
        self.res3a_branch2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3a_branch2b = nn.BatchNorm2d(128)
        self.res3a_branch2c = nn.Conv2d(128, 512, kernel_size=1)
        self.bn3a_branch2c = nn.BatchNorm2d(512)

        #7
        self.res3b_branch2a = nn.Conv2d(512, 128, kernel_size=1)
        self.bn3b_branch2a = nn.BatchNorm2d(128)
        self.res3b_branch2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3b_branch2b = nn.BatchNorm2d(128)
        self.res3b_branch2c = nn.Conv2d(128, 512, kernel_size=1)
        self.bn3b_branch2c = nn.BatchNorm2d(512)

        #8
        self.res3c_branch2a = nn.Conv2d(512, 128, kernel_size=1)
        self.bn3c_branch2a = nn.BatchNorm2d(128)
        self.res3c_branch2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3c_branch2b = nn.BatchNorm2d(128)
        self.res3c_branch2c = nn.Conv2d(128, 512, kernel_size=1)
        self.bn3c_branch2c = nn.BatchNorm2d(512)

        #9
        self.res3d_branch2a = nn.Conv2d(512, 128, kernel_size=1)
        self.bn3d_branch2a = nn.BatchNorm2d(128)
        self.res3d_branch2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3d_branch2b = nn.BatchNorm2d(128)
        self.res3d_branch2c = nn.Conv2d(128, 512, kernel_size=1)
        self.bn3d_branch2c = nn.BatchNorm2d(512)

        #10
        self.res4a_branch1 = nn.Conv2d(512, 1024, kernel_size=1, stride=2)
        self.bn4a_branch1 = nn.BatchNorm2d(1024)

        self.res4a_branch2a = nn.Conv2d(512, 256, kernel_size=1, stride=2)
        self.bn4a_branch2a = nn.BatchNorm2d(256)
        self.res4a_branch2b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4a_branch2b = nn.BatchNorm2d(256)
        self.res4a_branch2c = nn.Conv2d(256, 1024, kernel_size=1)
        self.bn4a_branch2c = nn.BatchNorm2d(1024)

        #11
        self.res4b_branch2a = nn.Conv2d(1024, 256, kernel_size=1)
        self.bn4b_branch2a = nn.BatchNorm2d(256)
        self.res4b_branch2b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4b_branch2b = nn.BatchNorm2d(256)
        self.res4b_branch2c = nn.Conv2d(256, 1024, kernel_size=1)
        self.bn4b_branch2c = nn.BatchNorm2d(1024)

        #12
        self.res4c_branch2a = nn.Conv2d(1024, 256, kernel_size=1)
        self.bn4c_branch2a = nn.BatchNorm2d(256)
        self.res4c_branch2b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4c_branch2b = nn.BatchNorm2d(256)
        self.res4c_branch2c = nn.Conv2d(256, 1024, kernel_size=1)
        self.bn4c_branch2c = nn.BatchNorm2d(1024)

        #13
        self.res4d_branch2a = nn.Conv2d(1024, 256, kernel_size=1)
        self.bn4d_branch2a = nn.BatchNorm2d(256)
        self.res4d_branch2b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4d_branch2b = nn.BatchNorm2d(256)
        self.res4d_branch2c = nn.Conv2d(256, 1024, kernel_size=1)
        self.bn4d_branch2c = nn.BatchNorm2d(1024)

        self.ave1 = nn.AvgPool2d(kernel_size=8)

        self.fc1 = nn.Linear(1024, nclasses)

    def forward(self, x):

        #1
        x = F.relu(self.bn1(self.conv1(x)))

        #2
        x1 = self.bn2a_branch1(self.res2a_branch1(x))
        x2 = F.relu(self.bn2a_branch2a(self.res2a_branch2a(x)))
        x2 = F.relu(self.bn2a_branch2b(self.res2a_branch2b(x2)))
        x2 = self.bn2a_branch2c(self.res2a_branch2c(x2))
        x2 += x1
        x2 = F.relu(x2)

        #3
        x1 = x2
        x2 = F.relu(self.bn2b_branch2a(self.res2b_branch2a(x2)))
        x2 = F.relu(self.bn2b_branch2b(self.res2b_branch2b(x2)))
        x2 = self.bn2b_branch2c(self.res2b_branch2c(x2))
        x2 += x1
        x2 = F.relu(x2)

        #4
        x1 = x2
        x2 = F.relu(self.bn2c_branch2a(self.res2c_branch2a(x2)))
        x2 = F.relu(self.bn2c_branch2b(self.res2c_branch2b(x2)))
        x2 = self.bn2c_branch2c(self.res2c_branch2c(x2))
        x2 += x1
        x2 = F.relu(x2)

        #5
        x1 = x2
        x2 = F.relu(self.bn2d_branch2a(self.res2d_branch2a(x2)))
        x2 = F.relu(self.bn2d_branch2b(self.res2d_branch2b(x2)))
        x2 = self.bn2d_branch2c(self.res2d_branch2c(x2))
        x2 += x1
        x2 = F.relu(x2)

        #6
        x = x2
        x1 = self.bn3a_branch1(self.res3a_branch1(x))
        x2 = F.relu(self.bn3a_branch2a(self.res3a_branch2a(x)))
        x2 = F.relu(self.bn3a_branch2b(self.res3a_branch2b(x2)))
        x2 = self.bn3a_branch2c(self.res3a_branch2c(x2))
        x2 += x1
        x2 = F.relu(x2)

        #7
        x1 = x2
        x2 = F.relu(self.bn3b_branch2a(self.res3b_branch2a(x2)))
        x2 = F.relu(self.bn3b_branch2b(self.res3b_branch2b(x2)))
        x2 = self.bn3b_branch2c(self.res3b_branch2c(x2))
        x2 += x1
        x2 = F.relu(x2)

        #8
        x1 = x2
        x2 = F.relu(self.bn3c_branch2a(self.res3c_branch2a(x2)))
        x2 = F.relu(self.bn3c_branch2b(self.res3c_branch2b(x2)))
        x2 = self.bn3c_branch2c(self.res3c_branch2c(x2))
        x2 += x1
        x2 = F.relu(x2)

        #9
        x1 = x2
        x2 = F.relu(self.bn3d_branch2a(self.res3d_branch2a(x2)))
        x2 = F.relu(self.bn3d_branch2b(self.res3d_branch2b(x2)))
        x2 = self.bn3d_branch2c(self.res3d_branch2c(x2))
        x2 += x1
        x2 = F.relu(x2)

        #10
        x = x2
        x1 = self.bn4a_branch1(self.res4a_branch1(x))
        x2 = F.relu(self.bn4a_branch2a(self.res4a_branch2a(x)))
        x2 = F.relu(self.bn4a_branch2b(self.res4a_branch2b(x2)))
        x2 = self.bn4a_branch2c(self.res4a_branch2c(x2))
        x2 += x1
        x2 = F.relu(x2)

        #11
        x1 = x2
        x2 = F.relu(self.bn4b_branch2a(self.res4b_branch2a(x2)))
        x2 = F.relu(self.bn4b_branch2b(self.res4b_branch2b(x2)))
        x2 = self.bn4b_branch2c(self.res4b_branch2c(x2))
        x2 += x1
        x2 = F.relu(x2)

        #12
        x1 = x2
        x2 = F.relu(self.bn4c_branch2a(self.res4c_branch2a(x2)))
        x2 = F.relu(self.bn4c_branch2b(self.res4c_branch2b(x2)))
        x2 = self.bn4c_branch2c(self.res4c_branch2c(x2))
        x2 += x1
        x2 = F.relu(x2)

        #13
        x1 = x2
        x2 = F.relu(self.bn4d_branch2a(self.res4d_branch2a(x2)))
        x2 = F.relu(self.bn4d_branch2b(self.res4d_branch2b(x2)))
        x2 = self.bn4d_branch2c(self.res4d_branch2c(x2))
        x2 += x1
        x2 = F.relu(x2)

        x2 = self.ave1(x2)

        x = x2.view(-1, 1024)

        x = F.relu(self.fc1(x))

        return F.log_softmax(x, dim=1)
