import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #1
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size=1)

        self.conv1_2 = nn.Conv2d(3, 16, kernel_size=1)

        self.conv1_3_1 = nn.Conv2d(3, 8, kernel_size=1)
        self.conv1_3_2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        self.conv1_4_1 = nn.Conv2d(3, 4, kernel_size=1)
        self.conv1_4_2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv1_4_3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        self.bn_1 = nn.BatchNorm2d(64)

        #2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.conv3_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(2304, 100)
        self.fc2 = nn.Linear(100, nclasses)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 2 * 2, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.Tensor([1, 0, 0, 0, 1, 0]).cuda())


    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 2 * 2)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.stn(x)

        #1
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(F.max_pool2d(x, kernel_size=3, stride=1, padding=1))
        x3 = self.conv1_3_2(F.relu(self.conv1_3_1(x)))
        x4 = self.conv1_4_3(F.relu(self.conv1_4_2(F.relu(self.conv1_4_1(x)))))
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.bn_1(F.relu(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        #2
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
