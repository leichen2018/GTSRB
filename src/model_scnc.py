import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #1
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv1_3 = nn.Conv2d(3, 16, kernel_size=7, padding=3)

        #2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)

        #3
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        self.drop1 = nn.Dropout2d(p=0.5)

        self.fc1 = nn.Linear(4608, 512)
        self.fc2 = nn.Linear(512, nclasses)

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
        x1 = F.relu(self.conv1_1(x))
        x2 = F.relu(self.conv1_2(x))
        x3 = F.relu(self.conv1_3(x))
        x = torch.cat((x1, x2, x3), 1)
        x = F.max_pool2d(x, 2)

        #2
        x1 = F.relu(self.conv2_1(x))
        x2 = F.relu(self.conv2_2(x))
        x = torch.cat((x1, x2), 1)
        x = F.max_pool2d(x, 2)

        #3
        x = F.relu(self.conv3(x))
        x = self.drop1(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
