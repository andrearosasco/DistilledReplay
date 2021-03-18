import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_in')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.uniform_()
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=1, stride=2)
        self.conv2_bn = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=1, stride=2)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=1, stride=2)
        self.conv4_bn = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=1, stride=2)
        self.conv5_bn = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 2 * 2, 2000)
        self.fc2 = nn.Linear(2000, 2000)
        self.fc3 = nn.Linear(2000, config['n_classes'])

        self.apply(initialize_weights)

    def forward(self, x):
        # [3, 32, 32] -> [16, 32, 32]
        x = F.relu(self.conv1_bn(self.conv1(x)))
        # [16, 32, 32] -> [32, 16, 16]
        x = F.relu(self.conv2_bn(self.conv2(x)))
        # [32, 16, 16] -> [64, 8, 8]
        x = F.relu(self.conv3_bn(self.conv3(x)))
        # [64, 8, 8] -> [128, 4, 4]
        x = F.relu(self.conv4_bn(self.conv4(x)))
        # [128, 4, 4] -> [256, 2, 2]
        x = F.relu(self.conv5_bn(self.conv5(x)))
        # [128, 2, 2] -> [512]
        x = x.view(-1, 256 * 2 * 2)
        # 1024 -> 2000
        x = F.relu(F.dropout(self.fc1(x), 0.5))
        # 2000 -> 2000
        x = F.relu(F.dropout(self.fc2(x), 0.5))
        # 2000 -> 100
        x = self.fc3(x)
        return x