import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.drop = nn.Dropout(config['dropout'])

        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, 10)

    def forward(self, x):
        # 784 -> 2000
        x = F.relu(self.drop(self.fc1(x)))
        # 2000 -> 2000
        x = F.relu(self.drop(self.fc2(x)))
        # 2000 -> 100
        x = self.fc5(x)
        return x