import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 50, 50, padding='same')
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 64, 2, 1, padding='same')
        self.pool2 = nn.MaxPool1d(2)
        self.dense = nn.Linear(1920, 10)

    def forward(self, x):
        # x: [batch, len]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = torch.flatten(x)
        x = self.dense(x)
        return x