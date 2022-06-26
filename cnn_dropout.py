import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 50, 50)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 64, 2, 1, padding='same')
        self.pool2 = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(1920, num_classes)

    def forward(self, x):
        # x: [batch, len] -> [batch, 1, len]
        x = x.view(x.shape[0], 1, x.shape[1])

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.dense(x)
        return x
