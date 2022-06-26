import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.p = 0.5
        self.conv1 = nn.Conv1d(1, 64, 50, 50)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, 2, 1, padding='same')
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, 2, 1, padding='same')
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(self.p)
        self.dense = nn.Linear(256, num_classes)

    def forward(self, x, weights=None, get_feat=None):
        # x: [batch, len] -> [batch, 1, len]
        x = x.view(x.shape[0], 1, x.shape[1])

        if weights is None:
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool1(x)  # x: [batch, 64, 60]

            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool2(x)  # x: [batch, 128, 30]

            x = self.conv3(x)
            x = F.relu(x)
            x = self.gap(x)  # x: [batch, 256, 1]

            feat = x.view(x.shape[0], -1)
            x = self.dropout(feat)
            x = self.dense(x)
            if get_feat:
                return x, feat
            else:
                return x
        else:
            x = F.conv1d(x, weights['conv1.weight'], stride=50)
            x = F.threshold(x, 0, 0, inplace=True)
            x = F.max_pool1d(x, kernel_size=2)

            x = F.conv1d(x, weights['conv2.weight'], stride=1, padding='same')
            x = F.threshold(x, 0, 0, inplace=True)
            x = F.max_pool1d(x, kernel_size=2)

            x = F.conv1d(x, weights['conv3.weight'], stride=1, padding='same')
            x = F.threshold(x, 0, 0, inplace=True)
            x = F.adaptive_avg_pool1d(x, output_size=1)

            x = x.view(x.size(0), -1)
            x = F.dropout(x, p=self.p, training=True, inplace=True)
            x = F.linear(x, weights['dense.weight'], weights['dense.bias'])
            return x
