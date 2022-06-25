import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes=10, p=0.5):
        super(Model, self).__init__()
        self.p = p
        self.conv1 = nn.Conv1d(1, 64, 50, 50)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 64, 2, 1, padding='same')
        self.pool2 = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(self.p)
        self.dense = nn.Linear(1920, num_classes)

    def forward(self, x, weights=None, get_feat=None):
        # x: [batch, len] -> [batch, 1, len]
        x = x.view(x.shape[0], 1, x.shape[1])

        if weights is None:
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool1(x)

            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool2(x)

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

            x = x.view(x.size(0), -1)
            x = F.dropout(x, p=self.p, training=True, inplace=True)
            x = F.linear(x, weights['dense.weight'], weights['dense.bias'])
            return x
