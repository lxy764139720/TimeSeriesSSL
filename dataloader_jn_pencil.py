from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import json
import os
import torch
from torchnet.meter import AUCMeter
import wandb


class jn_dataset(Dataset):
    def __init__(self, r, root_dir, mode, noise_mode, noise_file='', transform=None):
        self.transform = transform
        self.mode = mode  # training set or test set
        self.r = r
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise

        # now load the picked numpy arrays
        if self.mode == 'train':
            train_data, train_label = np.load(root_dir + '/X_train.npy'), np.load(root_dir + '/y_train.npy')
            train_labels = train_label.flatten().tolist()

            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file, "r"))
            else:  # inject noise
                noise_label = []
                idx = list(range(len(train_data)))
                random.shuffle(idx)
                num_noise = int(self.r * len(train_data))
                noise_idx = idx[:num_noise]
                for i in range(len(train_data)):
                    if i in noise_idx:
                        if noise_mode == 'sym':
                            noiselabel = random.randint(0, 9)
                            noise_label.append(noiselabel)
                        elif noise_mode == 'asym':
                            noiselabel = self.transition[train_labels[i]]
                            noise_label.append(noiselabel)
                    else:
                        noise_label.append(train_labels[i])
                print("save noisy labels to %s ..." % noise_file)
                # print(noise_label)
                json.dump(noise_label, open(noise_file, "w"))
            self.train_data = train_data
            self.train_labels = noise_label
        elif self.mode == 'test':
            self.test_data, self.test_label = np.load(root_dir + '/X_test.npy'), np.load(root_dir + '/y_test.npy')
            self.test_labels = self.test_label.flatten().tolist()

    def __getitem__(self, index):
        if self.mode == 'train':
            series, target = self.train_data[index], self.train_labels[index]
            series = self.transform(series)
            return series, target, index
        elif self.mode == 'test':
            series, target = self.test_data[index], self.test_labels[index]
            series = self.transform(series)
            return series, target

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'test':
            return len(self.test_data)


class Jitter:
    def __init__(self, sigma=0.03):
        self.sigma = sigma

    def __call__(self, x):
        return x + np.random.normal(loc=0., scale=self.sigma, size=x.shape)


class Scaling:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, x):
        factor = np.random.normal(loc=1., scale=self.sigma, size=x.shape)
        return np.multiply(x, factor)


class ToTensor:
    def __call__(self, x):
        return torch.from_numpy(x).to(dtype=torch.float32)


class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x