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
    def __init__(self, dataset, r, mode, transform=None, noise_mode=None,
                 noise_file='', random_state=0, epoch=-1):
        self.r = r
        self.mode = mode
        self.transform = transform
        self.dataset = 'jn'
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise
        self.noise_mode = noise_mode

        if self.mode == 'test':
            if dataset == 'jn':
                self.test_data, self.test_label = np.load('./JN_dataset/X_test.npy'), np.load('./JN_dataset/y_test.npy')
                self.test_label = self.test_label.flatten().tolist()
        elif self.mode == 'train':
            if dataset == 'jn':
                train_data, train_label = np.load('./JN_dataset/X_train.npy'), np.load('./JN_dataset/y_train.npy')
                train_label = train_label.flatten().tolist()

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
                            if dataset == 'jn':
                                noiselabel = random.randint(0, 9)
                            noise_label.append(noiselabel)
                        elif noise_mode == 'asym':
                            noiselabel = self.transition[train_label[i]]
                            noise_label.append(noiselabel)
                    else:
                        noise_label.append(train_label[i])
                print("save noisy labels to %s ..." % noise_file)
                # print(noise_label)
                json.dump(noise_label, open(noise_file, "w"))
            self.train_data = train_data
            self.noise_label = noise_label
            self.noise_or_not = np.array(self.noise_label) == np.array(train_label)

    def __getitem__(self, index):
        if self.mode == 'train':
            series, target = self.train_data[index], self.noise_label[index]
        elif self.mode == 'test':
            series, target = self.test_data[index], self.test_label[index]

        if self.transform is not None:
            series = self.transform(series)

        return series, target, index

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.mode)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class ToTensor:
    def __call__(self, x):
        return torch.from_numpy(x).to(dtype=torch.float32)
