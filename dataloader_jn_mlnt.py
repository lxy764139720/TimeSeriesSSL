from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import os
import json


class jn_dataset(Dataset):
    def __init__(self, dataset, r, noise_mode, root_dir, noise_file, transform, mode):
        self.r = r
        self.mode = mode
        self.transform = transform
        self.dataset = 'jn'
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise
        self.noise_mode = noise_mode

        if self.mode == 'test':
            if dataset == 'jn':
                self.test_data, self.test_labels = np.load('./JN_dataset/X_test.npy'), np.load('./JN_dataset/y_test.npy')
                self.test_labels = self.test_labels.flatten().tolist()
        else:
            if dataset == 'jn':
                train_data, train_labels = np.load('./JN_dataset/X_train.npy'), np.load('./JN_dataset/y_train.npy')
                train_labels = train_labels.flatten().tolist()

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
                            noiselabel = self.transition[train_labels[i]]
                            noise_label.append(noiselabel)
                    else:
                        noise_label.append(train_labels[i])
                print("save noisy labels to %s ..." % noise_file)
                # print(noise_label)
                json.dump(noise_label, open(noise_file, "w"))
            self.train_data = train_data
            self.train_labels = noise_label

    def __getitem__(self, index):
        if self.mode == 'train':
            series, target = self.train_data[index], self.train_labels[index]
            series = self.transform(series)
            return series, target
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


class jn_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, noise_file):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_train = ComposeTransform([
            Jitter(),
            Scaling(),
            ToTensor(),
        ])
        self.transform_test = ComposeTransform([
            ToTensor(),
        ])

    def run(self):
        train_dataset = jn_dataset(transform=self.transform_train, mode='train', r=self.r, noise_mode=self.noise_mode,
                                   root_dir=self.root_dir, noise_file=self.noise_file, dataset=self.dataset)
        test_dataset = jn_dataset(transform=self.transform_test, mode='test', r=self.r, noise_mode=self.noise_mode,
                                  root_dir=self.root_dir, noise_file=self.noise_file, dataset=self.dataset)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers)
        return train_loader, test_loader
