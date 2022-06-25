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


def load_fault_data():
    """
    Load fault_data of bearing from a sample frequency of 12k
    including rolling element, inner race and outer race fault with 7-mil
    14-mil,21-mil diameter, we use DE variable for working
    Reshape the raw data in a format of (samples,6000)
    """

    path = './CRWU_dataset/12kDriveEnd'
    files = os.listdir(path)
    temp = []
    label = []
    for mat in files:
        if ('28' not in mat) and ('.mat' in mat):
            temp1 = sio.loadmat(os.path.join(path, mat))
            for key in temp1.keys():
                if 'DE' in key:
                    temp.append(temp1[key][:120000])
                    if 'B' in mat:
                        if '07' in mat:
                            label.append([0] * 20)
                        if '14' in mat:
                            label.append([1] * 20)
                        if '21' in mat:
                            label.append([2] * 20)
                    if 'IR' in mat:
                        if '07' in mat:
                            label.append([3] * 20)
                        if '14' in mat:
                            label.append([4] * 20)
                        if '21' in mat:
                            label.append([5] * 20)
                    if 'OR' in mat:
                        if '07' in mat:
                            label.append([6] * 20)
                        if '14' in mat:
                            label.append([7] * 20)
                        if '21' in mat:
                            label.append([8] * 20)
    temp = np.asarray(temp)
    data1 = temp.reshape((-1, 6000))
    label1 = np.asarray(label)
    label1 = label1.reshape((-1, 1))
    return data1, label1


def load_normal_data():
    """
    Load normal_data of bearing
    we use DE variable for working
    Reshape the raw data in a format of (samples,6000)
    """

    path = './CRWU_dataset/Normal_Baseline_Data'
    files = os.listdir(path)
    temp = []
    label2 = []
    for mat in files:
        temp1 = sio.loadmat(os.path.join(path, mat))
        for key in temp1.keys():
            if 'DE' in key:
                if 240000 < len(temp1[key]) < 480000:
                    temp.append(temp1[key][:240000])
                if len(temp1[key]) > 480000:
                    temp.append(temp1[key][:480000])
    temp2 = np.concatenate((temp[0], temp[1], temp[2], temp[3]))
    data2 = temp2.reshape((-1, 6000))
    label2 = np.ones((data2.shape[0], 1)) * 9
    return data2, label2


def load_crwu_data():
    """
    combine all data to be a set, split train set and test set
    """

    data1, label1 = load_fault_data()
    data2, label2 = load_normal_data()
    data = np.concatenate((data1, data2))
    label = np.concatenate((label1, label2)).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=42, stratify=label)
    return X_train, X_test, y_train, y_test


class crwu_dataset(Dataset):
    def __init__(self, r, root_dir, mode, noise_mode, noise_file='', transform=None):
        self.transform = transform
        self.mode = mode  # training set or test set
        self.r = r
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise

        # now load the picked numpy arrays
        if self.mode == 'train':
            train_data, _, train_label, _ = load_crwu_data()
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
            _, self.test_data, _, self.test_label = load_crwu_data()
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